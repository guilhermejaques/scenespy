"""Production-oriented Qt workspace for non-destructive Scenespy projects."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QFontDatabase, QIcon,
                           QKeySequence, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGraphicsItem, QGraphicsLineItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSlider,
    QSpinBox, QSplitter, QStatusBar, QToolButton, QVBoxLayout, QWidget,
)

from .media import probe_media
from .project import CutEvent, FaceEvent, MediaItem, ScenespyProject
from .face_engine import FaceDetectionEngine
from .scene_engine import SceneEngine
from .shared import (ACCEL_OPTIONS, ACCENT, ALLOWED_VIDEO_EXTENSIONS, BG_CARD, BG_MAIN,
                     BG_PANEL, BORDER_SOFT2, PROFILES, SUCCESS, TEXT_MAIN, TEXT_MUTED,
                     USER_DATA_DIR, detect_available_encoder_accel, detect_available_inference_accel,
                     terminate_all_child_processes)


def timecode(seconds):
    total = max(0, int(round(float(seconds or 0) * 1000)))
    hours, total = divmod(total, 3_600_000)
    minutes, total = divmod(total, 60_000)
    secs, millis = divmod(total, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def transport_timecode(seconds):
    return timecode(seconds).split(".", 1)[0]


def parse_timecode(value):
    try:
        parts = value.strip().replace(",", ".").split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (TypeError, ValueError):
        return None


class ScrubSlider(QSlider):
    """Seek slider whose entire groove is directly draggable."""
    def _set_from_mouse(self, event):
        width = max(1, self.width() - 14)
        ratio = max(0.0, min(1.0, (event.position().x() - 7) / width))
        value = round(self.minimum() + ratio * (self.maximum() - self.minimum()))
        self.setValue(value)
        self.sliderMoved.emit(value)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(True)
            self._set_from_mouse(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            self._set_from_mouse(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isSliderDown():
            self._set_from_mouse(event)
            self.setSliderDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class VideoBackdrop(QWidget):
    """Absolute-black canvas shown behind the native video surface."""
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        painter.end()
        super().paintEvent(event)


class Task(QThread):
    progress = Signal(float, str)
    result = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, work, parent=None):
        super().__init__(parent)
        self.work = work
        self.cancel_callback = None

    def register_cancel(self, callback):
        self.cancel_callback = callback
        if self.isInterruptionRequested() and callback:
            callback()

    def cancel(self):
        self.requestInterruption()
        if self.cancel_callback:
            try:
                self.cancel_callback()
            except Exception:
                pass

    def run(self):
        try:
            result = self.work(self)
            if self.isInterruptionRequested():
                self.cancelled.emit()
            else:
                self.result.emit(result)
        except Exception:
            if self.isInterruptionRequested():
                self.cancelled.emit()
            else:
                self.failed.emit(traceback.format_exc())


class ProgressAdapter:
    def __init__(self, task):
        self.task, self.value, self.label = task, 0.0, ""

    def after(self, _delay, callback): callback()
    def reset(self): self.update(0)
    def update(self, value):
        self.value = float(value)
        self.task.progress.emit(self.value, self.label)
    def set_status(self, text, color=None):
        self.label = str(text or "")
        self.task.progress.emit(self.value, self.label)
    def mark_finished(self, text="Complete"):
        self.task.progress.emit(1.0, text)


class LogAdapter:
    def __init__(self, task): self.task = task
    def after(self, _delay, callback): callback()
    def write_status(self, detected=None, cut=None, eta=None):
        self.task.progress.emit(-1.0, f"Detected: {detected}" if detected is not None else "Analyzing")
    def append_message(self, text, kind="info"): self.task.progress.emit(-1.0, str(text))
    def write_finished(self, text): self.task.progress.emit(1.0, str(text))


class CutHandle(QGraphicsLineItem):
    def __init__(self, event, timeline):
        super().__init__(0, 0, 0, 64)
        self.event, self.timeline = event, timeline
        self.setPen(QPen(QColor("#94a3b8"), 2))
        self.setZValue(20)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        cap = QGraphicsRectItem(-5, -3, 10, 13, self)
        cap.setBrush(QColor("#94a3b8"))
        cap.setPen(QPen(QColor("#e2e8f0"), 1))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            point = QPointF(value)
            frame = self.timeline.frame_for_x(point.x())
            lower, upper = self.timeline.bounds_for(self.event)
            return QPointF(self.timeline.x_for_frame(max(lower, min(frame, upper))), 0)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self.scene():
            self.timeline.handle_moved(self)
        return super().itemChange(change, value)


class PlayheadItem(QGraphicsItem):
    """Wide-hit-area playhead which only seeks after a deliberate drag."""
    def __init__(self, timeline):
        super().__init__()
        self.timeline = timeline
        self.dragging = False
        self.setZValue(40)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

    def boundingRect(self):
        return QRectF(-10, 0, 20, 170)

    def shape(self):
        path = QPainterPath()
        path.addRect(-9, 0, 18, 170)
        return path

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#2d8cff")
        painter.setPen(QPen(color, 2))
        painter.drawLine(0, 12, 0, 170)
        painter.setPen(QPen(QColor("#bfdbfe"), 1))
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([QPointF(-8, 0), QPointF(8, 0), QPointF(0, 13)]))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            point = QPointF(value)
            return QPointF(max(0, min(point.x(), self.timeline.content_width)), 0)
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        self.dragging = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.dragging = False
        if self.timeline.media:
            self.timeline.seekRequested.emit(self.x() / self.timeline.pixels_per_second)

class FaceChip(QGraphicsRectItem):
    def __init__(self, event, center_x):
        super().__init__(center_x - 22, 100, 44, 30)
        self.event = event
        self.hovered = False
        self.thumbnail = QPixmap(event.crop_file) if event.crop_file else QPixmap()
        if not self.thumbnail.isNull():
            scaled = self.thumbnail.scaled(
                40, 26, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation)
            left = max(0, (scaled.width() - 40) // 2)
            top = max(0, (scaled.height() - 26) // 2)
            self.thumbnail = scaled.copy(left, top, 40, 26)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Face · click to inspect the best frame")
        self.setZValue(12)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(1.0 if self.event.enabled else .38)
        active = self.isSelected() or self.hovered
        border = QColor("#ffffff" if active else "#c4b5fd")
        painter.setPen(QPen(border, 2 if active else 1))
        painter.setBrush(QColor("#69569c" if self.hovered else "#4c3f72"))
        painter.drawRoundedRect(self.rect(), 5, 5)
        inner = self.rect().adjusted(2, 2, -2, -2)
        if not self.thumbnail.isNull():
            clip = QPainterPath()
            clip.addRoundedRect(inner, 4, 4)
            painter.save()
            painter.setClipPath(clip)
            painter.drawPixmap(inner.toRect(), self.thumbnail)
            painter.restore()
        else:
            center = inner.center()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#ddd6fe"))
            painter.drawEllipse(QPointF(center.x(), center.y() - 5), 5, 5)
            painter.drawRoundedRect(QRectF(center.x() - 9, center.y() + 1, 18, 9), 5, 5)

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)


class SceneSegmentItem(QGraphicsRectItem):
    def __init__(self, rect, color, tooltip):
        super().__init__(*rect)
        self.base_color = QColor(color)
        self.direct_hover = False
        self.linked_hover = False
        self.setAcceptHoverEvents(True)
        self.setToolTip(tooltip)
        self._refresh()

    def _refresh(self):
        active = self.direct_hover or self.linked_hover
        brush = QBrush(self.base_color.lighter(120) if active else self.base_color)
        brush.setStyle(Qt.BrushStyle.Dense7Pattern)
        self.setBrush(brush)
        self.setPen(QPen(QColor("#f8fafc" if active else "#50545b"), 1))
        self.setZValue(1)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if not (self.direct_hover or self.linked_hover):
            return
        inner = self.rect().adjusted(1, 1, -1, -1)
        glow = QLinearGradient(inner.topLeft(), inner.bottomLeft())
        glow.setColorAt(0.0, QColor(255, 255, 255, 48))
        glow.setColorAt(0.42, QColor(255, 255, 255, 15))
        glow.setColorAt(1.0, QColor(255, 255, 255, 5))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawRect(inner)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 38), 3))
        painter.drawRect(self.rect().adjusted(2, 2, -2, -2))

    def set_linked_hover(self, active):
        self.linked_hover = bool(active)
        self._refresh()

    def hoverEnterEvent(self, event):
        self.direct_hover = True
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.direct_hover = False
        self.unsetCursor()
        self._refresh()
        super().hoverLeaveEvent(event)


class ClipPreviewChip(QGraphicsRectItem):
    def __init__(self, start_frame, end_frame, index, x, timeline, segment):
        super().__init__(x - 20, 101, 40, 25)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.index = index
        self.timeline = timeline
        self.segment = segment
        self.hovered = False
        self.setAcceptHoverEvents(True)
        self.setBrush(QColor("#244d73"))
        self.setPen(QPen(QColor("#60a5fa"), 1))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"Preview clip {index:02d}")
        self.setZValue(15)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#ffffff" if self.hovered else "#60a5fa"), 1))
        painter.setBrush(QColor("#327fc4" if self.hovered else "#244d73"))
        painter.drawRoundedRect(self.rect(), 4, 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ffffff" if self.hovered else "#dbeafe"))
        painter.drawPolygon(QPolygonF([QPointF(-4, -6), QPointF(6, 0), QPointF(-4, 6)]).translated(self.rect().center()))

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.segment.set_linked_hover(True)
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.segment.set_linked_hover(False)
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if self.timeline.media:
            fps = self.timeline.media.fps
            self.timeline.segmentActivated.emit(self.start_frame / fps, self.end_frame / fps)
            event.accept()
            return
        super().mousePressEvent(event)


class TimelineView(QGraphicsView):
    seekRequested = Signal(float)
    cutSelected = Signal(object)
    segmentActivated = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.media = None
        self.content_width = 1000.0
        self.pixels_per_second = 70.0
        self.playhead = None
        self.setMinimumHeight(205)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor(BG_MAIN))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scene().selectionChanged.connect(self._selection_changed)

    def set_media(self, media):
        self.media = media
        self.rebuild()

    def rebuild(self):
        self.scene().clear()
        if not self.media:
            text = self.scene().addSimpleText("Import a video to begin")
            text.setBrush(QColor(TEXT_MUTED))
            return
        self.content_width = max(self.viewport().width() - 20,
                                 self.media.duration * self.pixels_per_second)
        self.scene().setSceneRect(0, 0, self.content_width, 175)
        self._draw_ruler()
        boundaries = [0, *(event.frame for event in sorted(self.media.cuts, key=lambda cut: cut.frame)),
                      self.media.total_frames]
        segment_colors = (QColor("#383b40"), QColor("#30343a"))
        segment_height = 57
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
            x = self.x_for_frame(start)
            width = max(1, self.x_for_frame(end) - x)
            tooltip = (f"Scene {index:02d} · {timecode(start / self.media.fps)} – "
                       f"{timecode(end / self.media.fps)}")
            segment = SceneSegmentItem(
                (x, 35, width, segment_height), segment_colors[(index - 1) % 2], tooltip)
            self.scene().addItem(segment)
            header = QGraphicsRectItem(x, 35, width, 5)
            header.setBrush(QColor("#315f8d" if index % 2 else "#3c72a6"))
            header.setPen(Qt.PenStyle.NoPen)
            header.setZValue(2)
            self.scene().addItem(header)
            if width >= 64:
                label = self.scene().addSimpleText(f"{index:02d}")
                label.setBrush(QColor("#cbd5e1"))
                label.setPos(x + 8, 48)
                label.setZValue(3)
            if self.media.mode != "faces":
                self.scene().addItem(ClipPreviewChip(
                    start, end, index, x + width / 2, self, segment))
        name = self.scene().addSimpleText(Path(self.media.path).name)
        name.setFont(QFont("JetBrains Mono", 8))
        name.setBrush(QColor(TEXT_MUTED))
        name.setPos(12, 68)
        name.setZValue(2)
        for event in self.media.cuts:
            handle = CutHandle(event, self)
            handle.setPos(self.x_for_frame(event.frame), 28)
            handle.setToolTip(f"Cut · {timecode(event.frame / self.media.fps)}")
            self.scene().addItem(handle)
        if self.media.mode != "faces":
            preview_track = QGraphicsRectItem(0, 98, self.content_width, 31)
            preview_track.setBrush(QColor("#20252b"))
            preview_track.setPen(QPen(QColor(BORDER_SOFT2)))
            preview_track.setZValue(4)
            self.scene().addItem(preview_track)
            for item in self.scene().items():
                if isinstance(item, ClipPreviewChip):
                    item.setZValue(15)
        if self.media.faces:
            face_y = 136 if self.media.mode != "faces" else 100
            face_track = QGraphicsRectItem(0, face_y - 2, self.content_width, 32)
            face_track.setBrush(QColor("#29272e"))
            face_track.setPen(QPen(QColor(BORDER_SOFT2)))
            self.scene().addItem(face_track)
            for event in self.media.faces:
                center_x = self.x_for_frame(event.best_frame)
                chip = FaceChip(event, center_x)
                chip.setY(face_y - 100)
                self.scene().addItem(chip)
        self.playhead = PlayheadItem(self)
        self.scene().addItem(self.playhead)

    def _draw_ruler(self):
        candidates = (.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)
        step = next((value for value in candidates if value * self.pixels_per_second >= 72), candidates[-1])
        moment = 0.0
        minor_step = step / 5
        minor = 0.0
        while minor <= self.media.duration + .001:
            x = minor * self.pixels_per_second
            tick = QGraphicsLineItem(x, 25, x, 31)
            tick.setPen(QPen(QColor("#454b54"), 1))
            self.scene().addItem(tick)
            minor += minor_step
        while moment <= self.media.duration + .001:
            x = moment * self.pixels_per_second
            tick = QGraphicsLineItem(x, 18, x, 31)
            tick.setPen(QPen(QColor("#6b7280"), 1))
            self.scene().addItem(tick)
            label = self.scene().addSimpleText(timecode(moment)[:-4])
            label.setBrush(QColor(TEXT_MUTED))
            label.setPos(x + 4, 4)
            moment += step

    def x_for_frame(self, frame):
        return frame / max(1, self.media.total_frames) * self.content_width

    def frame_for_x(self, x):
        return round(max(0, min(x, self.content_width)) / max(1, self.content_width) * self.media.total_frames)

    def bounds_for(self, event):
        frames = sorted(cut.frame for cut in self.media.cuts if cut.event_id != event.event_id)
        lower = max((frame + 1 for frame in frames if frame < event.frame), default=1)
        upper = min((frame - 1 for frame in frames if frame > event.frame),
                    default=self.media.total_frames - 1)
        return lower, upper

    def handle_moved(self, handle):
        handle.event.frame = max(1, min(self.media.total_frames - 1, self.frame_for_x(handle.x())))
        handle.setToolTip(f"Cut · {timecode(handle.event.frame / self.media.fps)}")

    def set_position(self, seconds):
        if self.playhead and self.media and not self.playhead.dragging:
            self.playhead.setX(max(0, min(seconds * self.pixels_per_second, self.content_width)))

    def _selection_changed(self):
        selected = self.scene().selectedItems()
        item = selected[0] if selected else None
        self.cutSelected.emit(item.event if isinstance(item, (CutHandle, FaceChip)) else None)

    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and not isinstance(item, (CutHandle, FaceChip, ClipPreviewChip)) and self.media:
            x = self.mapToScene(event.position().toPoint()).x()
            self.seekRequested.emit(max(0, min(self.media.duration, x / self.pixels_per_second)))
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        if self.media and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.pixels_per_second = max(8, min(500, self.pixels_per_second *
                                                (1.18 if event.angleDelta().y() > 0 else .85)))
            self.rebuild()
            return
        super().wheelEvent(event)


class ScenespyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = ScenespyProject()
        self.current_media: MediaItem | None = None
        self.selected_cut = None
        self.task = None
        self.preview_end_ms = None
        self.preview_start_ms = None
        self.preview_armed = False
        self.available_accel = detect_available_encoder_accel() | detect_available_inference_accel()
        self.setWindowTitle("Scenespy")
        icon = Path(__file__).parent / "assets" / "images" / "x.png"
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))
        self.resize(1440, 900)
        self.setMinimumSize(1080, 700)
        self._build_player()
        self._build_ui()
        self._build_actions()
        self._apply_theme()
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    def _build_player(self):
        self.audio = QAudioOutput(self)
        self.audio.setVolume(.8)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio)
        self.video = QVideoWidget()
        self.video.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.video.setStyleSheet("background: transparent;")
        self.video.setMinimumSize(500, 280)
        self.video.hide()
        self.player.setVideoOutput(self.video)
        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.playbackStateChanged.connect(
            lambda state: self.play_button.setText("Ⅱ" if state == QMediaPlayer.PlaybackState.PlayingState else "▶"))
        self.player.mediaStatusChanged.connect(self._media_status_changed)

    def _build_ui(self):
        root = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = root
        root.setChildrenCollapsible(False)
        self.setCentralWidget(root)
        left = QFrame(objectName="panel")
        layout = QVBoxLayout(left)
        layout.addWidget(QLabel("MEDIA", objectName="sectionTitle"))
        self.media_list = QListWidget()
        self.media_list.currentRowChanged.connect(self._select_media)
        layout.addWidget(self.media_list)
        self.import_button = QPushButton("+ Import video")
        self.import_button.setObjectName("secondary")
        self.import_button.clicked.connect(self.import_videos)
        layout.addWidget(self.import_button)
        root.addWidget(left)

        center = QSplitter(Qt.Orientation.Vertical)
        self.workspace_splitter = center
        viewer = QFrame(objectName="viewer")
        view_layout = QVBoxLayout(viewer)
        self.video_backdrop = VideoBackdrop()
        backdrop_layout = QVBoxLayout(self.video_backdrop)
        backdrop_layout.setContentsMargins(0, 0, 0, 0)
        backdrop_layout.addWidget(self.video)
        view_layout.addWidget(self.video_backdrop, 1)
        back = QToolButton(text="−5s")
        previous = QToolButton(text="|◀")
        self.play_button = QToolButton(text="▶")
        following = QToolButton(text="▶|")
        forward = QToolButton(text="+5s")
        previous.setObjectName("transportSkip")
        following.setObjectName("transportSkip")
        back.setObjectName("transportStep")
        forward.setObjectName("transportStep")
        self.play_button.setObjectName("transportPlay")
        previous.setToolTip("Previous event")
        back.setToolTip("Back 5 seconds")
        self.play_button.setToolTip("Play / Pause")
        forward.setToolTip("Forward 5 seconds")
        following.setToolTip("Next event")
        for button in (previous, back, forward, following):
            button.setFixedSize(42, 34)
        self.play_button.setFixedSize(44, 38)
        back.clicked.connect(lambda: self.relative_seek(-5000))
        previous.clicked.connect(lambda: self.jump_event(-1))
        self.play_button.clicked.connect(self.toggle_playback)
        following.clicked.connect(lambda: self.jump_event(1))
        forward.clicked.connect(lambda: self.relative_seek(5000))
        self.seek = ScrubSlider(Qt.Orientation.Horizontal)
        self.seek.setObjectName("mediaScrubber")
        self.seek.setMinimumHeight(26)
        self.seek.sliderMoved.connect(self.seek_player_ms)
        self.clock = QLineEdit("00:00:00")
        self.clock.setObjectName("timecode")
        self.clock.setFixedWidth(92)
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clock.editingFinished.connect(self.seek_from_timecode)
        self.time_separator = QLabel("/")
        self.time_separator.setObjectName("timeSeparator")
        self.time_separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.duration_clock = QLineEdit("00:00:00")
        self.duration_clock.setObjectName("timecode")
        self.duration_clock.setFixedWidth(92)
        self.duration_clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.duration_clock.setReadOnly(True)
        self.transport_group = QFrame()
        self.transport_group.setObjectName("transportGroup")
        transport_layout = QHBoxLayout(self.transport_group)
        transport_layout.setContentsMargins(6, 5, 8, 5)
        transport_layout.setSpacing(4)
        for widget in (previous, back, self.play_button, forward, following):
            transport_layout.addWidget(widget)
        transport_layout.addSpacing(5)
        transport_layout.addWidget(self.seek, 1)
        transport_layout.addSpacing(5)
        transport_layout.addWidget(self.clock)
        transport_layout.addWidget(self.time_separator)
        transport_layout.addWidget(self.duration_clock)
        view_layout.addWidget(self.transport_group)
        center.addWidget(viewer)
        self.timeline = TimelineView()
        self.timeline.seekRequested.connect(lambda sec: self.seek_player_ms(round(sec * 1000)))
        self.timeline.cutSelected.connect(self.show_cut)
        self.timeline.segmentActivated.connect(self.play_segment)
        center.addWidget(self.timeline)
        center.setSizes([650, 210])
        root.addWidget(center)

        right = QFrame(objectName="panel")
        layout = QVBoxLayout(right)
        layout.addWidget(QLabel("ANALYSIS", objectName="sectionTitle"))
        for title, widget in self._analysis_fields():
            layout.addWidget(QLabel(title, objectName="muted"))
            layout.addWidget(widget)
        self.analyze_button = QPushButton("Analyze video", objectName="primary")
        self.analyze_button.clicked.connect(self.toggle_analysis)
        layout.addWidget(self.analyze_button)
        layout.addSpacing(16)
        layout.addWidget(QLabel("SELECTION", objectName="sectionTitle"))
        self.selection_title = QLabel("No event selected")
        self.selection_detail = QLabel("Select a marker to inspect it.", objectName="muted")
        self.selection_detail.setWordWrap(True)
        self.selection_preview = QLabel()
        self.selection_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selection_preview.setFixedHeight(150)
        self.selection_preview.hide()
        self.remove_button = QPushButton("Remove marker")
        self.remove_button.setObjectName("secondary")
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self.remove_cut)
        layout.addWidget(self.selection_title)
        layout.addWidget(self.selection_detail)
        layout.addWidget(self.selection_preview)
        layout.addWidget(self.remove_button)
        layout.addStretch()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress_label = QLabel("Ready", objectName="muted")
        self.export_button = QPushButton("Export reviewed result", objectName="primary")
        self.export_button.clicked.connect(self.export_result)
        layout.addWidget(self.progress)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.export_button)
        root.addWidget(right)
        root.setSizes([225, 950, 275])
        self.setStatusBar(QStatusBar())

    def _analysis_fields(self):
        self.mode = QComboBox()
        self.mode.addItem("Scene detection", "scene")
        self.mode.addItem("Every seconds", "interval")
        self.mode.addItem("Detect faces", "faces")
        self.sensitivity = QComboBox()
        self.sensitivity.addItems(PROFILES.keys())
        self.sensitivity.setCurrentText("Normal")
        self.interval = QSpinBox()
        self.interval.setRange(1, 18000)
        self.interval.setValue(10)
        self.interval.setSuffix(" seconds")
        self.accel = QComboBox()
        for value in ACCEL_OPTIONS:
            if value in self.available_accel: self.accel.addItem(value.upper(), value)
        return (("Mode", self.mode), ("Sensitivity", self.sensitivity),
                ("Interval", self.interval), ("Acceleration", self.accel))

    def _build_actions(self):
        self.import_action = QAction("Import Video…", self)
        self.import_action.setShortcut(QKeySequence.StandardKey.Open)
        self.import_action.triggered.connect(self.import_videos)
        save_action = QAction("Save Project…", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_project)
        self.load_action = QAction("Open Project…", self)
        self.load_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.load_action.triggered.connect(self.open_project)
        self.add_cut_action = QAction("Add Cut at Playhead", self)
        self.add_cut_action.setShortcut("Ctrl+K")
        self.add_cut_action.triggered.connect(self.add_cut)
        playback_action = QAction("Play/Pause", self)
        playback_action.setShortcut(QKeySequence(Qt.Key.Key_Space))
        playback_action.triggered.connect(self.toggle_playback)
        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        reset_layout_action.triggered.connect(self.reset_layout)
        self.menuBar().addMenu("File").addActions([self.import_action, self.load_action, save_action])
        self.menuBar().addMenu("Edit").addAction(self.add_cut_action)
        self.menuBar().addMenu("View").addAction(reset_layout_action)
        self.addAction(playback_action)

    def reset_layout(self):
        self.showNormal()
        self.resize(1440, 900)
        self.main_splitter.setSizes([225, 950, 275])
        self.workspace_splitter.setSizes([650, 210])
        self.timeline.pixels_per_second = 70.0
        self.timeline.rebuild()
        self.statusBar().showMessage("Default layout restored", 3000)

    def _apply_theme(self):
        self.setStyleSheet(f"""
        QMainWindow, QWidget {{ background:{BG_MAIN}; color:{TEXT_MAIN}; font-size:10px; }}
        QFrame#panel {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #323438, stop:1 #292b2f); border:1px solid #45484e; border-radius:10px; }}
        QFrame#viewer {{ background:#242629; border:1px solid #45484e; border-radius:10px; }}
        QLabel {{ background:transparent; border:0; }}
        QLabel#sectionTitle {{ color:#f3f4f6; font-weight:700; font-size:11px; padding:6px 2px; }}
        QLabel#muted {{ color:{TEXT_MUTED}; }}
        QListWidget {{ background:#1d1f22; border:1px solid #464a50; border-radius:8px; padding:4px; outline:0; selection-background-color:{ACCENT}; }}
        QListWidget::item {{ color:#d1d5db; padding:9px 7px; border-radius:6px; margin:1px; }}
        QListWidget::item:hover {{ background:#353941; color:white; }}
        QListWidget::item:selected {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #214f7b, stop:1 #286da9); color:white; }}
        QComboBox, QSpinBox {{ min-height:20px; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #292c31, stop:1 #202226); color:#e5e7eb; border:1px solid #4a4e55; border-radius:7px; padding:6px 9px; selection-background-color:{ACCENT}; }}
        QComboBox:hover, QSpinBox:hover {{ border-color:#69717c; background:#2d3136; }}
        QComboBox:focus, QSpinBox:focus {{ border-color:#2d8cff; }}
        QComboBox::drop-down {{ width:25px; border:0; border-left:1px solid #444950; }}
        QComboBox QAbstractItemView {{ background:#25282c; color:#e5e7eb; border:1px solid #555a62; selection-background-color:#286da9; padding:3px; outline:0; }}
        QPushButton {{ min-height:20px; color:#dbe1e8; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #464a50, stop:1 #303339); border:1px solid #555a62; border-radius:8px; padding:7px 12px; font-weight:700; }}
        QPushButton:hover {{ color:white; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #59606a, stop:1 #3c4148); border-color:#737b87; }}
        QPushButton:pressed {{ background:#25282c; border-color:#2d8cff; padding-top:9px; padding-bottom:5px; }}
        QPushButton#primary {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #327fc4, stop:1 #1f538d); border-color:#4b91cf; color:white; }}
        QPushButton#primary:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4295dd, stop:1 #2869ad); border-color:#70b3ed; }}
        QPushButton#danger {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #b84b55, stop:1 #7f2932); border-color:#d26a73; color:white; }}
        QPushButton#danger:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #d25a65, stop:1 #97343e); border-color:#ef8991; }}
        QPushButton#secondary {{ color:#cbd5e1; }}
        QPushButton:disabled {{ color:#686e78; background:#27292d; border-color:#3a3d42; }}
        QFrame#transportGroup {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #35383d, stop:1 #25272b); border:1px solid #484c53; border-radius:10px; }}
        QFrame#transportGroup QToolButton {{ color:#d1d5db; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #464a50, stop:1 #303339); border:1px solid #555a62; border-radius:7px; padding:0; font-weight:700; }}
        QFrame#transportGroup QToolButton:hover {{ color:white; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #59606a, stop:1 #3c4148); border-color:#737b87; }}
        QFrame#transportGroup QToolButton:pressed {{ background:#24272b; border-color:#2d8cff; padding-top:2px; }}
        QFrame#transportGroup QToolButton#transportSkip {{ color:#9ca3af; }}
        QFrame#transportGroup QToolButton#transportStep {{ color:#cbd5e1; font-size:10px; }}
        QFrame#transportGroup QToolButton#transportPlay {{ color:white; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #3b96ff, stop:1 #1766b5); border:1px solid #75b8ff; border-radius:9px; font-size:15px; }}
        QFrame#transportGroup QToolButton#transportPlay:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #58a7ff, stop:1 #2378c9); }}
        QSlider#mediaScrubber {{ background:transparent; border:0; padding-left:7px; padding-right:7px; }}
        QSlider#mediaScrubber::groove:horizontal {{ height:3px; background:#565b63; border:0; border-radius:1px; }}
        QSlider#mediaScrubber::sub-page:horizontal {{ background:#2d8cff; border:0; border-radius:1px; }}
        QSlider#mediaScrubber::add-page:horizontal {{ background:#565b63; border:0; border-radius:1px; }}
        QSlider#mediaScrubber::handle:horizontal {{ background:#e5edf6; border:1px solid #2d8cff; width:14px; height:14px; margin:-6px 0; border-radius:7px; }}
        QSlider#mediaScrubber::handle:horizontal:hover {{ background:white; border-color:#60a5fa; }}
        QFrame#transportGroup QLineEdit#timecode {{ background:transparent; color:#60a5fa; border:0; border-radius:0; padding:5px 4px; font-weight:700; }}
        QFrame#transportGroup QLineEdit#timecode:focus {{ background:#202b3d; border:1px solid #3b82c4; border-radius:5px; }}
        QFrame#transportGroup QLineEdit#timecode:read-only {{ background:transparent; color:#60a5fa; border:0; }}
        QFrame#transportGroup QLabel#timeSeparator {{ background:transparent; color:#737b87; border:0; min-width:16px; max-width:16px; padding:5px 0; font-weight:700; }}
        QProgressBar {{ background:#202226; border:1px solid #41454b; border-radius:4px; height:7px; text-align:center; }}
        QProgressBar::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #188a51, stop:1 {SUCCESS}); border-radius:3px; }}
        QMenuBar {{ background:#2b2d31; color:#d1d5db; border-bottom:1px solid #45484e; padding:2px; }}
        QMenuBar::item {{ padding:5px 9px; border-radius:5px; }}
        QMenuBar::item:selected {{ background:#41454b; color:white; }}
        QMenu {{ background:#292c30; color:#e5e7eb; border:1px solid #50545b; padding:5px; }}
        QMenu::item {{ padding:7px 28px 7px 12px; border-radius:5px; }}
        QMenu::item:selected {{ background:#286da9; color:white; }}
        QStatusBar {{ background:#2b2d31; color:{TEXT_MUTED}; border-top:1px solid #45484e; }}
        QAbstractScrollArea::corner {{ background:transparent; border:0; }}
        QScrollBar:horizontal {{ background:transparent; border:0; height:16px; margin:2px 4px; }}
        QScrollBar::handle:horizontal {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #464a50, stop:1 #303339); border:1px solid #555a62; min-width:52px; border-radius:6px; margin:2px 0; }}
        QScrollBar::handle:horizontal:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #59606a, stop:1 #3c4148); border-color:#737b87; }}
        QScrollBar::handle:horizontal:pressed {{ background:#25282c; border-color:#2d8cff; }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background:transparent; border:0; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ background:transparent; border:0; width:0; }}
        QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {{ width:0; height:0; }}
        QScrollBar:vertical {{ background:transparent; border:0; width:16px; margin:4px 2px; }}
        QScrollBar::handle:vertical {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #464a50, stop:1 #303339); border:1px solid #555a62; min-height:52px; border-radius:6px; margin:0 2px; }}
        QScrollBar::handle:vertical:hover {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #59606a, stop:1 #3c4148); border-color:#737b87; }}
        QScrollBar::handle:vertical:pressed {{ background:#25282c; border-color:#2d8cff; }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background:transparent; border:0; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ background:transparent; border:0; height:0; }}
        QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{ width:0; height:0; }}
        QToolTip {{ background:#202226; color:#f3f4f6; border:1px solid #626872; padding:5px; }}
        QSplitter::handle {{ background:{BG_MAIN}; width:5px; height:5px; }}
        """)

    def import_videos(self):
        if self.task:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Import videos", "",
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v)")
        for path in paths:
            if Path(path).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS: continue
            try: media = probe_media(path)
            except Exception as error:
                QMessageBox.warning(self, "Could not import video", f"{path}\n\n{error}")
                continue
            self.project.media.append(media)
            item = QListWidgetItem(Path(path).name)
            item.setToolTip(path)
            self.media_list.addItem(item)
        if self.media_list.count() and self.media_list.currentRow() < 0:
            self.media_list.setCurrentRow(0)

    def _mode_changed(self):
        mode = self.mode.currentData()
        self.interval.setEnabled(mode == "interval")
        self.sensitivity.setEnabled(mode != "interval")
        if mode == "faces" and self.sensitivity.currentText() == "Auto":
            self.sensitivity.setCurrentText("Normal")
        if self.current_media and self.current_media.mode != mode:
            self.current_media.mode = mode
            self.current_media.cuts.clear()
            self.current_media.faces.clear()
            self.selected_cut = None
            self.timeline.set_media(self.current_media)
            self.show_cut(None)

    def _select_media(self, row):
        if self.task:
            return
        if not 0 <= row < len(self.project.media): return
        self.current_media = self.project.media[row]
        self.preview_end_ms = None
        self.preview_start_ms = None
        self.preview_armed = False
        self.timeline.set_media(self.current_media)
        self.player.setSource(QUrl.fromLocalFile(self.current_media.path))
        self.statusBar().showMessage(f"{self.current_media.width}×{self.current_media.height} · {self.current_media.fps:.3f} fps · {timecode(self.current_media.duration)}")

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState: self.player.pause()
        else: self.player.play()

    def relative_seek(self, amount):
        self.seek_player_ms(self.player.position() + amount)

    def seek_player_ms(self, value):
        self.preview_end_ms = None
        self.preview_start_ms = None
        self.preview_armed = False
        self.player.setPosition(max(0, min(self.player.duration(), int(value))))

    def play_segment(self, start_seconds, end_seconds):
        start_ms, end_ms = round(start_seconds * 1000), round(end_seconds * 1000)
        if end_ms <= start_ms:
            return
        self.player.pause()
        self.preview_start_ms = start_ms
        self.preview_end_ms = end_ms
        self.preview_armed = False
        self.player.setPosition(start_ms)
        QTimer.singleShot(80, self._ensure_segment_preview_started)
        self.statusBar().showMessage(
            f"Previewing clip · {timecode(start_seconds)} – {timecode(end_seconds)}")

    def _ensure_segment_preview_started(self):
        if self.preview_end_ms is None or self.preview_start_ms is None or self.preview_armed:
            return
        if abs(self.player.position() - self.preview_start_ms) <= 150:
            self.preview_armed = True
            self.player.play()
            return
        self.player.setPosition(self.preview_start_ms)
        QTimer.singleShot(80, self._ensure_segment_preview_started)

    def jump_event(self, direction):
        if not self.current_media:
            return
        current = round(self.player.position() / 1000 * self.current_media.fps)
        frames = [cut.frame for cut in self.current_media.cuts]
        frames.extend(face.best_frame for face in self.current_media.faces if face.enabled)
        frames = sorted(set(frames))
        target = (next((frame for frame in frames if frame > current + 1), None) if direction > 0
                  else next((frame for frame in reversed(frames) if frame < current - 1), None))
        if target is not None:
            self.seek_player_ms(round(target / self.current_media.fps * 1000))

    def _position_changed(self, value):
        if not self.seek.isSliderDown(): self.seek.setValue(value)
        self.timeline.set_position(value / 1000)
        if not self.clock.hasFocus():
            self.clock.setText(transport_timecode(value / 1000))
        self.duration_clock.setText(transport_timecode(self.player.duration() / 1000))
        if (self.preview_armed and self.preview_end_ms is not None and
                value >= self.preview_end_ms - 10):
            end = self.preview_end_ms
            self.preview_end_ms = None
            self.preview_start_ms = None
            self.preview_armed = False
            self.player.pause()
            self.player.setPosition(end)
            self.statusBar().showMessage("Clip preview complete", 3000)

    def _duration_changed(self, value):
        self.seek.setRange(0, max(0, value))
        self.duration_clock.setText(transport_timecode(value / 1000))

    def _media_status_changed(self, status):
        visible_statuses = {
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.BufferingMedia,
            QMediaPlayer.MediaStatus.StalledMedia,
            QMediaPlayer.MediaStatus.EndOfMedia,
        }
        self.video.setVisible(status in visible_statuses)

    def seek_from_timecode(self):
        seconds = parse_timecode(self.clock.text())
        if seconds is None:
            self.clock.setText(transport_timecode(self.player.position() / 1000))
            return
        self.seek_player_ms(round(seconds * 1000))

    def add_cut(self):
        if self.task:
            return
        if not self.current_media: return
        frame = round(self.player.position() / 1000 * self.current_media.fps)
        if 0 < frame < self.current_media.total_frames:
            self.current_media.cuts.append(CutEvent(frame=frame, source="manual"))
            self.current_media.normalize_cuts()
            self.timeline.rebuild()

    def show_cut(self, event):
        self.selected_cut = event
        self.remove_button.setEnabled(event is not None)
        if not event or not self.current_media:
            self.selection_title.setText("No event selected")
            self.selection_detail.setText("Select a marker to inspect it.")
            self.selection_preview.clear()
            self.selection_preview.hide()
            return
        if isinstance(event, FaceEvent):
            seconds = event.best_frame / self.current_media.fps
            self.selection_title.setText("Face event")
            self.selection_detail.setText(
                f"Best frame: {event.best_frame}\n"
                f"Visible: {timecode(event.start_frame / self.current_media.fps)} – "
                f"{timecode(event.end_frame / self.current_media.fps)}\n"
                f"Confidence: {event.confidence:.2f} · Quality: {event.quality:.2f}")
            self.remove_button.setText("Exclude face" if event.enabled else "Include face")
            pixmap = QPixmap(event.crop_file)
            if not pixmap.isNull():
                self.selection_preview.setPixmap(pixmap.scaled(
                    220, 140, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
                self.selection_preview.show()
        else:
            seconds = event.frame / self.current_media.fps
            self.selection_title.setText("Cut marker")
            self.selection_detail.setText(f"{timecode(seconds)}\nFrame {event.frame}\nSource: {event.source}")
            self.remove_button.setText("Remove marker")
            self.selection_preview.clear()
            self.selection_preview.hide()
        self.player.setPosition(round(seconds * 1000))

    def remove_cut(self):
        if self.task:
            return
        if not self.current_media or not self.selected_cut: return
        if isinstance(self.selected_cut, FaceEvent):
            self.selected_cut.enabled = not self.selected_cut.enabled
            self.timeline.rebuild()
            self.show_cut(self.selected_cut)
            return
        self.current_media.cuts = [cut for cut in self.current_media.cuts if cut.event_id != self.selected_cut.event_id]
        self.selected_cut = None
        self.timeline.rebuild()
        self.show_cut(None)

    def analyze(self):
        if not self.current_media or self.task: return
        mode = self.mode.currentData()
        self.player.pause()
        media, interval = self.current_media, self.interval.value()
        media.mode, media.sensitivity = mode, self.sensitivity.currentText()
        # A new pass replaces the previous mode's events for this media.
        media.cuts.clear()
        media.faces.clear()
        self.selected_cut = None
        self.timeline.set_media(media)
        self.show_cut(None)
        accel = self.accel.currentData() or "cpu"
        def work(task):
            if mode == "faces":
                cache = os.path.join(USER_DATA_DIR, "cache", media.media_id)
                os.makedirs(cache, exist_ok=True)
                engine = FaceDetectionEngine(
                    media.path, cache, LogAdapter(task), ProgressAdapter(task),
                    profile=media.sensitivity, accel=accel, preview_enabled=False)
                task.register_cancel(engine.stop)
                if not engine.run():
                    raise RuntimeError("Face analysis stopped before producing results")
                events = []
                for face in engine._saved_faces:
                    events.append(FaceEvent(
                        start_frame=int(face.get("first_frame", 0)),
                        end_frame=int(face.get("last_frame", 0)),
                        best_frame=int(face.get("best_frame", 0)),
                        crop_file=os.path.join(engine._output_dir or cache, face.get("file", "")),
                        confidence=float(face.get("confidence", 0)),
                        quality=float(face.get("quality", 0)),
                        box=list(face.get("box") or []),
                    ))
                return ("faces", events)
            cfg = PROFILES[media.sensitivity].copy()
            cfg.update(ENCODER=accel, INFERENCE="cpu", DEBUG=False)
            if mode == "interval": cfg["FIXED_INTERVAL"] = interval
            engine = SceneEngine(media.path, self.project.output_folder or os.path.dirname(media.path), cfg,
                LogAdapter(task), ProgressAdapter(task), preview_enabled=False, export_results=False)
            task.register_cancel(engine.stop)
            if not engine.run(scene_mode=mode == "scene"):
                raise RuntimeError("Analysis stopped before producing results")
            return ("cuts", [CutEvent(start, mode) for start, _end in engine.analysis_scenes if start > 0])
        self.start_task(work, lambda result, target=media: self.analysis_finished(target, result), "Analyzing video…")

    def analysis_finished(self, media, result):
        kind, events = result
        if kind == "faces":
            media.faces = events
        else:
            media.cuts = events
            media.normalize_cuts()
        if self.current_media is media:
            self.timeline.rebuild()
        label = f"{len(events)} faces" if kind == "faces" else f"{len(events) + 1} segments"
        self.statusBar().showMessage(f"Analysis complete · {label}", 8000)

    def export_result(self):
        if not self.current_media or self.task: return
        folder = QFileDialog.getExistingDirectory(self, "Export reviewed result", self.project.output_folder)
        if not folder: return
        self.project.output_folder = folder
        media, segments = self.current_media, self.current_media.segments()
        accel = self.accel.currentData() or "cpu"
        def work(task):
            if media.mode == "faces":
                destination = os.path.join(folder, f"{Path(media.path).stem}_faces_reviewed")
                os.makedirs(destination, exist_ok=True)
                saved = 0
                for event in media.faces:
                    if task.isInterruptionRequested():
                        return None
                    if not event.enabled or not os.path.isfile(event.crop_file):
                        continue
                    target = os.path.join(destination, f"face_{saved + 1:04d}.png")
                    __import__("shutil").copy2(event.crop_file, target)
                    saved += 1
                with open(os.path.join(destination, "faces.json"), "w", encoding="utf-8") as handle:
                    json.dump({"source": media.path, "faces": [vars(event) for event in media.faces],
                               "faces_exported": saved}, handle, indent=2)
                return destination
            cfg = PROFILES[media.sensitivity].copy()
            cfg.update(ENCODER=accel, INFERENCE="cpu", DEBUG=False)
            engine = SceneEngine(media.path, folder, cfg, LogAdapter(task), ProgressAdapter(task), preview_enabled=False)
            task.register_cancel(engine.stop)
            engine.scene_mode = media.mode == "scene"
            engine.detected, engine._start_time = len(segments), time.time()
            engine._cut_scenes(segments)
            return engine._cut_output_dir
        self.start_task(work, lambda output: QMessageBox.information(self, "Export complete", f"Files were exported to:\n{output}"), "Exporting…")

    def start_task(self, work, callback, label):
        if self.task:
            return
        self._set_working_ui(True)
        self.export_button.setEnabled(False)
        self.progress_label.setText(label)
        task = Task(work, self)
        self.task = task
        task.progress.connect(self.task_progress)
        task.result.connect(lambda result: self.task_done(task, callback, result))
        task.failed.connect(lambda message: self.task_failed(task, message))
        task.cancelled.connect(lambda: self.task_cancelled(task))
        task.start()

    def toggle_analysis(self):
        if self.task:
            self.stop_current_task()
        else:
            self.analyze()

    def stop_current_task(self, confirm=True):
        task = self.task
        if not task or not task.isRunning():
            return True
        if confirm:
            answer = QMessageBox.question(
                self, "Stop current process",
                "Stop the current analysis/export? Partial results will be discarded.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return False
        self.progress_label.setText("Stopping…")
        self.analyze_button.setEnabled(False)
        task.cancel()
        return True

    def _set_working_ui(self, working):
        self.media_list.setEnabled(not working)
        self.import_button.setEnabled(not working)
        self.import_action.setEnabled(not working)
        self.load_action.setEnabled(not working)
        self.add_cut_action.setEnabled(not working)
        for control in (self.mode, self.sensitivity, self.interval, self.accel):
            control.setEnabled(not working)
        if not working:
            self._mode_changed()
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("Stop" if working else "Analyze video")
        self.analyze_button.setObjectName("danger" if working else "primary")
        self.analyze_button.style().unpolish(self.analyze_button)
        self.analyze_button.style().polish(self.analyze_button)

    def _finish_task_ui(self, task, label):
        if self.task is task:
            self.task = None
        task.cancel_callback = None
        self._set_working_ui(False)
        self.export_button.setEnabled(True)
        self.progress_label.setText(label)

    def task_progress(self, value, label):
        if value >= 0: self.progress.setValue(round(value * 1000))
        if label: self.progress_label.setText(label)

    def task_done(self, task, callback, result):
        self._finish_task_ui(task, "Complete")
        self.progress.setValue(1000)
        callback(result)
        task.deleteLater()

    def task_failed(self, task, message):
        self._finish_task_ui(task, "Failed")
        QMessageBox.critical(self, "Scenespy error", message)
        task.deleteLater()

    def task_cancelled(self, task):
        self._finish_task_ui(task, "Stopped")
        self.progress.setValue(0)
        self.statusBar().showMessage("Process stopped. Partial results were discarded.", 6000)
        task.deleteLater()

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Scenespy project", self.project.project_path, "Scenespy Project (*.scenespy)")
        if path: self.project.save(path)

    def open_project(self):
        if self.task:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Scenespy project", "", "Scenespy Project (*.scenespy)")
        if not path:
            return
        try:
            project = ScenespyProject.load(path)
        except Exception as error:
            QMessageBox.critical(self, "Could not open project", str(error))
            return
        self.player.stop()
        self.project = project
        self.media_list.clear()
        for media in project.media:
            item = QListWidgetItem(Path(media.path).name)
            item.setToolTip(media.path)
            self.media_list.addItem(item)
        if self.media_list.count():
            self.media_list.setCurrentRow(0)

    def closeEvent(self, event):
        if self.task and self.task.isRunning():
            answer = QMessageBox.question(
                self, "Close Scenespy",
                "A process is still running. Stop it and close Scenespy?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            task = self.task
            task.cancel()
            terminate_all_child_processes(grace_seconds=.5)
            if not task.wait(7000):
                terminate_all_child_processes(grace_seconds=.1)
                if not task.wait(2000):
                    # Final exit-only safeguard for a stuck third-party decoder.
                    task.terminate()
                    task.wait(1000)
        self.player.stop()
        terminate_all_child_processes(grace_seconds=.2)
        event.accept()


def create_application(argv=None):
    app = QApplication.instance() or QApplication(argv or [])
    app.setApplicationName("Scenespy")
    assets = Path(__file__).parent / "assets" / "fonts"
    regular_id = QFontDatabase.addApplicationFont(str(assets / "JetBrainsMono-Regular.ttf"))
    QFontDatabase.addApplicationFont(str(assets / "JetBrainsMono-Bold.ttf"))
    families = QFontDatabase.applicationFontFamilies(regular_id) if regular_id >= 0 else []
    app.setFont(QFont(families[0] if families else "JetBrains Mono", 10))
    return app
