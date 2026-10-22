"""Production-oriented Qt workspace for non-destructive Scenespy projects."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import cv2
import qtawesome as qta

from PySide6.QtCore import (QEasingCurve, QPoint, QPointF, QProcess, QPropertyAnimation,
                            QRectF, QSize, Qt, QThread, QTimer, QUrl, Signal)
from PySide6.QtGui import (QAction, QBrush, QColor, QDesktopServices, QFont, QFontDatabase,
                           QIcon, QImage, QKeySequence, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGraphicsItem, QGraphicsLineItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
    QSizePolicy, QSpinBox, QSplitter, QStackedLayout, QStatusBar, QToolButton, QVBoxLayout,
    QWidget, QWidgetAction,
)

from .media import probe_media
from .project import CutEvent, FaceEvent, MediaItem, ScenespyProject
from .face_engine import FaceDetectionEngine
from .frame_map import FrameMap, create_analysis_proxy
from .scene_engine import SceneEngine
from .shared import (ACCEL_OPTIONS, ACCENT, ALLOWED_VIDEO_EXTENSIONS, BG_CARD, BG_MAIN,
                     BG_PANEL, BORDER_SOFT2, PROFILES, SUCCESS, TEXT_MAIN, TEXT_MUTED,
                     USER_DATA_DIR, detect_available_encoder_accel, detect_available_inference_accel,
                     terminate_all_child_processes)


# Muted pastel colors keep the patterned scene texture readable without making
# the timeline compete visually with the video preview.
SCENE_PALETTE = (
    "#48765a",  # green
    "#684a3a",  # brown
    "#626873",  # gray
    "#426f96",  # blue
    "#8a4e54",  # red
    "#81783c",  # yellow
    "#93603a",  # orange
    "#6c508c",  # purple
    "#925675",  # pink
)


def timecode(seconds, duration=None):
    total = max(0, int(round(float(seconds or 0) * 1000)))
    reference_total = max(0, int(round(float(duration or 0) * 1000)))
    hours, total = divmod(total, 3_600_000)
    minutes, total = divmod(total, 60_000)
    secs, millis = divmod(total, 1000)
    show_hours = max(hours * 3_600_000 + minutes * 60_000 + secs * 1000 + millis,
                     reference_total) >= 3_600_000
    if show_hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def transport_timecode(seconds, duration=None):
    return timecode(seconds, duration).split(".", 1)[0]


def timecode_has_hours(duration):
    return int(round(max(0.0, float(duration or 0)) * 1000)) >= 3_600_000


def scrub_timecode(seconds, duration):
    """Useful hover precision for short clips without cluttering longer media."""
    if float(duration or 0) < 60:
        return timecode(seconds, duration)
    return transport_timecode(seconds, duration)


def final_face_crop_box(box, frame_width, frame_height):
    """Return the exact expanded box used by the saved face crop."""
    if len(box or ()) != 4 or frame_width <= 0 or frame_height <= 0:
        return []
    x1, y1, x2, y2 = (float(value) for value in box)
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    crop_height = min(frame_height, max(face_height * 1.9, face_width * 1.75))
    crop_width = min(frame_width, max(face_width * 1.7, crop_height * .88))
    crop_height = min(frame_height, max(crop_height, crop_width / .92))
    eye_x = (x1 + x2) / 2
    eye_y = y1 + face_height * .38
    crop_x1 = min(max(0.0, eye_x - crop_width / 2), frame_width - crop_width)
    crop_y1 = min(max(0.0, eye_y - crop_height * .42), frame_height - crop_height)
    return [
        round(crop_x1), round(crop_y1),
        round(crop_x1 + crop_width), round(crop_y1 + crop_height),
    ]


def load_fresh_pixmap(path):
    image = QImage(path) if path else QImage()
    return QPixmap.fromImage(image) if not image.isNull() else QPixmap()


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


def themed_icon(name, color="#cbd5e1", scale_factor=1.0):
    return qta.icon(
        name,
        color=color,
        color_active="#ffffff",
        color_disabled="#565b63",
        scale_factor=scale_factor,
    )


class ScrubSlider(QSlider):
    """Seek slider whose entire groove is directly draggable."""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self._time_bubble = QLabel(None, Qt.WindowType.ToolTip)
        self._time_bubble.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._time_bubble.setStyleSheet(
            "QLabel { background:#35383d; color:#f3f4f6; border:1px solid #69717c; "
            "border-radius:0; padding:4px 7px; font-weight:700; }")
        self._time_bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._time_bubble.setFixedHeight(28)

    def _value_from_mouse(self, event):
        width = max(1, self.width() - 14)
        ratio = max(0.0, min(1.0, (event.position().x() - 7) / width))
        return round(self.minimum() + ratio * (self.maximum() - self.minimum()))

    def _show_time_tooltip(self, event, value=None):
        value = self._value_from_mouse(event) if value is None else value
        duration = self.maximum() / 1000
        text = scrub_timecode(value / 1000, duration)
        self._time_bubble.setText(text)
        sample = scrub_timecode(duration, duration)
        self._time_bubble.setFixedWidth(
            max(self._time_bubble.fontMetrics().horizontalAdvance(sample),
                self._time_bubble.fontMetrics().horizontalAdvance(text)) + 30)
        anchor = self.mapToGlobal(QPoint(round(event.position().x()), 0))
        self._time_bubble.move(
            anchor.x() - self._time_bubble.width() // 2,
            anchor.y() - self._time_bubble.height() - 8)
        self._time_bubble.show()

    def _set_from_mouse(self, event):
        value = self._value_from_mouse(event)
        self.setValue(value)
        self.sliderMoved.emit(value)
        self._show_time_tooltip(event, value)

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
        self._show_time_tooltip(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isSliderDown():
            self._set_from_mouse(event)
            self.setSliderDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._time_bubble.hide()
        super().leaveEvent(event)

    def hideEvent(self, event):
        self._time_bubble.hide()
        super().hideEvent(event)


class VideoBackdrop(QWidget):
    """Absolute-black canvas shown behind the native video surface."""
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        painter.end()
        super().paintEvent(event)


class FaceBoxOverlay(QWidget):
    """Four thin marker edges which never cover the native video surface."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._host = parent
        self._box = None
        self._source_size = QSize()
        self.hide()
        self._edges = [QWidget(parent) for _ in range(4)]
        for edge in self._edges:
            edge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            edge.setStyleSheet("background:#00ff00; border:0;")
            edge.hide()

    def set_face(self, box, source_width, source_height):
        if len(box or ()) != 4 or source_width <= 0 or source_height <= 0:
            self.clear_face()
            return
        self._box = tuple(float(value) for value in box)
        self._source_size = QSize(int(source_width), int(source_height))
        self.refresh_geometry()

    def clear_face(self):
        self._box = None
        for edge in self._edges:
            edge.hide()

    def refresh_geometry(self):
        if not self._box or self._source_size.isEmpty() or not self._host:
            return
        source_width = self._source_size.width()
        source_height = self._source_size.height()
        scale = min(self._host.width() / source_width, self._host.height() / source_height)
        draw_width = source_width * scale
        draw_height = source_height * scale
        offset_x = (self._host.width() - draw_width) / 2
        offset_y = (self._host.height() - draw_height) / 2
        x1, y1, x2, y2 = self._box
        marker = QRectF(
            offset_x + x1 * scale,
            offset_y + y1 * scale,
            max(1.0, (x2 - x1) * scale),
            max(1.0, (y2 - y1) * scale),
        ).intersected(QRectF(offset_x, offset_y, draw_width, draw_height))
        if marker.isEmpty():
            self.clear_face()
            return
        left = round(marker.left())
        top = round(marker.top())
        right = round(marker.right())
        bottom = round(marker.bottom())
        thickness = 2
        self._edges[0].setGeometry(left, top, max(thickness, right - left), thickness)
        self._edges[1].setGeometry(left, max(top, bottom - thickness),
                                   max(thickness, right - left), thickness)
        self._edges[2].setGeometry(left, top, thickness, max(thickness, bottom - top))
        self._edges[3].setGeometry(max(left, right - thickness), top,
                                   thickness, max(thickness, bottom - top))
        for edge in self._edges:
            edge.show()
            edge.raise_()


class FaceMarkerVideoWidget(QVideoWidget):
    rightClicked = Signal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.face_overlay = FaceBoxOverlay(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.face_overlay.refresh_geometry()

    def contextMenuEvent(self, event):
        self.rightClicked.emit(QPointF(event.pos()))
        event.accept()


class ProcessingPreviewLabel(QLabel):
    """Letterboxed preview which always keeps the complete source frame visible."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_source_pixmap(self, pixmap):
        self.source_pixmap = pixmap
        self._fit_pixmap()

    def clear_source(self):
        self.source_pixmap = None
        self.clear()

    def _fit_pixmap(self):
        if self.source_pixmap and not self.source_pixmap.isNull():
            self.setPixmap(self.source_pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_pixmap()


class FaceCropEditor(ProcessingPreviewLabel):
    cropChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.crop_box = []
        self.source_size = QSize()
        self._drag_mode = None
        self._drag_start = QPointF()
        self._start_rect = QRectF()
        self.setMouseTracking(True)

    def set_face_frame(self, pixmap, crop_box, source_width, source_height):
        self.crop_box = [float(value) for value in crop_box]
        self.source_size = QSize(int(source_width), int(source_height))
        self.set_source_pixmap(pixmap)
        self.update()

    def clear_source(self):
        super().clear_source()
        self.crop_box = []
        self.source_size = QSize()
        self._drag_mode = None

    def _image_rect(self):
        pixmap = self.pixmap()
        if not pixmap or pixmap.isNull():
            return QRectF()
        return QRectF(
            (self.width() - pixmap.width()) / 2,
            (self.height() - pixmap.height()) / 2,
            pixmap.width(), pixmap.height())

    def _widget_crop_rect(self):
        image_rect = self._image_rect()
        if image_rect.isEmpty() or self.source_size.isEmpty() or len(self.crop_box) != 4:
            return QRectF()
        sx = image_rect.width() / self.source_size.width()
        sy = image_rect.height() / self.source_size.height()
        x1, y1, x2, y2 = self.crop_box
        return QRectF(image_rect.left() + x1 * sx, image_rect.top() + y1 * sy,
                      (x2 - x1) * sx, (y2 - y1) * sy)

    def _set_from_widget_rect(self, rect):
        image_rect = self._image_rect()
        if image_rect.isEmpty() or self.source_size.isEmpty():
            return
        rect = rect.normalized().intersected(image_rect)
        sx = self.source_size.width() / image_rect.width()
        sy = self.source_size.height() / image_rect.height()
        self.crop_box = [
            round((rect.left() - image_rect.left()) * sx),
            round((rect.top() - image_rect.top()) * sy),
            round((rect.right() - image_rect.left()) * sx),
            round((rect.bottom() - image_rect.top()) * sy),
        ]
        self.update()

    def _hit_mode(self, point):
        rect = self._widget_crop_rect()
        if rect.isEmpty():
            return None
        handles = {
            "top_left": rect.topLeft(), "top_right": rect.topRight(),
            "bottom_left": rect.bottomLeft(), "bottom_right": rect.bottomRight(),
        }
        for mode, handle in handles.items():
            if QRectF(handle.x() - 7, handle.y() - 7, 14, 14).contains(point):
                return mode
        return "move" if rect.contains(point) else None

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = self._widget_crop_rect()
        if rect.isEmpty():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#00ff00"), 2))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor("#102010"), 1))
        painter.setBrush(QColor("#00ff00"))
        for point in (rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()):
            painter.drawRect(QRectF(point.x() - 3, point.y() - 3, 6, 6))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            mode = self._hit_mode(event.position())
            if mode:
                self._drag_mode = mode
                self._drag_start = event.position()
                self._start_rect = self._widget_crop_rect()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._drag_mode:
            mode = self._hit_mode(event.position())
            self.setCursor(Qt.CursorShape.SizeAllCursor if mode == "move"
                           else Qt.CursorShape.SizeFDiagCursor if mode
                           else Qt.CursorShape.ArrowCursor)
            return
        delta = event.position() - self._drag_start
        rect = QRectF(self._start_rect)
        if self._drag_mode == "move":
            rect.translate(delta)
            image_rect = self._image_rect()
            if rect.left() < image_rect.left(): rect.moveLeft(image_rect.left())
            if rect.right() > image_rect.right(): rect.moveRight(image_rect.right())
            if rect.top() < image_rect.top(): rect.moveTop(image_rect.top())
            if rect.bottom() > image_rect.bottom(): rect.moveBottom(image_rect.bottom())
        else:
            if "left" in self._drag_mode: rect.setLeft(event.position().x())
            if "right" in self._drag_mode: rect.setRight(event.position().x())
            if "top" in self._drag_mode: rect.setTop(event.position().y())
            if "bottom" in self._drag_mode: rect.setBottom(event.position().y())
            if rect.width() < 12 or rect.height() < 12:
                return
        self._set_from_widget_rect(rect)

    def mouseReleaseEvent(self, event):
        if self._drag_mode and event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = None
            self.cropChanged.emit([int(value) for value in self.crop_box])
            event.accept()
            return
        super().mouseReleaseEvent(event)


class FramedComboBox(QComboBox):
    """Combo box with a final overlay frame immune to Qt subcontrol clipping."""
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isEnabled():
            color = QColor("#34373c")
        elif self.hasFocus():
            color = QColor("#2d8cff")
        elif self.underMouse():
            color = QColor("#69717c")
        else:
            color = QColor("#4a4e55")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, 1.0))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(.5, .5, -.5, -.5), 7, 7)
        painter.end()


class CompactMenuItem(QWidget):
    activated = Signal()

    def __init__(self, icon, text, parent=None, hover_color="#a8323c"):
        super().__init__(parent)
        self.setObjectName("mediaRemoveItem")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._icon = icon
        self._text = text
        self._hover_color = hover_color
        self._icon_size = 16
        self._side_margin = 5
        self._vertical_margin = 7

    def sizeHint(self):
        metrics = self.fontMetrics()
        spacing = metrics.horizontalAdvance(" ")
        width = (self._side_margin * 2 + self._icon_size + spacing +
                 metrics.horizontalAdvance(self._text))
        height = self._vertical_margin * 2 + max(self._icon_size, metrics.height())
        return QSize(width, height)

    def paintEvent(self, event):
        painter = QPainter(self)
        hovered = self.underMouse()
        painter.fillRect(self.rect(), QColor(self._hover_color if hovered else "#292c30"))
        icon_rect = QRectF(
            self._side_margin,
            (self.height() - self._icon_size) / 2,
            self._icon_size,
            self._icon_size,
        ).toRect()
        mode = QIcon.Mode.Active if hovered else QIcon.Mode.Normal
        painter.drawPixmap(icon_rect, self._icon.pixmap(QSize(self._icon_size, self._icon_size), mode))
        text_x = icon_rect.right() + 1 + self.fontMetrics().horizontalAdvance(" ")
        painter.setPen(QColor("#ffffff" if hovered else "#e5e7eb"))
        painter.drawText(
            self.rect().adjusted(text_x, 0, -self._side_margin, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._text,
        )
        painter.end()

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.MouseButton.LeftButton and
                self.rect().contains(event.position().toPoint())):
            self.activated.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class Task(QThread):
    progress = Signal(float, str)
    preview = Signal(object, float)
    media_started = Signal(object)
    media_finished = Signal(object, bool)
    result = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, work, parent=None, done_label="Complete"):
        super().__init__(parent)
        self.work = work
        self.done_label = done_label
        self.cancel_callback = None
        self.preview_paused = False

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


class PreviewAdapter:
    """Thread-safe bridge between the processing engines and the Qt viewer."""
    def __init__(self, task, frame_map=None, fps=None):
        self.task = task
        self.frame_map = frame_map
        self.fps = fps

    def after(self, _delay, callback):
        if self.enabled:
            callback()

    @property
    def enabled(self):
        return not self.task.preview_paused and not self.task.isInterruptionRequested()

    def update_image(self, image):
        if image and self.enabled:
            self.task.preview.emit(image.copy(), -1.0)

    def update_image_at(self, image, seconds):
        if image and self.enabled:
            if self.frame_map is not None:
                seconds = self.frame_map.seconds_to_source(seconds, self.fps)
            self.task.preview.emit(image.copy(), float(seconds))

    def update_info(self, _text):
        pass


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
        # Keep scene boundaries above the playhead so their head never becomes
        # unreachable when both markers happen to occupy the same frame.
        self.setZValue(50)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

    def boundingRect(self):
        return QRectF(-7, -5, 14, 71)

    def shape(self):
        # A comfortable invisible target around the thin visible marker.  The
        # head is part of this item, so dragging it behaves exactly like the stem.
        path = QPainterPath()
        path.addRect(QRectF(-7, -5, 14, 18))
        path.addRect(QRectF(-4, 10, 8, 56))
        return path

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        active = self.isSelected()
        color = QColor("#cbd5e1" if active else "#94a3b8")
        painter.setPen(QPen(color, 2))
        painter.drawLine(QPointF(0, 7), QPointF(0, 64))
        painter.setPen(QPen(QColor("#f8fafc" if active else "#e2e8f0"), 1))
        painter.setBrush(color)
        painter.drawRoundedRect(QRectF(-5, -3, 10, 13), 2, 2)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            point = QPointF(value)
            frame = self.timeline.frame_for_x(point.x())
            lower, upper = self.timeline.bounds_for(self.event)
            return QPointF(
                self.timeline.x_for_frame(max(lower, min(frame, upper))), self.y())
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self.scene():
            self.timeline.handle_moved(self)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        QTimer.singleShot(0, self.timeline.finish_handle_move)


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
            self.timeline.seekRequested.emit(self.timeline.seconds_for_x(self.x()))

class FaceChip(QGraphicsRectItem):
    def __init__(self, event, start_x):
        super().__init__(start_x, 100, 44, 30)
        self.event = event
        self.hovered = False
        self.thumbnail = load_fresh_pixmap(event.crop_file)
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
        painter.setPen(QPen(border, .7))
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
        self._refresh()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.direct_hover = False
        self.unsetCursor()
        self._refresh()
        super().hoverLeaveEvent(event)


class ClipPreviewChip(QGraphicsRectItem):
    def __init__(self, start_frame, end_frame, index, x, timeline, segment,
                 thumbnail=None, enabled=True, color=None):
        super().__init__(x, 101, 40, 25)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.index = index
        self.timeline = timeline
        self.segment = segment
        self.hovered = False
        self.enabled = enabled
        self.base_color = QColor(color or "#426f96")
        self.thumbnail = thumbnail or QPixmap()
        self.setAcceptHoverEvents(True)
        self.setBrush(self.base_color)
        self.setPen(QPen(self.base_color.lighter(140), 1))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"Preview clip {index:02d}")
        self.setZValue(15)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(1.0 if self.enabled else .38)
        painter.setPen(QPen(
            QColor("#ffffff") if self.hovered else self.base_color.lighter(145), .7))
        painter.setBrush(self.base_color.lighter(120) if self.hovered else self.base_color)
        painter.drawRoundedRect(self.rect(), 4, 4)
        if not self.thumbnail.isNull():
            inner = self.rect().adjusted(2, 2, -2, -2)
            clip = QPainterPath()
            clip.addRoundedRect(inner, 3, 3)
            painter.save()
            painter.setClipPath(clip)
            painter.drawPixmap(inner.toRect(), self.thumbnail)
            painter.restore()
            if self.hovered:
                painter.fillRect(inner, QColor(255, 255, 255, 24))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#ffffff" if self.hovered else "#dbeafe"))
            painter.drawPolygon(QPolygonF(
                [QPointF(-4, -6), QPointF(6, 0), QPointF(-4, 6)]
            ).translated(self.rect().center()))

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
            self.timeline.segmentActivated.emit(
                self.timeline.media.seconds_for_frame(self.start_frame),
                self.timeline.media.seconds_for_frame(self.end_frame),
                self.index)
            event.accept()
            return
        super().mousePressEvent(event)


class TimelineView(QGraphicsView):
    seekRequested = Signal(float)
    cutSelected = Signal(object)
    segmentActivated = Signal(float, float, int)
    contentChanged = Signal()
    faceRemoveRequested = Signal(object)
    segmentRemoveRequested = Signal(int)
    cutAddRequested = Signal(float)

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setScene(QGraphicsScene(self))
        self.media = None
        self.current_seconds = 0.0
        self.content_width = 1000.0
        self.pixels_per_second = 70.0
        self.playhead = None
        self._clip_thumbnail_cache = {}
        self.setMinimumHeight(205)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QColor(BG_MAIN))
        # Extra face lanes normally fit in the unused lower area.  In the rare
        # case where many faces share the same moment, keep every lane reachable.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scene().selectionChanged.connect(self._selection_changed)

    def set_media(self, media):
        if media is not self.media:
            self.current_seconds = 0.0
            self._clip_thumbnail_cache.clear()
        self.media = media
        self.rebuild()

    def _clip_thumbnails(self, frames):
        """Decode tiny timeline thumbnails once and reuse them across rebuilds."""
        if not self.media or not frames:
            return {}
        path = os.path.abspath(self.media.path)
        missing = [int(frame) for frame in frames
                   if (path, int(frame)) not in self._clip_thumbnail_cache]
        if missing:
            cap = cv2.VideoCapture(path)
            try:
                if cap.isOpened():
                    for frame in missing:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                        ok, image = cap.read()
                        if not ok or image is None:
                            self._clip_thumbnail_cache[(path, frame)] = QPixmap()
                            continue
                        height, width = image.shape[:2]
                        target_ratio = 36 / 21
                        if width / max(1, height) > target_ratio:
                            crop_width = max(1, round(height * target_ratio))
                            left = max(0, (width - crop_width) // 2)
                            image = image[:, left:left + crop_width]
                        else:
                            crop_height = max(1, round(width / target_ratio))
                            top = max(0, (height - crop_height) // 2)
                            image = image[top:top + crop_height, :]
                        image = cv2.resize(image, (36, 21), interpolation=cv2.INTER_AREA)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        qimage = QImage(
                            image.data, 36, 21, 36 * 3, QImage.Format.Format_RGB888).copy()
                        self._clip_thumbnail_cache[(path, frame)] = QPixmap.fromImage(qimage)
            finally:
                cap.release()
        return {frame: self._clip_thumbnail_cache.get((path, int(frame)), QPixmap())
                for frame in frames}

    def _segment_colors(self, count):
        """Return stable pseudo-random palette entries with no adjacent repeats."""
        identity = (getattr(self.media, "media_id", "") or
                    os.path.abspath(self.media.path))
        seed = sum((index + 1) * ord(char) for index, char in enumerate(identity))
        state = seed & 0xffffffff
        colors = []
        previous = -1
        for _index in range(max(0, count)):
            state = (1664525 * state + 1013904223) & 0xffffffff
            palette_index = state % len(SCENE_PALETTE)
            if palette_index == previous:
                palette_index = (palette_index + 1 +
                                 ((state >> 8) % (len(SCENE_PALETTE) - 1))) % len(SCENE_PALETTE)
            colors.append(QColor(SCENE_PALETTE[palette_index]))
            previous = palette_index
        return colors

    def rebuild(self):
        self.scene().clear()
        if not self.media:
            text = self.scene().addSimpleText("Import a video to begin")
            text.setBrush(QColor(TEXT_MUTED))
            return
        self.content_width = max(self.viewport().width() - 20,
                                 self.media.duration * self.pixels_per_second)
        scene_height = 175
        self.scene().setSceneRect(0, 0, self.content_width + 44, scene_height)
        self._draw_ruler()
        boundaries = [0, *(event.frame for event in sorted(self.media.cuts, key=lambda cut: cut.frame)),
                      self.media.total_frames]
        clip_frames = [(start + end) // 2 for start, end in zip(boundaries, boundaries[1:])]
        has_sequences = self.media.analyzed or bool(self.media.cuts)
        clip_thumbnails = (self._clip_thumbnails(clip_frames)
                           if (self.media.mode != "faces" or not self.media.analyzed)
                           and has_sequences else {})
        segment_colors = (self._segment_colors(len(boundaries) - 1)
                          if has_sequences else
                          [QColor("#383b40")] * (len(boundaries) - 1))
        segment_height = 57
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
            segment_enabled = index not in self.media.excluded_segments
            x = self.x_for_frame(start)
            width = max(1, self.x_for_frame(end) - x)
            tooltip = (f"Scene {index:02d} · "
                       f"{timecode(self.media.seconds_for_frame(start), self.media.duration)} – "
                       f"{timecode(self.media.seconds_for_frame(end), self.media.duration)}")
            segment = SceneSegmentItem(
                (x, 35, width, segment_height), segment_colors[index - 1], tooltip)
            if not has_sequences:
                segment.setToolTip("")
                segment.setAcceptHoverEvents(False)
            elif not self.media.analyzed:
                segment.setToolTip(
                    f"Clip {index:02d} Â· "
                    f"{timecode(self.media.seconds_for_frame(start), self.media.duration)} â€“ "
                    f"{timecode(self.media.seconds_for_frame(end), self.media.duration)}")
            segment.setOpacity(1.0 if segment_enabled else .38)
            self.scene().addItem(segment)
            header = QGraphicsRectItem(x, 35, width, 5)
            header.setBrush(segment_colors[index - 1].lighter(118))
            header.setPen(Qt.PenStyle.NoPen)
            header.setOpacity(1.0 if segment_enabled else .38)
            header.setZValue(2)
            self.scene().addItem(header)
            if has_sequences and width >= 64:
                label = self.scene().addSimpleText(f"{index:02d}")
                label.setBrush(QColor("#cbd5e1"))
                label.setPos(x + 8, 48)
                label.setOpacity(1.0 if segment_enabled else .38)
                label.setZValue(3)
            if ((self.media.mode != "faces" or not self.media.analyzed) and
                    has_sequences and
                    index not in self.media.removed_segments):
                self.scene().addItem(ClipPreviewChip(
                    start, end, index, x, self, segment,
                    clip_thumbnails.get((start + end) // 2),
                    segment_enabled,
                    segment_colors[index - 1]))
        name = self.scene().addSimpleText(Path(self.media.path).name)
        name.setFont(QFont("JetBrains Mono", 8))
        name.setBrush(QColor(TEXT_MUTED))
        name.setPos(12, 68)
        name.setZValue(2)
        for event in self.media.cuts:
            handle = CutHandle(event, self)
            handle.setPos(self.x_for_frame(event.frame), 28)
            handle.setToolTip(
                f"Cut · {timecode(self.media.seconds_for_frame(event.frame), self.media.duration)}")
            self.scene().addItem(handle)
        if self.media.faces:
            face_y = 136 if self.media.mode != "faces" else 100
            lane_right_edges = []
            for event in sorted(self.media.faces, key=lambda face: face.best_frame):
                start_x = self.x_for_frame(event.best_frame)
                left = start_x
                right = start_x + 44
                lane = next(
                    (index for index, occupied_right in enumerate(lane_right_edges)
                     if left >= occupied_right - 4),
                    len(lane_right_edges),
                )
                if lane == len(lane_right_edges):
                    lane_right_edges.append(right)
                else:
                    lane_right_edges[lane] = right
                chip = FaceChip(event, start_x)
                chip.setY(face_y - 100 + lane * 28)
                self.scene().addItem(chip)
            scene_height = max(scene_height, face_y + len(lane_right_edges) * 28 + 8)
            self.scene().setSceneRect(0, 0, self.content_width + 44, scene_height)
        self.playhead = PlayheadItem(self)
        self.scene().addItem(self.playhead)
        self.playhead.setX(self.x_for_seconds(self.current_seconds))

    def _draw_ruler(self):
        candidates = (.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)
        effective_pps = self.content_width / max(self.media.duration, 0.001)
        step = next((value for value in candidates if value * effective_pps >= 72), candidates[-1])
        moment = 0.0
        minor_step = step / 5
        minor = 0.0
        while minor <= self.media.duration + .001:
            x = self.x_for_seconds(minor)
            tick = QGraphicsLineItem(x, 25, x, 31)
            tick.setPen(QPen(QColor("#454b54"), 1))
            self.scene().addItem(tick)
            minor += minor_step
        while moment <= self.media.duration + .001:
            x = self.x_for_seconds(moment)
            tick = QGraphicsLineItem(x, 18, x, 31)
            tick.setPen(QPen(QColor("#6b7280"), 1))
            self.scene().addItem(tick)
            label = self.scene().addSimpleText(
                transport_timecode(moment, self.media.duration))
            label.setBrush(QColor(TEXT_MUTED))
            label.setPos(x + 4, 4)
            moment += step

    def x_for_frame(self, frame):
        return self.x_for_seconds(self.media.seconds_for_frame(frame))

    def x_for_seconds(self, seconds):
        if not self.media or self.media.duration <= 0:
            return 0.0
        ratio = max(0.0, min(1.0, float(seconds) / self.media.duration))
        return ratio * self.content_width

    def seconds_for_x(self, x):
        if not self.media:
            return 0.0
        ratio = max(0.0, min(1.0, float(x) / max(1.0, self.content_width)))
        return ratio * self.media.duration

    def frame_for_x(self, x):
        return self.media.frame_for_seconds(self.seconds_for_x(x))

    def bounds_for(self, event):
        frames = sorted(cut.frame for cut in self.media.cuts if cut.event_id != event.event_id)
        lower = max((frame + 1 for frame in frames if frame < event.frame), default=1)
        upper = min((frame - 1 for frame in frames if frame > event.frame),
                    default=self.media.total_frames - 1)
        return lower, upper

    def handle_moved(self, handle):
        if not handle.event.original_frame:
            handle.event.original_frame = handle.event.frame
        handle.event.frame = max(1, min(self.media.total_frames - 1, self.frame_for_x(handle.x())))
        handle.setToolTip(
            f"Cut · {timecode(self.media.seconds_for_frame(handle.event.frame), self.media.duration)}")
        self.contentChanged.emit()

    def finish_handle_move(self):
        if not self.media:
            return
        self.media.normalize_cuts()
        self.rebuild()

    def set_position(self, seconds):
        self.current_seconds = max(
            0.0, min(float(seconds), self.media.duration if self.media else 0.0))
        if self.playhead and self.media and not self.playhead.dragging:
            self.playhead.setX(self.x_for_seconds(self.current_seconds))

    def _selection_changed(self):
        selected = self.scene().selectedItems()
        item = selected[0] if selected else None
        self.cutSelected.emit(item.event if isinstance(item, (CutHandle, FaceChip)) else None)

    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and not isinstance(
                item, (CutHandle, PlayheadItem, FaceChip, ClipPreviewChip)) and self.media:
            x = self.mapToScene(event.position().toPoint()).x()
            self.seekRequested.emit(self.seconds_for_x(x))
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, (FaceChip, ClipPreviewChip)):
            menu = QMenu(self)
            menu.setObjectName("mediaContextMenu")
            remove_item = CompactMenuItem(
                themed_icon("fa6s.trash", scale_factor=.85), "Remove", menu)
            remove_action = QWidgetAction(menu)
            remove_action.setDefaultWidget(remove_item)
            menu.addAction(remove_action)
            selected = {"remove": False}

            def select_remove():
                selected["remove"] = True
                menu.close()

            remove_item.activated.connect(select_remove)
            menu.exec(event.globalPos())
            if selected["remove"]:
                if isinstance(item, FaceChip):
                    self.faceRemoveRequested.emit(item.event)
                else:
                    self.segmentRemoveRequested.emit(item.index)
            event.accept()
            return
        if (self.media and (self.media.mode != "faces" or not self.media.analyzed) and
                not isinstance(item, (CutHandle, FaceChip))):
            # The context menu may be opened elsewhere, but the blue playhead is
            # always the precise source of truth for a newly created cut.
            seconds = self.current_seconds
            frame = self.media.frame_for_seconds(seconds)
            if 0 < frame < self.media.total_frames:
                seconds = self.media.seconds_for_frame(frame)
                menu = QMenu(self)
                menu.setObjectName("mediaContextMenu")
                cut_item = CompactMenuItem(
                    themed_icon("fa6s.scissors", scale_factor=.85),
                    "Add new cut", menu)
                cut_action = QWidgetAction(menu)
                cut_action.setDefaultWidget(cut_item)
                menu.addAction(cut_action)
                selected = {"cut": False}

                def select_cut():
                    selected["cut"] = True
                    menu.close()

                cut_item.activated.connect(select_cut)
                menu.exec(event.globalPos())
                if selected["cut"]:
                    self.cutAddRequested.emit(seconds)
                event.accept()
                return
        super().contextMenuEvent(event)

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
        self.selected_segment = None
        self.task = None
        self.batch_mode = False
        self.batch_active = False
        self.processing_media = None
        self._batch_completed_media = set()
        self._manual_face_draft = None
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
        self.video = FaceMarkerVideoWidget()
        self.video.rightClicked.connect(self.show_video_context_menu)
        self.video.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.video.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.video.setStyleSheet("background: transparent;")
        self.video.setMinimumSize(500, 280)
        self.video.hide()
        self.player.setVideoOutput(self.video)
        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.playbackStateChanged.connect(self._update_play_button)
        self.player.mediaStatusChanged.connect(self._media_status_changed)

    def _build_ui(self):
        root = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = root
        root.setChildrenCollapsible(False)
        self.setCentralWidget(root)
        left = QFrame(objectName="mediaPanel")
        layout = QVBoxLayout(left)
        layout.setContentsMargins(13, 12, 13, 12)
        layout.setSpacing(7)
        media_header = QHBoxLayout()
        media_header.setContentsMargins(0, 0, 0, 0)
        media_header.setSpacing(7)
        self.media_title = QLabel("MEDIA", objectName="sectionTitle")
        self.media_title.setTextFormat(Qt.TextFormat.RichText)
        media_header.addWidget(
            self.media_title,
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        media_header.addStretch()
        self.auto_process = QCheckBox("Auto process")
        self.auto_process.setObjectName("batchSwitch")
        self.auto_process.setToolTip(
            "Processes two or more media items sequentially and saves each result in the selected output folder.")
        self.auto_process.toggled.connect(self._batch_mode_changed)
        self.auto_process.hide()
        media_header.addWidget(self.auto_process)
        self.clear_media_button = QToolButton()
        self.clear_media_button.setObjectName("mediaListMenu")
        self.clear_media_button.setIcon(qta.icon(
            "fa6s.bars",
            color="#9ca3af",
            color_active="#9ca3af",
            color_disabled="#9ca3af",
        ))
        self.clear_media_button.setIconSize(QSize(14, 14))
        self.clear_media_button.setFixedSize(26, 26)
        self.clear_media_button.setToolTip("Media list options")
        self.clear_media_button.clicked.connect(self._show_media_list_menu)
        media_header.addWidget(self.clear_media_button)
        layout.addLayout(media_header)
        self.media_list = QListWidget()
        self.media_list.setObjectName("mediaList")
        self.media_list.setIconSize(QSize(13, 13))
        self.media_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.media_list.customContextMenuRequested.connect(self._show_media_context_menu)
        self.media_list.currentRowChanged.connect(self._select_media)
        layout.addWidget(self.media_list)
        media_footer = QHBoxLayout()
        media_footer.setSpacing(6)
        self.import_button = QPushButton("Input")
        self.import_button.setObjectName("secondary")
        self.import_button.clicked.connect(self.import_videos)
        self.output_button = QPushButton("Output")
        self.output_button.setObjectName("secondary")
        self.output_button.setToolTip("Choose where the processed folders will be saved.")
        self.output_button.clicked.connect(self.choose_batch_output)
        media_footer.addWidget(self.import_button, 1)
        media_footer.addWidget(self.output_button, 1)
        layout.addLayout(media_footer)
        root.addWidget(left)

        center = QSplitter(Qt.Orientation.Vertical)
        self.workspace_splitter = center
        viewer = QFrame(objectName="viewer")
        view_layout = QVBoxLayout(viewer)
        self.media_info = QLabel(self.menuBar())
        self.media_info.setObjectName("mediaInfo")
        self.media_info.setTextFormat(Qt.TextFormat.RichText)
        self.media_info.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.media_info.hide()
        self.video_backdrop = VideoBackdrop()
        backdrop_layout = QStackedLayout(self.video_backdrop)
        backdrop_layout.setContentsMargins(0, 0, 0, 0)
        backdrop_layout.addWidget(self.video)
        self.face_review_preview = FaceCropEditor()
        self.face_review_preview.setStyleSheet("background:#000000; border:0;")
        self.face_review_preview.cropChanged.connect(self.update_face_crop_box)
        self.face_review_preview.hide()
        backdrop_layout.addWidget(self.face_review_preview)
        self.processing_preview = ProcessingPreviewLabel()
        self.processing_preview.setStyleSheet("background:#000000; border:0;")
        self.processing_preview.hide()
        backdrop_layout.addWidget(self.processing_preview)
        self.viewer_stack = backdrop_layout
        self._processing_preview_active = False
        view_layout.addWidget(self.video_backdrop, 1)
        self.back_button = QToolButton()
        self.previous_button = QToolButton()
        self.play_button = QToolButton()
        self.following_button = QToolButton()
        self.forward_button = QToolButton()
        self.volume_button = QToolButton()
        back = self.back_button
        previous = self.previous_button
        following = self.following_button
        forward = self.forward_button
        previous.setIcon(themed_icon("fa6s.backward-step", "#9ca3af"))
        back.setIcon(themed_icon("fa6s.backward"))
        self._play_icon = themed_icon("fa6s.play", "#ffffff")
        self._pause_icon = themed_icon("fa6s.pause", "#ffffff")
        self.play_button.setIcon(self._play_icon)
        forward.setIcon(themed_icon("fa6s.forward"))
        following.setIcon(themed_icon("fa6s.forward-step", "#9ca3af"))
        self._volume_icon = themed_icon("fa6s.volume-high", scale_factor=.85)
        self._muted_icon = themed_icon("fa6s.volume-xmark", scale_factor=.85)
        self.volume_button.setIcon(self._volume_icon)
        previous.setObjectName("transportSkip")
        following.setObjectName("transportSkip")
        back.setObjectName("transportStep")
        forward.setObjectName("transportStep")
        self.volume_button.setObjectName("transportStep")
        self.play_button.setObjectName("transportPlay")
        previous.setToolTip("Previous event")
        back.setToolTip("Back 5 seconds")
        self.play_button.setToolTip("Play / Pause")
        forward.setToolTip("Forward 5 seconds")
        following.setToolTip("Next event")
        self.volume_button.setToolTip("Mute audio")
        for button in (previous, back, forward, following, self.volume_button):
            button.setFixedSize(42, 34)
            button.setIconSize(QSize(14, 14))
        self.play_button.setFixedSize(44, 38)
        self.play_button.setIconSize(QSize(15, 15))
        self.volume_button.setIconSize(QSize(15, 15))
        back.clicked.connect(lambda: self.relative_seek(-5000))
        previous.clicked.connect(lambda: self.jump_event(-1))
        self.play_button.clicked.connect(self.toggle_playback)
        following.clicked.connect(lambda: self.jump_event(1))
        forward.clicked.connect(lambda: self.relative_seek(5000))
        self.volume_button.clicked.connect(self.toggle_mute)
        self.audio.mutedChanged.connect(self._update_volume_button)
        self.seek = ScrubSlider(Qt.Orientation.Horizontal)
        self.seek.setObjectName("mediaScrubber")
        self.seek.setMinimumHeight(26)
        self.seek.sliderMoved.connect(self.seek_player_ms)
        self.clock = QLineEdit("00:00")
        self.clock.setObjectName("editableTimecode")
        self.clock.setFixedWidth(72)
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clock.editingFinished.connect(self.seek_from_timecode)
        self.time_separator = QLabel("/")
        self.time_separator.setObjectName("timeSeparator")
        self.time_separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.duration_clock = QLineEdit("00:00")
        self.duration_clock.setObjectName("timecode")
        self.duration_clock.setFixedWidth(72)
        self.duration_clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.duration_clock.setReadOnly(True)
        self.transport_group = QFrame()
        self.transport_group.setObjectName("transportGroup")
        transport_layout = QHBoxLayout(self.transport_group)
        transport_layout.setContentsMargins(6, 5, 8, 5)
        transport_layout.setSpacing(4)
        for widget in (previous, back, self.play_button, forward, following, self.volume_button):
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
        self.timeline.segmentActivated.connect(self.show_segment)
        self.timeline.contentChanged.connect(self._mark_current_unexported)
        self.timeline.faceRemoveRequested.connect(self.remove_face_permanently)
        self.timeline.segmentRemoveRequested.connect(self.remove_segment_permanently)
        self.timeline.cutAddRequested.connect(self.add_cut_at)
        center.addWidget(self.timeline)
        center.setSizes([650, 210])
        root.addWidget(center)

        right = QFrame(objectName="analysisPanel")
        right.setMinimumWidth(210)
        layout = QVBoxLayout(right)
        layout.setContentsMargins(13, 12, 13, 12)
        layout.setSpacing(7)
        analysis_title = QLabel("ANALYSIS", objectName="analysisTitle")
        analysis_title.setIndent(0)
        layout.addWidget(analysis_title)
        for title, widget in self._analysis_fields():
            layout.addWidget(QLabel(title, objectName="muted"))
            layout.addWidget(widget)
        layout.addSpacing(7)
        layout.addWidget(QLabel("SELECTION", objectName="analysisTitle"))
        self.selection_card = QFrame(objectName="selectionCard")
        selection_card_layout = QVBoxLayout(self.selection_card)
        selection_card_layout.setContentsMargins(9, 9, 9, 9)
        selection_card_layout.setSpacing(7)
        self.selection_title = QLabel("No event selected")
        self.selection_detail = QLabel("Select a marker to inspect it.", objectName="muted")
        self.selection_detail.setWordWrap(True)
        self.selection_preview = ProcessingPreviewLabel()
        self.selection_preview.setObjectName("selectionPreview")
        self.selection_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selection_preview.setFixedHeight(150)
        self.selection_preview.hide()
        selection_actions = QHBoxLayout()
        selection_actions.setContentsMargins(0, 0, 0, 0)
        selection_actions.setSpacing(6)
        self.face_previous_button = QToolButton()
        self.face_previous_button.setIcon(themed_icon("fa6s.chevron-left"))
        self.face_previous_button.setToolTip("Previous frame for this face")
        self.face_previous_button.clicked.connect(lambda: self.step_face_frame(-1))
        self.face_next_button = QToolButton()
        self.face_next_button.setIcon(themed_icon("fa6s.chevron-right"))
        self.face_next_button.setToolTip("Next frame for this face")
        self.face_next_button.clicked.connect(lambda: self.step_face_frame(1))
        self.face_reset_button = QToolButton()
        self.face_reset_button.setIcon(themed_icon("fa6s.rotate-left"))
        self.face_reset_button.setToolTip("Restore automatically selected frame")
        self.face_reset_button.clicked.connect(self.reset_face_frame)
        self.face_add_button = QToolButton()
        self.face_add_button.setIcon(themed_icon("fa6s.check", "#86efac"))
        self.face_add_button.setToolTip("Add this face crop")
        self.face_add_button.clicked.connect(self.confirm_manual_face)
        self.face_add_button.hide()
        self.face_cancel_button = QToolButton()
        self.face_cancel_button.setIcon(themed_icon("fa6s.xmark", "#fca5a5"))
        self.face_cancel_button.setToolTip("Cancel manual face crop")
        self.face_cancel_button.clicked.connect(self.cancel_manual_face)
        self.face_cancel_button.hide()
        self.remove_button = QToolButton()
        self.remove_button.setIcon(themed_icon("fa6s.trash"))
        self.remove_button.setToolTip("Remove selected event")
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self.remove_cut)
        self.selection_result_actions = QWidget()
        self.selection_result_actions.setLayout(selection_actions)
        for button in (self.face_previous_button, self.face_next_button,
                       self.face_reset_button, self.face_add_button,
                       self.face_cancel_button, self.remove_button):
            button.setObjectName("selectionAction")
            button.setMinimumSize(26, 30)
            button.setMaximumSize(34, 30)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setIconSize(QSize(14, 14))
            selection_actions.addWidget(button, 1)
        selection_actions.addStretch(1)
        self.preanalysis_actions = QWidget()
        preanalysis_layout = QHBoxLayout(self.preanalysis_actions)
        preanalysis_layout.setContentsMargins(0, 0, 0, 0)
        preanalysis_layout.setSpacing(6)
        self.preanalysis_cut_button = QToolButton()
        self.preanalysis_cut_button.setObjectName("selectionAction")
        self.preanalysis_cut_button.setIcon(themed_icon("fa6s.scissors"))
        self.preanalysis_cut_button.setText("Cut")
        self.preanalysis_cut_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.preanalysis_cut_button.setToolTip("Add cut at playhead (Ctrl+K)")
        self.preanalysis_cut_button.clicked.connect(self.add_cut)
        self.preanalysis_disable_button = QToolButton()
        self.preanalysis_disable_button.setObjectName("selectionAction")
        self.preanalysis_disable_button.setIcon(themed_icon("fa6s.eye-slash"))
        self.preanalysis_disable_button.setText("Disable")
        self.preanalysis_disable_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.preanalysis_disable_button.setToolTip("Disable selected clip")
        self.preanalysis_disable_button.clicked.connect(
            self._toggle_preanalysis_segment)
        for button in (self.preanalysis_cut_button,
                       self.preanalysis_disable_button):
            button.setMinimumHeight(30)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            preanalysis_layout.addWidget(button, 1)
        selection_card_layout.addWidget(self.selection_title)
        selection_card_layout.addWidget(self.selection_detail)
        selection_card_layout.addWidget(self.selection_preview)
        selection_card_layout.addWidget(self.selection_result_actions)
        selection_card_layout.addWidget(self.preanalysis_actions)
        self.selection_result_actions.hide()
        self.selection_card.setEnabled(False)
        layout.addWidget(self.selection_card)
        layout.addStretch()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.valueChanged.connect(self._update_progress_percent)
        self.progress_animation = QPropertyAnimation(self.progress, b"value", self)
        self.progress_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.progress_label = QLabel("Ready", objectName="muted")
        self.progress_percent = QLabel("", objectName="muted")
        self.progress_percent.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.progress_percent.hide()
        self.progress_state_icon = QLabel()
        self.progress_state_icon.setFixedSize(13, 13)
        self.progress_state_icon.hide()
        progress_status_row = QHBoxLayout()
        progress_status_row.setContentsMargins(0, 0, 0, 0)
        progress_status_row.setSpacing(5)
        progress_status_row.addWidget(self.progress_state_icon)
        progress_status_row.addWidget(self.progress_label, 1)
        progress_status_row.addWidget(self.progress_percent)
        self.action_button = QPushButton("Start", objectName="primary")
        self.action_button.clicked.connect(self.toggle_primary_action)
        layout.addLayout(progress_status_row)
        layout.addWidget(self.progress)
        layout.addWidget(self.action_button)
        analysis_scroll = QScrollArea()
        analysis_scroll.setObjectName("analysisScroll")
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setFrameShape(QFrame.Shape.NoFrame)
        analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        analysis_scroll.setWidget(right)
        analysis_scroll.setMinimumWidth(220)
        root.addWidget(analysis_scroll)
        root.setSizes([225, 950, 275])
        root.splitterMoved.connect(lambda _position, _index: self._position_media_info())
        self.setStatusBar(QStatusBar())
        QTimer.singleShot(0, self._position_media_info)

    def _position_media_info(self):
        if not hasattr(self, "media_info") or not hasattr(self, "video_backdrop"):
            return
        menu_bar = self.menuBar()
        left = self.video_backdrop.mapTo(
            menu_bar, self.video_backdrop.rect().topLeft()).x()
        self.media_info.setGeometry(
            left, 0, max(0, self.video_backdrop.width()), menu_bar.height())
        self.media_info.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._position_media_info)

    def _analysis_fields(self):
        self.mode = FramedComboBox()
        self.mode.addItem("Scene detection", "scene")
        self.mode.addItem("Every seconds", "interval")
        self.mode.addItem("Detect faces", "faces")
        self.sensitivity = FramedComboBox()
        self.sensitivity.addItems(PROFILES.keys())
        self.sensitivity.setCurrentText("Normal")
        self.interval = QSpinBox()
        self.interval.setRange(1, 18000)
        self.interval.setValue(10)
        self.interval.setSuffix(" seconds")
        self.accel = FramedComboBox()
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
        self.menuBar().addMenu("&File").addActions(
            [self.import_action, self.load_action, save_action])
        self.menuBar().addMenu("&Edit").addAction(self.add_cut_action)
        self.menuBar().addMenu("&View").addAction(reset_layout_action)
        self.addAction(playback_action)

    def reset_layout(self):
        self.showMaximized()
        QTimer.singleShot(0, self._apply_maximized_layout)

    def _apply_maximized_layout(self):
        width = max(1080, self.main_splitter.width())
        media_width = max(225, round(width * .155))
        analysis_width = max(275, round(width * .19))
        workspace_width = max(500, width - media_width - analysis_width)
        self.main_splitter.setSizes(
            [media_width, workspace_width, analysis_width])

        height = max(700, self.workspace_splitter.height())
        timeline_height = max(205, round(height * .245))
        preview_height = max(280, height - timeline_height)
        self.workspace_splitter.setSizes([preview_height, timeline_height])
        self.timeline.pixels_per_second = 70.0
        self.timeline.rebuild()
        self._position_media_info()
        self.statusBar().showMessage("Default layout restored", 3000)

    def _apply_theme(self):
        image_dir = Path(__file__).parent / "assets" / "images"
        spin_up = (image_dir / "spin-up.svg").as_posix()
        spin_down = (image_dir / "spin-down.svg").as_posix()
        self.setStyleSheet(f"""
        QMainWindow, QWidget {{ background:{BG_MAIN}; color:{TEXT_MAIN}; font-size:10px; }}
        QFrame#panel, QFrame#mediaPanel, QFrame#analysisPanel {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #323438, stop:1 #292b2f); border:1px solid #45484e; border-radius:10px; }}
        QFrame#mediaPanel {{ border-top-left-radius:0; }}
        QFrame#analysisPanel {{ border-top-right-radius:0; }}
        QFrame#selectionCard {{ background:#292b2f; border:1px solid #45484e; border-radius:7px; }}
        QLabel#selectionPreview {{ background:#202226; border:1px solid #45484e; border-radius:6px; }}
        QFrame#selectionCard QLabel {{ background:transparent; border:0; }}
        QFrame#viewer {{ background:#242629; border:1px solid #45484e; border-radius:10px; }}
        QLabel {{ background:transparent; border:0; }}
        QLabel#sectionTitle {{ color:#f3f4f6; font-weight:700; font-size:11px; padding:6px 2px; }}
        QLabel#analysisTitle {{ color:#f3f4f6; font-weight:700; font-size:11px; margin-left:0; padding-left:0; padding-right:0; padding-top:6px; padding-bottom:6px; }}
        QLabel#muted {{ color:{TEXT_MUTED}; }}
        QLabel#mediaInfo {{ color:#d1d5db; font-weight:700; padding:2px 0; }}
        QListWidget {{ background:#1d1f22; border:1px solid #464a50; border-radius:8px; padding:4px; outline:0; selection-background-color:{ACCENT}; }}
        QListWidget::item {{ color:#d1d5db; padding:9px 7px; border-radius:6px; margin:1px; }}
        QListWidget#mediaList::item:hover {{ background:#353941; color:white; border-radius:0; }}
        QListWidget#mediaList::item:selected {{ background:#4a4e55; color:white; border-radius:0; }}
        QListWidget#mediaList QScrollBar:vertical, QListWidget#mediaList QWidget#qt_scrollarea_vcontainer {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #323438, stop:1 #292b2f); border:0; }}
        QListWidget#mediaList QScrollBar:horizontal, QListWidget#mediaList QWidget#qt_scrollarea_hcontainer {{ background:#292b2f; border:0; }}
        QListWidget#mediaList QScrollBar::add-page:horizontal, QListWidget#mediaList QScrollBar::sub-page:horizontal, QListWidget#mediaList QScrollBar::add-page:vertical, QListWidget#mediaList QScrollBar::sub-page:vertical {{ background:transparent; border:0; }}
        QListWidget#mediaList QScrollBar::add-line:horizontal, QListWidget#mediaList QScrollBar::sub-line:horizontal {{ background:transparent; border:0; width:0; }}
        QListWidget#mediaList QScrollBar::add-line:vertical, QListWidget#mediaList QScrollBar::sub-line:vertical {{ background:transparent; border:0; height:0; }}
        QListWidget#mediaList::corner {{ background:#292b2f; border:0; }}
        QScrollArea#analysisScroll,
        QScrollArea#analysisScroll QWidget#qt_scrollarea_viewport {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #323438, stop:1 #292b2f); border:0; }}
        QScrollArea#analysisScroll QScrollBar:vertical,
        QScrollArea#analysisScroll QWidget#qt_scrollarea_vcontainer {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #323438, stop:1 #292b2f); border:0; }}
        QScrollArea#analysisScroll QScrollBar::add-page:vertical,
        QScrollArea#analysisScroll QScrollBar::sub-page:vertical {{ background:transparent; border:0; }}
        QScrollArea#analysisScroll QScrollBar::add-line:vertical,
        QScrollArea#analysisScroll QScrollBar::sub-line:vertical {{ background:transparent; border:0; height:0; }}
        QScrollArea#analysisScroll::corner {{ background:#292b2f; border:0; }}
        QComboBox, QSpinBox {{ min-height:20px; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #292c31, stop:1 #202226); color:#e5e7eb; border:1px solid #4a4e55; border-radius:7px; padding:6px 9px; selection-background-color:{ACCENT}; }}
        QComboBox:hover, QSpinBox:hover {{ border-color:#69717c; background:#2d3136; }}
        QComboBox:focus, QSpinBox:focus {{ border-color:#2d8cff; }}
        QComboBox::drop-down {{ width:25px; border:0; border-left:1px solid #444950; }}
        QComboBox QAbstractItemView {{ background:#25282c; color:#e5e7eb; border:1px solid #555a62; selection-background-color:#286da9; padding:3px; outline:0; }}
        QSpinBox {{ padding-right:26px; }}
        QSpinBox::up-button {{ subcontrol-origin:border; subcontrol-position:top right; width:24px; background:transparent; border:0; }}
        QSpinBox::down-button {{ subcontrol-origin:border; subcontrol-position:bottom right; width:24px; background:transparent; border:0; }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background:transparent; border:0; }}
        QSpinBox::up-arrow {{ image:url("{spin_up}"); width:8px; height:5px; }}
        QSpinBox::down-arrow {{ image:url("{spin_down}"); width:8px; height:5px; }}
        QPushButton {{ min-height:20px; color:#dbe1e8; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #464a50, stop:1 #303339); border:1px solid #555a62; border-radius:8px; padding:7px 12px; font-weight:700; }}
        QPushButton:hover {{ color:white; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #59606a, stop:1 #3c4148); border-color:#737b87; }}
        QPushButton:pressed {{ background:#25282c; border-color:#2d8cff; padding-top:9px; padding-bottom:5px; }}
        QPushButton#primary {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #327fc4, stop:1 #1f538d); border-color:#4b91cf; color:white; }}
        QPushButton#primary:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4295dd, stop:1 #2869ad); border-color:#70b3ed; }}
        QPushButton#warning {{ color:#211b08; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #f3d45f, stop:1 #c99f28); border-color:#f6df7c; }}
        QPushButton#warning:hover {{ color:#171204; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ffe782, stop:1 #ddb43a); border-color:#fff0a5; }}
        QPushButton#danger {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #b84b55, stop:1 #7f2932); border-color:#d26a73; color:white; }}
        QPushButton#danger:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #d25a65, stop:1 #97343e); border-color:#ef8991; }}
        QPushButton#success:disabled {{ color:#eafff3; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #279b61, stop:1 #17683f); border:1px solid #43b979; }}
        QMessageBox QPushButton#confirmRemoveAll:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #d25a65, stop:1 #97343e); border-color:#ef8991; color:white; }}
        QMessageBox QPushButton#confirmRemoveAll:pressed {{ background:#7f2932; border-color:#d26a73; color:white; }}
        QMessageBox QDialogButtonBox {{ padding-top:3px; }}
        QPushButton#secondary {{ color:#cbd5e1; }}
        QCheckBox#batchSwitch {{ color:#d1d5db; spacing:7px; font-size:9px; padding:4px 8px; border:1px solid #4a4e55; border-radius:7px; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #292c31, stop:1 #202226); }}
        QCheckBox#batchSwitch:hover {{ color:white; border-color:#69717c; background:#2d3136; }}
        QCheckBox#batchSwitch:checked {{ border-color:#555a62; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #292c31, stop:1 #202226); }}
        QCheckBox#batchSwitch:checked:hover {{ border-color:#69717c; background:#2d3136; }}
        QCheckBox#batchSwitch:disabled {{ color:#5d626a; border-color:#34373c; background:#222428; }}
        QCheckBox#batchSwitch::indicator {{ width:28px; height:14px; border-radius:7px; border:1px solid #555a62; background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #aeb5bf, stop:0.39 #aeb5bf, stop:0.40 #24272b, stop:1 #24272b); }}
        QCheckBox#batchSwitch::indicator:checked {{ border-color:#4b91cf; background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #214f7b, stop:0.60 #214f7b, stop:0.61 #dbe7f2, stop:1 #dbe7f2); image:none; }}
        QCheckBox#batchSwitch::indicator:disabled {{ border-color:#34373c; background:#292b2f; }}
        QToolButton#mediaListMenu {{ background:transparent; border:0; border-radius:5px; padding:0; }}
        QToolButton#mediaListMenu:hover {{ background:#41454b; }}
        QToolButton#mediaListMenu:pressed {{ background:#25282c; }}
        QToolButton#selectionAction {{ background:#303339; border:1px solid #555a62; border-radius:7px; padding:5px; }}
        QToolButton#selectionAction:hover {{ background:#464a50; border-color:#737b87; }}
        QToolButton#selectionAction:pressed {{ background:#25282c; border-color:#2d8cff; }}
        QPushButton:disabled {{ color:#5d626a; background:#222428; border-color:#34373c; }}
        QComboBox:disabled, QSpinBox:disabled {{ color:#5d626a; background:#222428; border-color:#34373c; }}
        QComboBox:disabled::drop-down {{ border-left-color:#34373c; }}
        QCheckBox:disabled {{ color:#5d626a; background:#222428; border-color:#34373c; }}
        QCheckBox:disabled::indicator {{ border-color:#34373c; background:#292b2f; }}
        QToolButton:disabled {{ color:#565b63; background:#25272a; border-color:#34373c; }}
        QLineEdit:disabled {{ color:#5d626a; background:#222428; border-color:#34373c; }}
        QListWidget:disabled {{ color:#626770; background:#1c1e21; border-color:#34373c; }}
        QListWidget:disabled::item {{ color:#626770; background:transparent; }}
        QSlider:disabled::groove:horizontal {{ background:#35383d; }}
        QSlider:disabled::sub-page:horizontal {{ background:#46505a; }}
        QSlider:disabled::add-page:horizontal {{ background:#35383d; }}
        QSlider:disabled::handle:horizontal {{ background:#626770; border-color:#44484e; }}
        QFrame#transportGroup {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #35383d, stop:1 #25272b); border:1px solid #484c53; border-radius:10px; }}
        QFrame#transportGroup QToolButton {{ color:#d1d5db; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #464a50, stop:1 #303339); border:1px solid #555a62; border-radius:7px; padding:0; font-weight:700; }}
        QFrame#transportGroup QToolButton:hover {{ color:white; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #59606a, stop:1 #3c4148); border-color:#737b87; }}
        QFrame#transportGroup QToolButton:pressed {{ background:#24272b; border-color:#2d8cff; padding-top:2px; }}
        QFrame#transportGroup QToolButton#transportSkip {{ color:#9ca3af; }}
        QFrame#transportGroup QToolButton#transportStep {{ color:#cbd5e1; font-size:10px; }}
        QFrame#transportGroup QToolButton#transportPlay {{ color:white; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #3b96ff, stop:1 #1766b5); border:1px solid #75b8ff; border-radius:9px; font-size:15px; }}
        QFrame#transportGroup QToolButton#transportPlay:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #58a7ff, stop:1 #2378c9); }}
        QSlider#mediaScrubber {{ background:transparent; border:0; padding-left:7px; padding-right:7px; }}
        QSlider#mediaScrubber::groove:horizontal {{ height:4px; background:#565b63; border:0; border-radius:2px; }}
        QSlider#mediaScrubber::sub-page:horizontal {{ background:#2d8cff; border:0; border-radius:1px; }}
        QSlider#mediaScrubber::add-page:horizontal {{ background:#565b63; border:0; border-radius:1px; }}
        QSlider#mediaScrubber::handle:horizontal {{ background:#e5edf6; border:0; width:14px; height:14px; margin:-5px 0; border-radius:7px; }}
        QSlider#mediaScrubber::handle:horizontal:hover {{ background:white; border:0; }}
        QFrame#transportGroup QLineEdit#editableTimecode, QFrame#transportGroup QLineEdit#timecode {{ background:transparent; color:#f3f4f6; border:0; border-radius:0; padding:5px 4px; font-weight:700; }}
        QFrame#transportGroup QLineEdit#editableTimecode:hover, QFrame#transportGroup QLineEdit#editableTimecode:focus {{ background:#2d3136; color:white; border:1px solid #69717c; border-radius:7px; padding:4px 8px; }}
        QFrame#transportGroup QLabel#timeSeparator {{ background:transparent; color:#737b87; border:0; min-width:16px; max-width:16px; padding:5px 0; font-weight:700; }}
        QProgressBar {{ background:#202226; border:1px solid #41454b; border-radius:4px; height:7px; text-align:center; }}
        QProgressBar::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #188a51, stop:1 {SUCCESS}); border-radius:3px; }}
        QMenuBar {{ background:#2b2d31; color:#d1d5db; border-bottom:1px solid #45484e; padding:2px; }}
        QMenuBar::item {{ padding:5px 9px; border-radius:5px; }}
        QMenuBar::item:selected {{ background:#41454b; color:white; }}
        QMenu {{ background:#292c30; color:#e5e7eb; border:1px solid #50545b; padding:5px; }}
        QMenu::item {{ padding:7px 28px 7px 12px; border-radius:5px; }}
        QMenu::item:selected {{ background:#286da9; color:white; }}
        QMenu#mediaContextMenu {{ padding:2px; }}
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
        self._refresh_media_item_states()
        if self.media_list.count() and self.media_list.currentRow() < 0:
            self.media_list.setCurrentRow(0)
        self._update_batch_availability()

    def _show_media_context_menu(self, position):
        if self.task:
            return
        item = self.media_list.itemAt(position)
        if item is None:
            return
        row = self.media_list.row(item)
        menu = QMenu(self.media_list)
        menu.setObjectName("mediaContextMenu")
        remove_item = CompactMenuItem(
            themed_icon("fa6s.trash", scale_factor=.85), "Remove from list", menu)
        remove_item.setToolTip(
            "Removes this video from the list without deleting its file.")
        remove_action = QWidgetAction(menu)
        remove_action.setDefaultWidget(remove_item)
        menu.addAction(remove_action)
        folder_item = CompactMenuItem(
            themed_icon("fa6s.folder-open", scale_factor=.85),
            "Show in folder", menu, hover_color=ACCENT)
        folder_action = QWidgetAction(menu)
        folder_action.setDefaultWidget(folder_item)
        menu.addAction(folder_action)
        selected = {"remove": False, "folder": False}

        def select_remove():
            selected["remove"] = True
            menu.close()

        def select_folder():
            selected["folder"] = True
            menu.close()

        remove_item.activated.connect(select_remove)
        folder_item.activated.connect(select_folder)
        menu.exec(self.media_list.viewport().mapToGlobal(position))
        if selected["remove"]:
            self._remove_media_from_list(row)
        elif selected["folder"]:
            self._reveal_media_file(self.project.media[row].path)

    def _reveal_media_file(self, path):
        target = os.path.normpath(os.path.abspath(path))
        if os.name == "nt":
            QProcess.startDetached("explorer.exe", ["/select,", target])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(target).parent)))

    def _remove_media_from_list(self, row):
        if self.task or not 0 <= row < len(self.project.media):
            return
        removed_media = self.project.media[row]
        removed_name = Path(removed_media.path).name
        removed_current = removed_media is self.current_media

        self.media_list.blockSignals(True)
        self.media_list.takeItem(row)
        self.project.media.pop(row)

        if self.project.media:
            if removed_current or self.current_media not in self.project.media:
                selected_row = min(row, len(self.project.media) - 1)
            else:
                selected_row = self.project.media.index(self.current_media)
            self.media_list.setCurrentRow(selected_row)
        self.media_list.blockSignals(False)

        if self.project.media:
            if removed_current:
                self._select_media(self.media_list.currentRow())
        else:
            self.current_media = None
            self.player.stop()
            self.player.setSource(QUrl())
            self.video.hide()
            self.timeline.set_media(None)
            self.show_cut(None)
            self.seek.setRange(0, 0)
            self.clock.setText("00:00")
            self.duration_clock.setText("00:00")
            self.media_info.clear()
            self.media_info.hide()

        self._update_batch_availability()
        self._refresh_action_button()
        self.statusBar().showMessage(
            f"Removed from list · {removed_name} (the original file was not deleted)", 5000)

    def _clear_media_list(self):
        if self.task or not self.project.media:
            return
        count = len(self.project.media)
        confirmation = QMessageBox(self)
        confirmation.setWindowTitle("Remove all videos")
        confirmation.setIcon(QMessageBox.Icon.Question)
        confirmation.setText(
            f"Remove all {count} videos from the list?\n\n"
            "The original files will not be deleted.")
        confirmation.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        confirmation.setDefaultButton(QMessageBox.StandardButton.No)
        confirmation.button(QMessageBox.StandardButton.Yes).setObjectName(
            "confirmRemoveAll")
        answer = confirmation.exec()
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.player.stop()
        self.player.setSource(QUrl())
        self.video.hide()
        self.project.media.clear()
        self.current_media = None
        self.media_list.clear()
        self.timeline.set_media(None)
        self.show_cut(None)
        self.seek.setRange(0, 0)
        self.clock.setText("00:00")
        self.duration_clock.setText("00:00")
        self.media_info.clear()
        self.media_info.hide()
        self._update_batch_availability()
        self._refresh_action_button()
        self.statusBar().showMessage(
            f"Removed {count} videos from the list · original files were not deleted", 5000)

    def _show_media_list_menu(self):
        if self.task:
            return
        menu = QMenu(self.clear_media_button)
        menu.setObjectName("mediaContextMenu")
        remove_item = CompactMenuItem(
            themed_icon("fa6s.trash", scale_factor=.85),
            "Remove all videos", menu)
        remove_action = QWidgetAction(menu)
        remove_action.setDefaultWidget(remove_item)
        menu.addAction(remove_action)
        auto_item = CompactMenuItem(
            themed_icon(
                "fa6s.toggle-on" if self.auto_process.isChecked() else "fa6s.toggle-off",
                ACCENT, scale_factor=.85),
            "Auto process", menu, hover_color=ACCENT)
        auto_action = QWidgetAction(menu)
        auto_action.setDefaultWidget(auto_item)
        menu.addAction(auto_action)
        selected = {"remove": False, "auto": False}

        def select_remove():
            selected["remove"] = True
            menu.close()

        def select_auto():
            selected["auto"] = True
            menu.close()

        remove_item.activated.connect(select_remove)
        auto_item.activated.connect(select_auto)
        menu.exec(self.clear_media_button.mapToGlobal(
            QPoint(0, self.clear_media_button.height())))
        if selected["remove"]:
            self._clear_media_list()
        elif selected["auto"]:
            self.auto_process.setChecked(not self.auto_process.isChecked())

    def _batch_mode_changed(self, enabled):
        if enabled and len(self.project.media) < 2:
            self.auto_process.blockSignals(True)
            self.auto_process.setChecked(False)
            self.auto_process.blockSignals(False)
            self.batch_mode = False
            self.statusBar().showMessage(
                "A queue requires at least two videos in Media.", 6000)
            self._refresh_action_button()
            return
        self.batch_mode = bool(enabled)
        self._refresh_action_button()
        self.statusBar().showMessage(
            "Automatic queue enabled · processing remains sequential" if enabled
            else "Automatic queue disabled", 4000)

    def _update_batch_availability(self):
        media_count = len(self.project.media)
        available = media_count >= 2
        self.media_title.setText(
            "MEDIA "
            f"<span style='color:#9ca3af; font-size:10px; font-weight:400'>"
            f"({media_count})</span>" if available else "MEDIA")
        self.auto_process.hide()
        self.clear_media_button.show()
        self.clear_media_button.setEnabled(not self.task)
        if not available and self.auto_process.isChecked():
            self.auto_process.setChecked(False)

    def choose_batch_output(self):
        if self.task:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.project.output_folder)
        if folder:
            self.project.output_folder = folder
            self.output_button.setToolTip(f"Output folder: {folder}")
            self.statusBar().showMessage(f"Output · {folder}", 5000)
            self._refresh_action_button()

    def _mode_changed(self):
        mode = self.mode.currentData()
        self.interval.setEnabled(mode == "interval")
        self.sensitivity.setEnabled(mode != "interval")
        if mode == "faces" and self.sensitivity.currentText() == "Auto":
            self.sensitivity.setCurrentText("Normal")
        if self.current_media and self.current_media.mode != mode:
            self.current_media.mode = mode
            self.current_media.analyzed = False
            self.current_media.exported = False
            self.current_media.faces.clear()
            self.selected_cut = None
            self.timeline.set_media(self.current_media)
            self.show_cut(None)
            self._refresh_media_item_states()
        self._set_selection_available(bool(
            self.current_media and self.current_media.analyzed))
        self._refresh_action_button()

    def _select_media(self, row):
        if not 0 <= row < len(self.project.media): return
        self.current_media = self.project.media[row]
        self.preview_end_ms = None
        self.preview_start_ms = None
        self.preview_armed = False
        self.timeline.set_media(self.current_media)
        self._set_selection_available(self.current_media.analyzed)
        self.player.setSource(QUrl.fromLocalFile(self.current_media.path))
        if self._processing_preview_selected():
            self.player.pause()
            self.viewer_stack.setCurrentWidget(self.processing_preview)
            self.video.hide()
            self.processing_preview.show()
        else:
            self.viewer_stack.setCurrentWidget(self.video)
            self.video.show()
        self._update_transport_access()
        self._refresh_action_button()
        self._update_media_info_text()
        duration_text = transport_timecode(
            self.current_media.duration, self.current_media.duration)
        time_width = max(
            72, self.clock.fontMetrics().horizontalAdvance(duration_text) + 24)
        self.clock.setFixedWidth(time_width)
        self.duration_clock.setFixedWidth(time_width)
        self.media_info.setToolTip(self.current_media.path)
        self._position_media_info()
        self.media_info.show()
        self._refresh_progress_state_icon()

    def _update_media_info_text(self):
        if not self.current_media:
            self.media_info.clear()
            return
        fps_text = f"{self.current_media.fps:.2f}".rstrip("0").rstrip(".")
        extension = Path(self.current_media.path).suffix.lstrip(".").upper()
        details = (
            f"{self.current_media.width} × {self.current_media.height} · "
            f"{fps_text} fps · "
            f"{transport_timecode(self.current_media.duration, self.current_media.duration)}")
        if extension:
            details += f" · {extension}"
        if self._processing_preview_selected():
            details += (
                " <span style='color:#9ca3af; font-weight:500'>"
                "· Preview resolution does not represent the original video quality"
                "</span>")
        self.media_info.setText(details)

    def toggle_playback(self):
        if self._processing_preview_selected() and self.task:
            self.task.preview_paused = not self.task.preview_paused
            paused = self.task.preview_paused
            self.play_button.setIcon(self._play_icon if paused else self._pause_icon)
            self.play_button.setToolTip(
                "Continue processing preview" if paused else "Pause processing preview")
            self.statusBar().showMessage(
                "Preview paused · analysis continues" if paused
                else "Preview resumed", 3000)
            return
        if self.viewer_stack.currentWidget() is self.face_review_preview:
            self.face_review_preview.clear_source()
            self.face_review_preview.hide()
            self.viewer_stack.setCurrentWidget(self.video)
            self.video.show()
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState: self.player.pause()
        else: self.player.play()

    def _update_play_button(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setIcon(self._pause_icon if playing else self._play_icon)

    def toggle_mute(self):
        self.audio.setMuted(not self.audio.isMuted())

    def _update_volume_button(self, muted):
        self.volume_button.setIcon(self._muted_icon if muted else self._volume_icon)
        self.volume_button.setToolTip("Unmute audio" if muted else "Mute audio")

    def relative_seek(self, amount):
        self.seek_player_ms(self.player.position() + amount)

    def seek_player_ms(self, value):
        if self.viewer_stack.currentWidget() is self.face_review_preview:
            self.show_cut(None)
        self.preview_end_ms = None
        self.preview_start_ms = None
        self.preview_armed = False
        self.player.setPosition(max(0, min(self.player.duration(), int(value))))

    def _set_selection_available(self, available):
        """Keep Selection reserved for results produced by an analysis."""
        if not hasattr(self, "selection_card"):
            return
        available = bool(available)
        has_media = self.current_media is not None
        self.selection_card.setEnabled(has_media and not self.task)
        self.selection_result_actions.setVisible(available)
        self.preanalysis_actions.setVisible(not available)
        self.preanalysis_cut_button.setEnabled(has_media and not self.task)
        self.preanalysis_disable_button.setEnabled(False)
        if available:
            if self.selected_cut is None and self.selected_segment is None:
                self.selection_title.setText("No event selected")
                self.selection_detail.setText("Select a marker to inspect it.")
            return
        self._manual_face_draft = None
        self.selected_cut = None
        self.selected_segment = None
        self._set_manual_face_controls(False)
        self._set_face_navigation_state(None)
        self.remove_button.setEnabled(False)
        self.video.face_overlay.clear_face()
        self.face_review_preview.clear_source()
        self.face_review_preview.hide()
        self.selection_preview.clear_source()
        self.selection_preview.hide()
        self.selection_title.setText("No event selected")
        self.selection_detail.setText(
            "Add a cut at the playhead or select a clip to disable it.")
        if not self._processing_preview_selected():
            self.viewer_stack.setCurrentWidget(self.video)
            self.video.show()

    def show_segment(self, start_seconds, end_seconds, index):
        if not self.current_media or end_seconds <= start_seconds:
            return
        if not self.current_media.analyzed:
            self._set_selection_available(False)
            self.selected_segment = (int(index), float(start_seconds), float(end_seconds))
            excluded = index in self.current_media.excluded_segments
            self.selection_title.setText(f"Clip {index:02d}")
            self.selection_detail.setText(
                "This clip will be ignored during analysis." if excluded else
                "This clip will be included in the analysis.")
            self.preanalysis_disable_button.setEnabled(not self.task)
            self.preanalysis_disable_button.setIcon(themed_icon(
                "fa6s.eye" if excluded else "fa6s.eye-slash"))
            self.preanalysis_disable_button.setText(
                "Enable" if excluded else "Disable")
            self.preanalysis_disable_button.setToolTip(
                "Enable selected clip" if excluded else "Disable selected clip")
            self.play_segment(start_seconds, end_seconds)
            return
        self._set_selection_available(True)
        self._manual_face_draft = None
        self._set_manual_face_controls(False)
        self.selected_cut = None
        self.selected_segment = (int(index), float(start_seconds), float(end_seconds))
        self._set_face_navigation_state(None)
        cuts = sorted(self.current_media.cuts, key=lambda cut: cut.frame)
        adjacent = []
        if index > 1 and index - 2 < len(cuts): adjacent.append(cuts[index - 2])
        if index - 1 < len(cuts): adjacent.append(cuts[index - 1])
        for cut in adjacent:
            if not cut.original_frame:
                cut.original_frame = cut.frame
        changed = any(cut.frame != cut.original_frame for cut in adjacent)
        self.face_previous_button.setIcon(themed_icon("fa6s.backward-step"))
        self.face_previous_button.setEnabled(True)
        self.face_previous_button.setToolTip("Show first frame")
        self.face_next_button.setIcon(themed_icon("fa6s.forward-step"))
        self.face_next_button.setEnabled(True)
        self.face_next_button.setToolTip("Show last frame")
        self.face_reset_button.setEnabled(changed)
        self.face_reset_button.setToolTip("Restore original sequence boundaries")
        self.remove_button.setEnabled(True)
        excluded = index in self.current_media.excluded_segments
        self.remove_button.setIcon(themed_icon(
            "fa6s.rotate-left" if excluded else "fa6s.trash"))
        self.remove_button.setToolTip(
            "Include sequence" if excluded else "Exclude sequence")
        self.video.face_overlay.clear_face()
        self.face_review_preview.clear_source()
        self.face_review_preview.hide()
        if not self._processing_preview_selected():
            self.viewer_stack.setCurrentWidget(self.video)
            self.video.show()
        self.player.pause()
        midpoint = (start_seconds + end_seconds) / 2
        self.player.setPosition(round(midpoint * 1000))
        duration = end_seconds - start_seconds
        kind = ("Clip" if not self.current_media.analyzed else
                "Scene" if self.current_media.mode == "scene" else "Segment")
        self.selection_title.setText(f"{kind} {index:02d}")
        self.selection_detail.setText(
            f"Start: {timecode(start_seconds, self.current_media.duration)}\n"
            f"End: {timecode(end_seconds, self.current_media.duration)}\n"
            f"Duration: {timecode(duration, duration)}\n"
            f"Status: {'Excluded' if excluded else 'Included'}\n"
            "Move the boundary markers to adjust this range.")
        self.selection_preview.clear_source()
        frame_number = self.current_media.frame_for_seconds(midpoint)
        self._show_segment_still(frame_number)
        self.play_segment(start_seconds, end_seconds)

    def _show_segment_still(self, frame_number):
        if not self.current_media:
            return
        frame_number = max(0, min(int(frame_number), self.current_media.total_frames - 1))
        self.player.pause()
        self.player.setPosition(round(
            self.current_media.seconds_for_frame(frame_number) * 1000))
        ok, frame = False, None
        cap = cv2.VideoCapture(self.current_media.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
            ok, frame = cap.read()
        finally:
            cap.release()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            scale = min(1.0, 440 / max(1, width), 280 / max(1, height))
            if scale < 1.0:
                frame = cv2.resize(
                    frame, (max(1, round(width * scale)),
                            max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview_height, preview_width = frame.shape[:2]
            image = QImage(
                frame.data, preview_width, preview_height, preview_width * 3,
                QImage.Format.Format_RGB888).copy()
            self.selection_preview.set_source_pixmap(QPixmap.fromImage(image))
            self.selection_preview.show()
        else:
            self.selection_preview.hide()

    def play_segment(self, start_seconds, end_seconds):
        start_ms, end_ms = round(start_seconds * 1000), round(end_seconds * 1000)
        if end_ms <= start_ms:
            return
        if self.viewer_stack.currentWidget() is self.face_review_preview:
            self.show_cut(None)
        self.player.pause()
        self.preview_start_ms = start_ms
        self.preview_end_ms = end_ms
        self.preview_armed = False
        self.player.setPosition(start_ms)
        QTimer.singleShot(80, self._ensure_segment_preview_started)
        self.statusBar().showMessage(
            f"Previewing clip · {timecode(start_seconds, self.current_media.duration)} – "
            f"{timecode(end_seconds, self.current_media.duration)}")

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
        current = self.current_media.frame_for_seconds(self.player.position() / 1000)
        frames = [cut.frame for cut in self.current_media.cuts]
        frames.extend(face.best_frame for face in self.current_media.faces if face.enabled)
        frames = sorted(set(frames))
        target = (next((frame for frame in frames if frame > current + 1), None) if direction > 0
                  else next((frame for frame in reversed(frames) if frame < current - 1), None))
        if target is not None:
            self.seek_player_ms(round(self.current_media.seconds_for_frame(target) * 1000))

    def _position_changed(self, value):
        if not self.seek.isSliderDown(): self.seek.setValue(value)
        self.timeline.set_position(value / 1000)
        if not self.clock.hasFocus():
            duration = self.current_media.duration if self.current_media else self.player.duration() / 1000
            self.clock.setText(transport_timecode(value / 1000, duration))
        duration = self.current_media.duration if self.current_media else self.player.duration() / 1000
        self.duration_clock.setText(transport_timecode(self.player.duration() / 1000, duration))
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
        if self.current_media and value > 0:
            source = self.player.source().toLocalFile()
            same_source = bool(source) and os.path.normcase(os.path.abspath(source)) == os.path.normcase(
                os.path.abspath(self.current_media.path))
            actual_duration = value / 1000
            if same_source and abs(self.current_media.duration - actual_duration) >= .001:
                self.current_media.duration = actual_duration
                self.timeline.rebuild()
                duration_text = transport_timecode(actual_duration, actual_duration)
                self._update_media_info_text()
                time_width = max(
                    72, self.clock.fontMetrics().horizontalAdvance(duration_text) + 24)
                self.clock.setFixedWidth(time_width)
                self.duration_clock.setFixedWidth(time_width)
        duration = self.current_media.duration if self.current_media else value / 1000
        self.duration_clock.setText(transport_timecode(value / 1000, duration))

    def _media_status_changed(self, status):
        visible_statuses = {
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.BufferingMedia,
            QMediaPlayer.MediaStatus.StalledMedia,
            QMediaPlayer.MediaStatus.EndOfMedia,
        }
        self.video.setVisible(
            status in visible_statuses and
            self.viewer_stack.currentWidget() is self.video)

    def _begin_processing_preview(self, media):
        if not media:
            return
        self._processing_preview_media = media
        self.processing_media = media
        self._refresh_media_item_states()
        self._processing_restore_position = self.player.position()
        self._processing_preview_active = True
        self.processing_preview.clear_source()
        self.processing_preview.setText("Preparing preview…")
        self.processing_preview.setStyleSheet(
            "background:#000000; color:#8b949e; border:0; font-size:12px;")
        if self.task:
            self.task.preview_paused = False
        if self._processing_preview_selected():
            self.player.pause()
            self.viewer_stack.setCurrentWidget(self.processing_preview)
            self.video.hide()
            self.processing_preview.show()
        self._update_media_info_text()
        self._sync_processing_position(0.0)
        self._update_transport_access()

    def _processing_preview_selected(self):
        return bool(
            self._processing_preview_active and self.current_media is not None and
            self.current_media is getattr(self, "_processing_preview_media", None))

    def _update_transport_access(self):
        locked = self._processing_preview_selected()
        self.timeline.setEnabled(not locked)
        self.seek.setEnabled(not locked)
        self.clock.setEnabled(not locked)
        for button in (self.previous_button, self.back_button, self.forward_button,
                       self.following_button, self.volume_button):
            button.setEnabled(not locked)
        self.play_button.setEnabled(True)
        if locked and self.task:
            paused = self.task.preview_paused
            self.play_button.setIcon(self._play_icon if paused else self._pause_icon)
            self.play_button.setToolTip(
                "Continue processing preview" if paused else "Pause processing preview")
        elif not locked:
            self.play_button.setToolTip("Play / Pause")
            self._update_play_button(self.player.playbackState())

    def _show_processing_frame(self, image, seconds=-1.0):
        if not self._processing_preview_active or image is None:
            return
        try:
            rgb = image.convert("RGB")
            qimage = QImage(
                rgb.tobytes("raw", "RGB"), rgb.width, rgb.height,
                rgb.width * 3, QImage.Format.Format_RGB888).copy()
            self.processing_preview.setText("")
            self.processing_preview.set_source_pixmap(QPixmap.fromImage(qimage))
            if seconds >= 0:
                self._sync_processing_position(seconds)
        except Exception:
            pass

    def _sync_processing_position(self, seconds):
        media = getattr(self, "_processing_preview_media", None)
        if not self._processing_preview_selected() or not media:
            return
        seconds = max(0.0, min(float(media.duration), float(seconds)))
        if not self.seek.isSliderDown():
            self.seek.setValue(round(seconds * 1000))
        self.timeline.set_position(seconds)
        self.clock.setText(transport_timecode(seconds, media.duration))
        self.duration_clock.setText(transport_timecode(media.duration, media.duration))

    def _end_processing_preview(self):
        if not self._processing_preview_active:
            return
        was_selected = self._processing_preview_selected()
        restore = getattr(self, "_processing_restore_position", self.player.position())
        self._processing_preview_active = False
        self._processing_preview_media = None
        self.processing_preview.clear_source()
        self.processing_preview.hide()
        if was_selected:
            self.viewer_stack.setCurrentWidget(self.video)
            self.video.show()
            self.player.setPosition(restore)
        self._update_media_info_text()
        self.play_button.setToolTip("Play / Pause")
        self._update_play_button(self.player.playbackState())
        self._update_transport_access()

    def seek_from_timecode(self):
        seconds = parse_timecode(self.clock.text())
        if seconds is None:
            duration = self.current_media.duration if self.current_media else self.player.duration() / 1000
            self.clock.setText(transport_timecode(self.player.position() / 1000, duration))
            return
        self.seek_player_ms(round(seconds * 1000))

    def add_cut(self):
        if self.task:
            return
        self.add_cut_at(self.player.position() / 1000)

    def add_cut_at(self, seconds):
        if self.task or not self.current_media:
            return
        frame = self.current_media.frame_for_seconds(seconds)
        if 0 < frame < self.current_media.total_frames:
            existing = sorted(cut.frame for cut in self.current_media.cuts)
            if frame in existing:
                return
            split_index = 1 + sum(cut_frame < frame for cut_frame in existing)

            def split_segment_state(values):
                adjusted = []
                for value in values:
                    if value < split_index:
                        adjusted.append(value)
                    elif value == split_index:
                        adjusted.extend((value, value + 1))
                    else:
                        adjusted.append(value + 1)
                return sorted(set(adjusted))

            self.current_media.excluded_segments = split_segment_state(
                self.current_media.excluded_segments)
            self.current_media.removed_segments = split_segment_state(
                self.current_media.removed_segments)
            self.current_media.cuts.append(CutEvent(
                frame=frame, source="manual", original_frame=frame))
            self.current_media.normalize_cuts()
            self._mark_current_unexported()
            self.timeline.rebuild()

    def _mark_current_unexported(self):
        if self.current_media and self.current_media.exported:
            self.current_media.exported = False
            self._refresh_media_item_states()
            self._refresh_action_button()

    def show_video_context_menu(self, point):
        if not (self.current_media and not self.task and
                self.current_media.mode == "faces" and self.current_media.analyzed):
            return
        menu = QMenu(self.video)
        menu.setObjectName("mediaContextMenu")
        crop_item = CompactMenuItem(
            themed_icon("fa6s.scissors", scale_factor=.85),
            "Crop new face", menu, hover_color="#286da9")
        crop_action = QWidgetAction(menu)
        crop_action.setDefaultWidget(crop_item)
        menu.addAction(crop_action)
        selected = {"crop": False}

        def select_crop():
            selected["crop"] = True
            menu.close()

        crop_item.activated.connect(select_crop)
        menu.exec(self.video.mapToGlobal(point.toPoint()))
        if selected["crop"]:
            self.open_manual_face_editor(point)

    def _set_manual_face_controls(self, active):
        for button in (self.face_previous_button, self.face_next_button,
                       self.face_reset_button, self.remove_button):
            button.setVisible(not active)
        self.face_add_button.setVisible(active)
        self.face_cancel_button.setVisible(active)

    def open_manual_face_editor(self, point):
        media = self.current_media
        if not media or self.task or media.mode != "faces" or not media.analyzed:
            return
        source_width, source_height = media.width, media.height
        if source_width <= 0 or source_height <= 0:
            video_size = self.video.videoSink().videoSize()
            source_width, source_height = video_size.width(), video_size.height()
        if source_width <= 0 or source_height <= 0:
            self.statusBar().showMessage("Video dimensions are not available.", 5000)
            return
        scale = min(self.video.width() / source_width, self.video.height() / source_height)
        draw_width, draw_height = source_width * scale, source_height * scale
        offset_x = (self.video.width() - draw_width) / 2
        offset_y = (self.video.height() - draw_height) / 2
        if not QRectF(offset_x, offset_y, draw_width, draw_height).contains(point):
            self.statusBar().showMessage("Right-click inside the video image.", 4000)
            return
        center_x = (point.x() - offset_x) / scale
        center_y = (point.y() - offset_y) / scale
        crop_width = min(source_width * .28, source_height * .42 * .92)
        crop_height = crop_width / .92
        crop_x1 = min(max(0.0, center_x - crop_width / 2), source_width - crop_width)
        crop_y1 = min(max(0.0, center_y - crop_height / 2), source_height - crop_height)
        crop_box = [round(crop_x1), round(crop_y1),
                    round(crop_x1 + crop_width), round(crop_y1 + crop_height)]
        frame = max(1, min(media.total_frames,
                           media.frame_for_seconds(self.player.position() / 1000) + 1))
        event = FaceEvent(
            start_frame=frame, end_frame=frame, best_frame=frame,
            box=list(crop_box), crop_box=list(crop_box),
            review_frames=[{"frame": frame, "box": list(crop_box),
                            "crop_box": list(crop_box)}],
            original_best_frame=frame,
            original_box=list(crop_box), original_crop_box=list(crop_box),
        )
        self._manual_face_draft = event
        self.selected_cut = None
        self.player.pause()
        self._set_manual_face_controls(True)
        self.selection_title.setText("New face crop")
        self.selection_detail.setText(
            f"Frame {frame}\nDrag to move · drag a corner to resize")
        self.selection_preview.clear_source()
        self.selection_preview.hide()
        if not self._show_face_review_frame(event, crop_box):
            self._manual_face_draft = None
            self._set_manual_face_controls(False)
            self.statusBar().showMessage("Could not open this frame for cropping.", 5000)
            self.show_cut(None)

    def cancel_manual_face(self):
        self._manual_face_draft = None
        self._set_manual_face_controls(False)
        self.show_cut(None)

    def confirm_manual_face(self):
        event, media = self._manual_face_draft, self.current_media
        if not event or not media:
            return
        cache = os.path.join(USER_DATA_DIR, "cache", media.media_id)
        os.makedirs(cache, exist_ok=True)
        event.crop_file = os.path.join(cache, f"manual_face_{event.event_id}.png")
        candidate = event.review_frames[0]
        media.faces.append(event)
        self._manual_face_draft = None
        self._set_manual_face_controls(False)
        if not self._apply_face_frame(event, candidate):
            media.faces = [face for face in media.faces if face.event_id != event.event_id]
            self.show_cut(None)
            return
        media.analyzed = True
        self.statusBar().showMessage("Manual face crop added.", 4000)

    def _face_review_frames(self, event):
        frames = [candidate for candidate in event.review_frames
                  if len(candidate.get("box") or ()) == 4 and
                  len(candidate.get("crop_box") or ()) == 4]
        if not frames and len(event.box) == 4 and len(event.crop_box) == 4:
            frames = [{"frame": event.best_frame, "box": list(event.box),
                       "crop_box": list(event.crop_box)}]
        return sorted(frames, key=lambda candidate: int(candidate["frame"]))

    def _set_face_navigation_state(self, event=None):
        self.face_previous_button.setIcon(themed_icon("fa6s.chevron-left"))
        self.face_next_button.setIcon(themed_icon("fa6s.chevron-right"))
        self.face_previous_button.setToolTip("Previous frame for this face")
        self.face_next_button.setToolTip("Next frame for this face")
        self.face_reset_button.setToolTip("Restore automatically selected frame")
        frames = self._face_review_frames(event) if isinstance(event, FaceEvent) else []
        current = next((index for index, candidate in enumerate(frames)
                        if int(candidate["frame"]) == event.best_frame), -1)
        self.face_previous_button.setEnabled(current > 0)
        self.face_next_button.setEnabled(0 <= current < len(frames) - 1)
        changed_from_original = bool(
            frames and event.original_best_frame and (
                event.best_frame != event.original_best_frame or
                (event.original_crop_box and
                 list(event.crop_box) != list(event.original_crop_box))))
        self.face_reset_button.setEnabled(changed_from_original)

    def _apply_face_frame(self, event, candidate):
        if not self.current_media or not event.crop_file:
            return False
        crop_box = [int(value) for value in candidate["crop_box"]]
        ok, frame = False, None
        cap = cv2.VideoCapture(self.current_media.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(candidate["frame"]) - 1))
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            self.statusBar().showMessage("Could not read the selected face frame.", 5000)
            return False
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = crop_box
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            self.statusBar().showMessage("The selected face crop is outside the frame.", 5000)
            return False
        target = Path(event.crop_file)
        temporary = target.with_name(f"{target.stem}.review.tmp{target.suffix or '.png'}")
        if not cv2.imwrite(str(temporary), crop):
            self.statusBar().showMessage("Could not update the selected face crop.", 5000)
            return False
        os.replace(temporary, target)
        event.best_frame = int(candidate["frame"])
        event.box = [int(value) for value in candidate["box"]]
        event.crop_box = crop_box
        self._mark_current_unexported()
        self.timeline.rebuild()
        self.show_cut(event)
        return True

    def step_face_frame(self, direction):
        if self.selected_segment and self.current_media:
            _index, start_seconds, end_seconds = self.selected_segment
            seconds = start_seconds if direction < 0 else end_seconds
            frame = self.current_media.frame_for_seconds(seconds)
            if direction > 0:
                frame = max(0, frame - 1)
            self._show_segment_still(frame)
            return
        event = self.selected_cut
        if not isinstance(event, FaceEvent):
            return
        frames = self._face_review_frames(event)
        current = next((index for index, candidate in enumerate(frames)
                        if int(candidate["frame"]) == event.best_frame), -1)
        target = current + (1 if direction > 0 else -1)
        if 0 <= target < len(frames):
            self._apply_face_frame(event, frames[target])

    def reset_face_frame(self):
        if self.selected_segment and self.current_media:
            index = self.selected_segment[0]
            cuts = sorted(self.current_media.cuts, key=lambda cut: cut.frame)
            adjacent = []
            if index > 1 and index - 2 < len(cuts): adjacent.append(cuts[index - 2])
            if index - 1 < len(cuts): adjacent.append(cuts[index - 1])
            for cut in adjacent:
                if cut.original_frame:
                    cut.frame = cut.original_frame
            self.current_media.normalize_cuts()
            segments = self.current_media.segments()
            self.timeline.rebuild()
            self._mark_current_unexported()
            if 0 < index <= len(segments):
                start, end = segments[index - 1]
                self.show_segment(
                    self.current_media.seconds_for_frame(start),
                    self.current_media.seconds_for_frame(end), index)
            return
        event = self.selected_cut
        if not isinstance(event, FaceEvent) or not event.original_best_frame:
            return
        candidate = next((candidate for candidate in self._face_review_frames(event)
                          if int(candidate["frame"]) == event.original_best_frame), None)
        if candidate:
            original = dict(candidate)
            original["box"] = list(event.original_box or candidate["box"])
            original["crop_box"] = list(
                event.original_crop_box or candidate["crop_box"])
            self._apply_face_frame(event, original)

    def update_face_crop_box(self, crop_box):
        if self._manual_face_draft is not None:
            event = self._manual_face_draft
            event.box = [int(value) for value in crop_box]
            event.crop_box = [int(value) for value in crop_box]
            event.review_frames[0]["box"] = list(event.box)
            event.review_frames[0]["crop_box"] = list(event.crop_box)
            width = max(0, event.crop_box[2] - event.crop_box[0])
            height = max(0, event.crop_box[3] - event.crop_box[1])
            self.selection_detail.setText(
                f"Frame {event.best_frame}\nFinal crop: {width} x {height} px\n"
                "Drag to move · drag a corner to resize")
            return
        event = self.selected_cut
        if not isinstance(event, FaceEvent) or len(crop_box or ()) != 4:
            return
        candidate = {
            "frame": event.best_frame,
            "box": list(event.box),
            "crop_box": [int(value) for value in crop_box],
        }
        if self._apply_face_frame(event, candidate):
            for saved in event.review_frames:
                if int(saved.get("frame", -1)) == event.best_frame:
                    saved["crop_box"] = list(event.crop_box)
                    break

    def _show_face_review_frame(self, event, crop_box):
        """Show one lightweight composed frame; native video overlays are unreliable on Windows."""
        if not self.current_media or len(crop_box or ()) != 4:
            return False
        ok, frame = False, None
        cap = cv2.VideoCapture(self.current_media.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(event.best_frame) - 1))
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            self.face_review_preview.clear_source()
            self.face_review_preview.hide()
            if not self._processing_preview_selected():
                self.viewer_stack.setCurrentWidget(self.video)
                self.video.show()
            return False
        x1, y1, x2, y2 = (int(value) for value in crop_box)
        height, width = frame.shape[:2]
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        scale = min(1.0, 960 / max(1, width), 540 / max(1, height))
        if scale < 1.0:
            frame = cv2.resize(
                frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA)
            x1, y1, x2, y2 = (round(value * scale) for value in (x1, y1, x2, y2))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preview_height, preview_width = frame.shape[:2]
        image = QImage(
            frame.data, preview_width, preview_height, preview_width * 3,
            QImage.Format.Format_RGB888).copy()
        self.face_review_preview.set_face_frame(
            QPixmap.fromImage(image), crop_box, width, height)
        self.viewer_stack.setCurrentWidget(self.face_review_preview)
        self.video.hide()
        self.face_review_preview.show()
        return True

    def show_cut(self, event):
        if not self.current_media or not self.current_media.analyzed:
            self._set_selection_available(False)
            if event is not None and self.current_media:
                seconds = self.current_media.seconds_for_frame(event.frame)
                self.player.setPosition(round(seconds * 1000))
            return
        self._set_selection_available(True)
        self._manual_face_draft = None
        self._set_manual_face_controls(False)
        self.selected_cut = event
        self.selected_segment = None
        self.remove_button.setEnabled(event is not None)
        self._set_face_navigation_state(event)
        if not event or not self.current_media:
            self.video.face_overlay.clear_face()
            self.face_review_preview.clear_source()
            self.face_review_preview.hide()
            if not self._processing_preview_selected():
                self.viewer_stack.setCurrentWidget(self.video)
                self.video.show()
            self.selection_title.setText("No event selected")
            self.selection_detail.setText("Select a marker to inspect it.")
            self.selection_preview.clear_source()
            self.selection_preview.hide()
            return
        if isinstance(event, FaceEvent):
            if not event.original_best_frame:
                event.original_best_frame = event.best_frame
            if not event.original_box:
                event.original_box = list(event.box)
            if not event.original_crop_box:
                event.original_crop_box = list(event.crop_box)
            review_frames = self._face_review_frames(event)
            review_index = next(
                (index for index, candidate in enumerate(review_frames)
                 if int(candidate["frame"]) == event.best_frame), 0)
            seconds = self.current_media.seconds_for_frame(event.best_frame)
            video_size = self.video.videoSink().videoSize()
            source_width = self.current_media.width or video_size.width()
            source_height = self.current_media.height or video_size.height()
            crop_box = event.crop_box or final_face_crop_box(
                event.box, source_width, source_height)
            crop_width = max(0, crop_box[2] - crop_box[0]) if len(crop_box) == 4 else 0
            crop_height = max(0, crop_box[3] - crop_box[1]) if len(crop_box) == 4 else 0
            self.video.face_overlay.clear_face()
            self.player.pause()
            self._show_face_review_frame(event, crop_box)
            self.selection_title.setText("Face event")
            self.selection_detail.setText(
                f"Best frame: {event.best_frame}\n"
                f"Visible: {timecode(self.current_media.seconds_for_frame(event.start_frame), self.current_media.duration)} – "
                f"{timecode(self.current_media.seconds_for_frame(event.end_frame), self.current_media.duration)}\n"
                f"Confidence: {event.confidence:.2f} · Quality: {event.quality:.2f}")
            self.selection_detail.setText(
                self.selection_detail.text() +
                f"\nFinal crop: {crop_width} x {crop_height} px" +
                (f"\nCandidate: {review_index + 1}/{len(review_frames)}"
                 if review_frames else ""))
            self.remove_button.setIcon(themed_icon(
                "fa6s.trash" if event.enabled else "fa6s.rotate-left"))
            self.remove_button.setToolTip(
                "Exclude face" if event.enabled else "Include face")
            self.selection_preview.clear_source()
            pixmap = load_fresh_pixmap(event.crop_file)
            if not pixmap.isNull():
                self.selection_preview.set_source_pixmap(pixmap)
                self.selection_preview.show()
            else:
                self.selection_preview.hide()
        else:
            self.video.face_overlay.clear_face()
            self.face_review_preview.clear_source()
            self.face_review_preview.hide()
            if not self._processing_preview_selected():
                self.viewer_stack.setCurrentWidget(self.video)
                self.video.show()
            seconds = self.current_media.seconds_for_frame(event.frame)
            self.selection_title.setText("Cut marker")
            self.selection_detail.setText(
                f"{timecode(seconds, self.current_media.duration)}\n"
                f"Frame {event.frame}\nSource: {event.source}")
            self.remove_button.setIcon(themed_icon("fa6s.trash"))
            self.remove_button.setToolTip("Remove marker")
            self.selection_preview.clear_source()
            self.selection_preview.hide()
        self.player.setPosition(round(seconds * 1000))

    def _toggle_preanalysis_segment(self):
        if (self.task or not self.current_media or self.current_media.analyzed or
                not self.selected_segment):
            return
        self.remove_cut()

    def remove_cut(self):
        if self.task:
            return
        if not self.current_media:
            return
        if self.selected_segment:
            index, start_seconds, end_seconds = self.selected_segment
            excluded = set(self.current_media.excluded_segments)
            if index in excluded:
                excluded.remove(index)
            else:
                excluded.add(index)
            self.current_media.excluded_segments = sorted(excluded)
            self._mark_current_unexported()
            self.timeline.rebuild()
            self.show_segment(start_seconds, end_seconds, index)
            return
        if not self.selected_cut:
            return
        if isinstance(self.selected_cut, FaceEvent):
            event = self.selected_cut
            event.enabled = not event.enabled
            self._mark_current_unexported()
            self.timeline.rebuild()
            self.show_cut(event)
            return
        ordered_cuts = sorted(self.current_media.cuts, key=lambda cut: cut.frame)
        boundary_index = next(
            (index for index, cut in enumerate(ordered_cuts)
             if cut.event_id == self.selected_cut.event_id), None)
        if boundary_index is not None:
            left_index = boundary_index + 1
            right_index = left_index + 1

            def merged_state(values):
                values = set(values)
                merged = [value for value in values if value < left_index]
                if left_index in values and right_index in values:
                    merged.append(left_index)
                merged.extend(value - 1 for value in values if value > right_index)
                return sorted(set(merged))

            self.current_media.excluded_segments = merged_state(
                self.current_media.excluded_segments)
            self.current_media.removed_segments = merged_state(
                self.current_media.removed_segments)
        self.current_media.cuts = [cut for cut in self.current_media.cuts if cut.event_id != self.selected_cut.event_id]
        self._mark_current_unexported()
        self.selected_cut = None
        self.timeline.rebuild()
        self.show_cut(None)

    def remove_face_permanently(self, event):
        if self.task or not self.current_media or not isinstance(event, FaceEvent):
            return
        before = len(self.current_media.faces)
        self.current_media.faces = [
            face for face in self.current_media.faces
            if face.event_id != event.event_id]
        if len(self.current_media.faces) == before:
            return
        self._mark_current_unexported()
        self.selected_cut = None
        self.timeline.rebuild()
        self.show_cut(None)

    def remove_segment_permanently(self, index):
        if (self.task or not self.current_media or
                (self.current_media.mode == "faces" and self.current_media.analyzed)):
            return
        index = int(index)
        if not 0 < index <= len(self.current_media.segments()):
            return
        removed = set(self.current_media.removed_segments)
        removed.add(index)
        self.current_media.removed_segments = sorted(removed)
        self.current_media.excluded_segments = [
            value for value in self.current_media.excluded_segments if value != index]
        self._mark_current_unexported()
        if self.selected_segment and self.selected_segment[0] == index:
            self.selected_segment = None
            self.show_cut(None)
        self.timeline.rebuild()
        self.statusBar().showMessage("Sequence removed from the project.", 4000)

    def _analysis_edit_plan(self, media):
        segments = list(media.segments())
        excluded_indexes = set(media.excluded_segments)
        removed_indexes = set(media.removed_segments)
        mapping = FrameMap.from_segments(
            segments, excluded_indexes, removed_indexes, media.total_frames)
        excluded_ranges = [segment for index, segment in enumerate(segments, 1)
                           if index in excluded_indexes]
        removed_ranges = [segment for index, segment in enumerate(segments, 1)
                          if index in removed_indexes]
        return mapping, excluded_ranges, removed_ranges

    @staticmethod
    def _mapped_timeline(mapping, analysis_boundaries, mode,
                         excluded_ranges, removed_ranges):
        boundaries, excluded = mapping.rebuild_segments(analysis_boundaries)
        segments = list(zip(boundaries, boundaries[1:]))
        removed = [index for index, (start, end) in enumerate(segments, 1)
                   if any(a <= start and end <= b for a, b in removed_ranges)]
        excluded.extend(
            index for index, (start, end) in enumerate(segments, 1)
            if any(a <= start and end <= b for a, b in excluded_ranges))
        cuts = [CutEvent(frame, mode, original_frame=frame)
                for frame in boundaries[1:-1]]
        return cuts, sorted(set(excluded) - set(removed)), sorted(set(removed))

    @staticmethod
    def _mapped_face_events(engine, mapping, output_root):
        def source_frame(value):
            mapped = mapping.to_source(max(0, int(value) - 1))
            return max(1, int(mapped or 0) + 1)

        events = []
        for face in engine._saved_faces:
            review_frames = []
            for candidate in face.get("review_frames") or []:
                candidate = dict(candidate)
                candidate["frame"] = source_frame(candidate.get("frame", 0))
                review_frames.append(candidate)
            best_frame = source_frame(face.get("best_frame", 0))
            events.append(FaceEvent(
                start_frame=source_frame(face.get("first_frame", 0)),
                end_frame=source_frame(face.get("last_frame", 0)),
                best_frame=best_frame,
                crop_file=os.path.join(
                    engine._output_dir or output_root, face.get("file", "")),
                confidence=float(face.get("confidence", 0)),
                quality=float(face.get("quality", 0)),
                box=list(face.get("box") or []),
                crop_box=list(face.get("crop_box") or []),
                review_frames=review_frames,
                original_best_frame=best_frame,
                original_box=list(face.get("box") or []),
                original_crop_box=list(face.get("crop_box") or []),
            ))
        return events

    def analyze(self):
        if not self.current_media or self.task: return
        if self.batch_mode:
            self.process_batch()
            return
        mode = self.mode.currentData()
        self.player.pause()
        media, interval = self.current_media, self.interval.value()
        media.mode, media.sensitivity = mode, self.sensitivity.currentText()
        frame_map, excluded_ranges, removed_ranges = self._analysis_edit_plan(media)
        if not frame_map.ranges:
            self.statusBar().showMessage(
                "Include at least one sequence before starting analysis.", 5000)
            return
        media.analyzed = False
        media.exported = False
        # Keep the edit plan intact while the worker runs.  Successful results
        # replace it atomically; cancellation or failure leaves it recoverable.
        self.selected_cut = None
        self._set_selection_available(False)
        self.timeline.set_media(media)
        self.show_cut(None)
        accel = self.accel.currentData() or "cpu"
        def work(task):
            cache = os.path.join(USER_DATA_DIR, "cache", media.media_id)
            os.makedirs(cache, exist_ok=True)
            analysis_path = media.path
            proxy_path = os.path.join(cache, "preanalysis_proxy.mp4")
            if not frame_map.is_identity:
                analysis_path = create_analysis_proxy(
                    media.path, proxy_path, frame_map, media.fps,
                    stop_cb=task.isInterruptionRequested)

            def final_timeline(analysis_boundaries):
                return self._mapped_timeline(
                    frame_map, analysis_boundaries, mode,
                    excluded_ranges, removed_ranges)

            try:
                if mode == "faces":
                    previewer = PreviewAdapter(task, frame_map, media.fps)
                    engine = FaceDetectionEngine(
                        analysis_path, cache, LogAdapter(task), ProgressAdapter(task),
                        previewer=previewer, profile=media.sensitivity,
                        accel=accel, preview_enabled=True,
                        reset_frames=frame_map.join_frames)
                    task.register_cancel(engine.stop)
                    if not engine.run():
                        raise RuntimeError("Face analysis stopped before producing results")
                    events = self._mapped_face_events(engine, frame_map, cache)
                    cuts, excluded, removed = final_timeline(frame_map.join_frames)
                    return ("faces", events, cuts, excluded, removed)
                cfg = PROFILES[media.sensitivity].copy()
                cfg.update(ENCODER=accel, INFERENCE="cpu", DEBUG=False)
                if mode == "interval": cfg["FIXED_INTERVAL"] = interval
                engine = SceneEngine(analysis_path, cache, cfg,
                    LogAdapter(task), ProgressAdapter(task),
                    previewer=PreviewAdapter(task, frame_map, media.fps),
                    preview_enabled=True, preview_pool=[analysis_path], export_results=False)
                task.register_cancel(engine.stop)
                if not engine.run(scene_mode=mode == "scene"):
                    raise RuntimeError("Analysis stopped before producing results")
                analysis_boundaries = [start for start, _end in engine.analysis_scenes
                                       if start > 0]
                cuts, excluded, removed = final_timeline(analysis_boundaries)
                return ("cuts", cuts, cuts, excluded, removed)
            finally:
                if analysis_path == proxy_path:
                    try:
                        os.remove(proxy_path)
                    except OSError:
                        pass
        self.start_task(
            work, lambda result, target=media: self.analysis_finished(target, result),
            "Analyzing video…", preview_media=media,
            done_label="Video analyzed")

    def _process_batch_item(self, task, media, edit_plan, output, mode,
                            sensitivity, interval, accel):
        mapping, excluded_ranges, removed_ranges = edit_plan
        if not mapping.ranges:
            raise RuntimeError("No sequence is included for analysis")
        cache = os.path.join(USER_DATA_DIR, "cache", media.media_id)
        os.makedirs(cache, exist_ok=True)
        proxy_path = os.path.join(cache, "preanalysis_proxy.mp4")
        analysis_path = media.path
        if not mapping.is_identity:
            analysis_path = create_analysis_proxy(
                media.path, proxy_path, mapping, media.fps,
                stop_cb=task.isInterruptionRequested)
        try:
            previewer = PreviewAdapter(task, mapping, media.fps)
            if mode == "faces":
                engine = FaceDetectionEngine(
                    analysis_path, output, LogAdapter(task), ProgressAdapter(task),
                    previewer=previewer, profile=sensitivity, accel=accel,
                    preview_enabled=True, reset_frames=mapping.join_frames)
                task.register_cancel(engine.stop)
                if not engine.run():
                    raise RuntimeError("Face processing did not finish")
                events = self._mapped_face_events(engine, mapping, output)
                cuts, excluded, removed = self._mapped_timeline(
                    mapping, mapping.join_frames, mode,
                    excluded_ranges, removed_ranges)
                return media, "faces", events, cuts, excluded, removed

            cfg = PROFILES[sensitivity].copy()
            cfg.update(ENCODER=accel, INFERENCE="cpu", DEBUG=False)
            if mode == "interval":
                cfg["FIXED_INTERVAL"] = interval
            engine = SceneEngine(
                analysis_path, output, cfg, LogAdapter(task), ProgressAdapter(task),
                previewer=previewer, preview_enabled=True,
                preview_pool=[analysis_path], export_results=False)
            task.register_cancel(engine.stop)
            if not engine.run(scene_mode=mode == "scene"):
                raise RuntimeError("Video processing did not finish")
            analysis_boundaries = [start for start, _end in engine.analysis_scenes
                                   if start > 0]
            cuts, excluded, removed = self._mapped_timeline(
                mapping, analysis_boundaries, mode,
                excluded_ranges, removed_ranges)
            boundaries = [0, *(cut.frame for cut in cuts), media.total_frames]
            unavailable = set(excluded) | set(removed)
            export_segments = [segment for segment_index, segment in enumerate(
                zip(boundaries, boundaries[1:]), 1)
                if segment_index not in unavailable]
            exporter = SceneEngine(
                media.path, output, cfg, LogAdapter(task), ProgressAdapter(task),
                preview_enabled=False)
            exporter.scene_mode = mode == "scene"
            exporter._start_time = time.time()
            task.register_cancel(exporter.stop)
            exporter._cut_scenes(export_segments)
            if task.isInterruptionRequested():
                raise RuntimeError("Video processing was cancelled")
            return media, "cuts", cuts, cuts, excluded, removed
        finally:
            if analysis_path == proxy_path:
                try:
                    os.remove(proxy_path)
                except OSError:
                    pass

    def process_batch(self):
        if self.task:
            return
        if len(self.project.media) < 2:
            self.statusBar().showMessage(
                "A queue requires at least two videos in Media.", 6000)
            return
        output = self.project.output_folder
        if not output or not os.path.isdir(output):
            self.statusBar().showMessage(
                "Choose a valid Output folder before starting auto process.", 6000)
            return
        media_queue = list(self.project.media)
        mode = self.mode.currentData()
        sensitivity = self.sensitivity.currentText()
        interval = self.interval.value()
        accel = self.accel.currentData() or "cpu"
        self.player.pause()
        edit_plans = {
            media.media_id: self._analysis_edit_plan(media)
            for media in media_queue
        }
        for media in media_queue:
            media.mode, media.sensitivity = mode, sensitivity
            media.analyzed = False
            media.exported = False
        self.selected_cut = None
        self.timeline.set_media(self.current_media)
        self.show_cut(None)

        def work(task):
            completed, failures = [], []
            total = len(media_queue)
            for index, media in enumerate(media_queue, 1):
                if task.isInterruptionRequested():
                    break
                task.media_started.emit(media)
                task.progress.emit(-1, f"Processing {index}/{total} · {Path(media.path).name}")
                try:
                    completed.append(self._process_batch_item(
                        task, media, edit_plans[media.media_id], output, mode,
                        sensitivity, interval, accel))
                    task.media_finished.emit(media, True)
                except Exception as error:
                    if task.isInterruptionRequested():
                        break
                    failures.append((Path(media.path).name, str(error) or type(error).__name__))
                    task.media_finished.emit(media, False)
            return completed, failures

        self.batch_active = True
        self.start_task(
            work, self.batch_finished, f"Processing queue · {len(media_queue)} videos",
            preview_media=self.current_media)

    def batch_finished(self, result):
        completed, failures = result
        for media, kind, events, cuts, excluded, removed in completed:
            media.analyzed = True
            media.exported = True
            media.cuts = cuts
            media.normalize_cuts()
            media.excluded_segments = excluded
            media.removed_segments = removed
            if kind == "faces":
                media.faces = events
        if self.current_media:
            self.timeline.set_media(self.current_media)
            self._set_selection_available(self.current_media.analyzed)
        self._batch_completed_media.clear()
        self._refresh_media_item_states()
        self._refresh_action_button()
        summary = f"Queue complete · {len(completed)}/{len(self.project.media)} videos processed"
        self.statusBar().showMessage(summary, 10000)
        if failures:
            details = "\n".join(f"• {name}: {message}" for name, message in failures[:8])
            QMessageBox.warning(self, "Queue completed with errors", f"{summary}\n\n{details}")

    def analysis_finished(self, media, result):
        kind, events, cuts, excluded, removed = result
        media.cuts = cuts
        media.normalize_cuts()
        media.excluded_segments = excluded
        media.removed_segments = removed
        if kind == "faces":
            media.faces = events
        media.analyzed = True
        media.exported = False
        self._refresh_media_item_states()
        if self.current_media is media:
            self.timeline.rebuild()
            self._set_selection_available(True)
            self._refresh_action_button()
        included_count = max(0, len(media.segments()) - len(set(excluded) | set(removed)))
        label = f"{len(events)} faces" if kind == "faces" else f"{included_count} segments"
        self.statusBar().showMessage(f"Analysis complete · {label}", 8000)

    def export_result(self):
        if not self.current_media or self.task: return
        if self.current_media.exported:
            self.statusBar().showMessage("Results already exported.", 4000)
            return
        folder = self.project.output_folder
        if not folder or not os.path.isdir(folder):
            folder = QFileDialog.getExistingDirectory(
                self, "Choose export folder", folder or os.path.dirname(self.current_media.path))
            if not folder:
                self.statusBar().showMessage("Export cancelled · no output folder selected", 5000)
                return
            self.project.output_folder = folder
            self.output_button.setToolTip(f"Output folder: {folder}")
        media = self.current_media
        unavailable = set(media.excluded_segments) | set(media.removed_segments)
        segments = [segment for index, segment in enumerate(media.segments(), 1)
                    if index not in unavailable]
        if media.mode != "faces" and not segments:
            self.statusBar().showMessage(
                "Include at least one sequence before exporting.", 5000)
            return
        accel = self.accel.currentData() or "cpu"
        def work(task):
            if media.mode == "faces":
                destination = os.path.join(folder, f"{Path(media.path).stem}_faces_reviewed")
                os.makedirs(destination, exist_ok=True)
                saved = 0
                for event in media.faces:
                    if task.isInterruptionRequested():
                        return None
                    source_frame = max(0, event.best_frame - 1)
                    if (not event.enabled or not os.path.isfile(event.crop_file) or
                            not any(start <= source_frame < end for start, end in segments)):
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
        self.start_task(
            work, lambda output, target=media: self.export_finished(target, output),
            "Exporting…")

    def export_finished(self, media, output):
        media.exported = True
        self._refresh_media_item_states()
        self._refresh_action_button()
        QMessageBox.information(
            self, "Export complete", f"Files were exported to:\n{output}")

    def start_task(self, work, callback, label, preview_media=None, done_label="Complete"):
        if self.task:
            return
        self._set_working_ui(True)
        self._set_progress_value(0, immediate=True)
        self.progress_label.setText(label)
        self._refresh_progress_state_icon()
        task = Task(work, self, done_label=done_label)
        self.task = task
        task.progress.connect(self.task_progress)
        task.preview.connect(self._show_processing_frame)
        task.media_started.connect(self.batch_media_started)
        task.media_finished.connect(self.batch_media_finished)
        task.result.connect(lambda result: self.task_done(task, callback, result))
        task.failed.connect(lambda message: self.task_failed(task, message))
        task.cancelled.connect(lambda: self.task_cancelled(task))
        if preview_media:
            self._begin_processing_preview(preview_media)
        task.start()

    def toggle_primary_action(self):
        if self.task:
            self.stop_current_task()
        elif self.batch_mode:
            self.process_batch()
        elif self.current_media and self.current_media.analyzed:
            self.export_result()
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
        self._refresh_progress_state_icon()
        self.action_button.setEnabled(False)
        task.cancel()
        return True

    def _refresh_action_button(self):
        if not hasattr(self, "action_button") or self.task:
            return
        batch_finished = bool(
            self.batch_mode and self.project.media and
            all(media.exported for media in self.project.media))
        if batch_finished or (not self.batch_mode and self.current_media and
                              self.current_media.exported):
            label = "Finished"
            enabled = False
            button_style = "success"
        elif self.batch_mode:
            label = "Process queue"
            has_output = bool(
                self.project.output_folder and os.path.isdir(self.project.output_folder))
            enabled = bool(self.project.media) and has_output
            button_style = "primary"
        elif self.current_media and self.current_media.analyzed:
            label = "Export"
            enabled = True
            button_style = "warning"
        else:
            label = "Start"
            enabled = self.current_media is not None
            button_style = "primary"
        self.action_button.setText(label)
        self.action_button.setEnabled(enabled)
        self.action_button.setToolTip(
            "Results exported." if button_style == "success" else
            "Choose an Output folder before auto process."
            if self.batch_mode and not enabled else "")
        self.action_button.setObjectName(button_style)
        self.action_button.style().unpolish(self.action_button)
        self.action_button.style().polish(self.action_button)

    def _set_working_ui(self, working):
        self.media_list.setEnabled(True)
        self.import_button.setEnabled(not working)
        self.output_button.setEnabled(not working)
        self.auto_process.setEnabled(not working)
        self.clear_media_button.setEnabled(not working)
        self.import_action.setEnabled(not working)
        self.load_action.setEnabled(not working)
        self.add_cut_action.setEnabled(not working)
        for control in (self.mode, self.sensitivity, self.interval, self.accel):
            control.setEnabled(not working)
        self._update_transport_access()
        if not working:
            self._mode_changed()
        self.action_button.setEnabled(True)
        if working:
            self.action_button.setText("Stop")
            self.action_button.setObjectName("danger")
            self.action_button.style().unpolish(self.action_button)
            self.action_button.style().polish(self.action_button)
        else:
            self._refresh_action_button()

    def _finish_task_ui(self, task, label):
        if self.task is task:
            self.task = None
        task.cancel_callback = None
        self.batch_active = False
        self._mark_processing_media(None)
        self._end_processing_preview()
        self._set_working_ui(False)
        self.progress_label.setText(label)
        self._refresh_progress_state_icon()

    def task_progress(self, value, label):
        if value >= 0:
            self._set_progress_value(round(value * 1000))
        if (value >= 0 and self._processing_preview_selected() and self.task and
                not self.task.preview_paused):
            media = getattr(self, "_processing_preview_media", None)
            if media:
                self._sync_processing_position(float(value) * media.duration)
        if label:
            self.progress_label.setText(label)
            self._refresh_progress_state_icon()

    def _set_progress_value(self, value, immediate=False):
        target = max(self.progress.minimum(), min(self.progress.maximum(), int(value)))
        current = self.progress.value()
        self.progress_animation.stop()
        if immediate or target <= current or target == current:
            self.progress.setValue(target)
            return
        distance = target - current
        self.progress_animation.setDuration(max(90, min(280, 80 + distance // 4)))
        self.progress_animation.setStartValue(current)
        self.progress_animation.setEndValue(target)
        self.progress_animation.start()

    def _update_progress_percent(self, value):
        if value <= self.progress.minimum():
            self.progress_percent.clear()
            self.progress_percent.hide()
            return
        span = max(1, self.progress.maximum() - self.progress.minimum())
        percent = round((value - self.progress.minimum()) * 100 / span)
        self.progress_percent.setText(f"{percent}%")
        self.progress_percent.show()

    def batch_media_started(self, media):
        if not self.batch_active:
            return
        follow_processing = self.processing_media is None or self.current_media is self.processing_media
        self._begin_processing_preview(media)
        self._mark_processing_media(media)
        if not follow_processing:
            return
        try:
            row = next(index for index, item in enumerate(self.project.media) if item is media)
        except StopIteration:
            return
        self.media_list.setCurrentRow(row)

    def batch_media_finished(self, media, succeeded):
        if succeeded:
            self._batch_completed_media.add(media.media_id)
        else:
            self._batch_completed_media.discard(media.media_id)
        self._refresh_media_item_states()

    def _mark_processing_media(self, media):
        self.processing_media = media
        self._refresh_media_item_states()

    def _refresh_media_item_states(self):
        for row, item_media in enumerate(self.project.media):
            item = self.media_list.item(row)
            if not item:
                continue
            name = Path(item_media.path).name
            item.setText(name)
            if item_media is self.processing_media:
                item.setIcon(qta.icon("fa6s.circle", color="#ffffff"))
                item.setToolTip(f"Processing now\n{item_media.path}")
                item.setForeground(QColor("#ffffff"))
            elif item_media.exported or item_media.media_id in self._batch_completed_media:
                item.setIcon(qta.icon("fa6s.circle-check", color="#38c982"))
                item.setToolTip(f"Exported\n{item_media.path}")
                item.setForeground(QColor("#b8e8cf"))
            elif item_media.analyzed:
                item.setIcon(qta.icon("fa6s.circle-info", color="#f2c94c"))
                item.setToolTip(f"Analyzed · ready to export\n{item_media.path}")
                item.setForeground(QColor("#eadb91"))
            else:
                item.setIcon(QIcon())
                item.setToolTip(item_media.path)
                item.setForeground(QColor("#d1d5db"))
        self._refresh_progress_state_icon()

    def _refresh_progress_state_icon(self):
        if not hasattr(self, "progress_state_icon"):
            return
        media = self.current_media
        status = self.progress_label.text()
        if media and media.exported and status == "Complete" and not self.task:
            icon = qta.icon("fa6s.circle-check", color="#38c982")
            self.progress_state_icon.setPixmap(icon.pixmap(QSize(13, 13)))
            self.progress_state_icon.setToolTip(f"Exported\n{media.path}")
            self.progress_state_icon.show()
        elif (media and media.analyzed and not media.exported and
              status == "Video analyzed" and not self.task):
            icon = qta.icon("fa6s.circle-info", color="#f2c94c")
            self.progress_state_icon.setPixmap(icon.pixmap(QSize(13, 13)))
            self.progress_state_icon.setToolTip(
                f"Analyzed · ready to export\n{media.path}")
            self.progress_state_icon.show()
        else:
            self.progress_state_icon.clear()
            self.progress_state_icon.setToolTip("")
            self.progress_state_icon.hide()

    def task_done(self, task, callback, result):
        self._finish_task_ui(task, task.done_label)
        self._set_progress_value(1000)
        callback(result)
        task.deleteLater()

    def task_failed(self, task, message):
        self._batch_completed_media.clear()
        self._finish_task_ui(task, "Failed")
        QMessageBox.critical(self, "Scenespy error", message)
        task.deleteLater()

    def task_cancelled(self, task):
        self._batch_completed_media.clear()
        self._finish_task_ui(task, "Stopped")
        self._set_progress_value(0, immediate=True)
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
        self._refresh_media_item_states()
        self._update_batch_availability()
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
