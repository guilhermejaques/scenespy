"""Production-oriented Qt workspace for non-destructive Scenespy projects."""

from __future__ import annotations

import json
import copy
import os
import sys
import threading
import time
import traceback
import uuid
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import av
import cv2
import qtawesome as qta

from PySide6.QtCore import (QEasingCurve, QEvent, QPoint, QPointF, QProcess, QPropertyAnimation,
                            QRectF, QSettings, QSize, Qt, QThread, QTimer, QUrl, QVariantAnimation,
                            Signal)
from PySide6.QtGui import (QAction, QBrush, QColor, QDesktopServices, QFont, QFontDatabase,
                           QFontMetrics,
                           QIcon, QImage, QKeySequence, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGraphicsItem, QGraphicsLineItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
    QScrollBar, QSizePolicy, QSpinBox, QSplitter, QStackedLayout, QStatusBar, QStyle,
    QStyleOptionFrame,
    QStyleOptionSlider, QToolButton, QVBoxLayout, QWidget, QWidgetAction,
)

from .media import probe_media
from . import __version__
from .project import CutEvent, FaceEvent, MediaItem, ScenespyProject
from .face_engine import FaceDetectionEngine
from .frame_map import FrameMap, create_analysis_proxy, create_playback_proxy
from .scene_engine import SceneEngine
from .shared import (ACCEL_OPTIONS, ACCENT, ALLOWED_VIDEO_EXTENSIONS, BG_CARD, BG_MAIN,
                     BG_PANEL, BORDER_SOFT2, MODE_ACCEL_COMPAT, PROFILES, SUCCESS,
                     TEXT_MAIN, TEXT_MUTED,
                     USER_DATA_DIR, detect_available_encoder_accel, detect_available_inference_accel,
                     status_labels, terminate_all_child_processes)


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

MODE_TOOLTIPS = {
    "scene": "Finds visual scene changes and exports each detected scene as a clip.",
    "interval": "Splits the video into equal-length clips using the seconds value below.",
    "faces": "Scans video frames for human faces and saves the best facial images.",
}

SCENE_THUMBNAIL_SIZE = QSize(72, 42)
FACE_THUMBNAIL_SIZE = QSize(80, 52)
INDICATOR_PLAY_POINTS = (
    QPointF(-2.5, -4), QPointF(4, 0), QPointF(-2.5, 4))


def primary_modifier(platform=None):
    """Return Ctrl on Windows/Linux and Command on macOS."""
    platform = platform or sys.platform
    return (Qt.KeyboardModifier.MetaModifier if platform == "darwin" else
            Qt.KeyboardModifier.ControlModifier)


def primary_shortcut(key, shift=False, platform=None):
    """Build a Ctrl shortcut on Windows/Linux and Command on macOS."""
    modifier = primary_modifier(platform)
    if shift:
        modifier |= Qt.KeyboardModifier.ShiftModifier
    return QKeySequence(modifier | key)


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
    total_seconds = max(0, int(float(seconds or 0)))
    reference_seconds = max(0, float(duration or 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if max(float(seconds or 0), reference_seconds) >= 3600:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def player_timecode(seconds, duration=None):
    """Precise editable time used by the player transport."""
    return timecode(seconds, duration)


def timecode_has_hours(duration):
    return int(round(max(0.0, float(duration or 0)) * 1000)) >= 3_600_000


def scrub_timecode(seconds, duration):
    """Compact time shown while hovering over transport controls."""
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


def face_event_source_frame(event):
    """Convert the face engine's one-based frame number to media coordinates."""
    return max(0, int(event.best_frame) - 1)


def face_event_is_included(media, event):
    """Return whether a face's selected frame belongs to an active segment."""
    if not media or not event:
        return False
    unavailable = set(media.excluded_segments) | set(media.removed_segments)
    source_frame = face_event_source_frame(event)
    return any(
        start <= source_frame < end and index not in unavailable
        for index, (start, end) in enumerate(media.segments(), 1))


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


class MillisecondTimeEdit(QLineEdit):
    """Line edit that de-emphasizes its millisecond suffix when not editing."""
    timeStepped = Signal(float)

    def keyPressEvent(self, event):
        if (event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down) and
                event.modifiers() == Qt.KeyboardModifier.NoModifier):
            current = parse_timecode(self.text())
            if current is not None:
                text = self.text()
                cursor = (self.selectionStart() if self.hasSelectedText()
                          else self.cursorPosition())
                dot = text.rfind(".")
                colons = [index for index, char in enumerate(text) if char == ":"]
                if dot >= 0 and cursor >= dot:
                    amount = .001
                elif colons and cursor > colons[-1]:
                    amount = 1.0
                elif len(colons) >= 2 and cursor > colons[-2]:
                    amount = 60.0
                elif colons:
                    amount = 3600.0 if len(colons) >= 2 else 60.0
                else:
                    amount = 1.0
                direction = 1.0 if event.key() == Qt.Key.Key_Up else -1.0
                self.timeStepped.emit(current + direction * amount)
                self.setCursorPosition(min(cursor, len(self.text())))
                event.accept()
                return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        text = self.text()
        if self.hasFocus() or len(text) < 5 or text[-4] != ".":
            super().paintEvent(event)
            return

        option = QStyleOptionFrame()
        self.initStyleOption(option)
        painter = QPainter(self)
        self.style().drawPrimitive(
            QStyle.PrimitiveElement.PE_PanelLineEdit,
            option, painter, self)
        contents = self.style().subElementRect(
            QStyle.SubElement.SE_LineEditContents,
            option, self).adjusted(4, 0, -4, 0)
        main_text, millis_text = text[:-4], text[-4:]
        metrics = self.fontMetrics()
        main_width = metrics.horizontalAdvance(main_text)
        millis_width = metrics.horizontalAdvance(millis_text)
        x = contents.center().x() - (main_width + millis_width) / 2
        baseline = (contents.center().y() +
                    (metrics.ascent() - metrics.descent()) / 2)
        painter.setFont(self.font())
        painter.setPen(QColor("#f3f4f6" if self.isEnabled() else "#5d626a"))
        painter.drawText(QPointF(x, baseline), main_text)
        painter.setPen(QColor("#7d8794" if self.isEnabled() else "#555a62"))
        painter.drawText(QPointF(x + main_width, baseline), millis_text)
        painter.end()


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
            color = QColor("#ffffff")
        elif self.underMouse():
            color = QColor("#69717c")
        else:
            color = QColor("#4a4e55")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, 1.0))
        left_inset = 1.5 if self.property("protectLeftBorder") else .5
        painter.drawRoundedRect(
            QRectF(self.rect()).adjusted(left_inset, .5, -.5, -.5), 7, 7)
        painter.end()


class IntervalSpinBox(QSpinBox):
    """Seconds editor with a read-only minute conversion in the same field."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSuffix(" seconds")
        self._minutes_label = QLabel(self)
        self._minutes_label.setObjectName("intervalMinutesHint")
        self._minutes_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._minutes_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.valueChanged.connect(self._update_minutes_hint)
        self._update_minutes_hint(self.value())

    def _update_minutes_hint(self, seconds):
        minutes = ("0" if seconds < 60 else
                   f"{seconds / 60:.3f}".rstrip("0").rstrip("."))
        self._minutes_label.setText(f"{minutes} min")
        self._layout_minutes_hint()

    def _layout_minutes_hint(self):
        hint_width = self._minutes_label.sizeHint().width()
        right = 34
        seconds_width = self.fontMetrics().horizontalAdvance(
            f"{self.value()} seconds")
        required_width = seconds_width + hint_width + right + 20
        if self.width() < required_width:
            self._minutes_label.hide()
            self.lineEdit().setTextMargins(0, 0, 0, 0)
            return
        self._minutes_label.show()
        self._minutes_label.setGeometry(
            max(0, self.width() - right - hint_width), 0,
            hint_width, self.height())
        self.lineEdit().setTextMargins(0, 0, hint_width + 14, 0)
        self._minutes_label.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_minutes_hint()


class ShortcutMenu(QMenu):
    """Menu whose shortcut column stays muted, including during hover."""

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        metrics = painter.fontMetrics()
        active_action = self.activeAction()
        for action in self.actions():
            shortcut = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            label = action.text()
            if not shortcut and "\t" in label:
                label, shortcut = label.split("\t", 1)
            if not shortcut or action.isSeparator() or not action.isVisible():
                continue
            item_rect = self.actionGeometry(action)
            active = action is active_action
            painter.fillRect(item_rect, QColor("#286da9" if active else "#292c30"))

            shortcut_width = metrics.horizontalAdvance(shortcut)
            label_rect = item_rect.adjusted(12, 0, -shortcut_width - 36, 0)
            painter.setPen(QColor("#ffffff" if active else "#e5e7eb"))
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter |
                Qt.TextFlag.TextShowMnemonic,
                label,
            )

            shortcut_rect = item_rect.adjusted(
                item_rect.width() - shortcut_width - 24, 0, -12, 0)
            painter.setPen(QColor("#9ca3af"))
            painter.drawText(
                shortcut_rect,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                shortcut,
            )
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

    def setIcon(self, icon):
        self._icon = icon
        self.update()

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
    metrics = Signal(object, object)
    preview = Signal(object, float)
    face_found = Signal(object, object)
    segments_found = Signal(object, object)
    media_started = Signal(object)
    media_analyzed = Signal(object)
    media_finished = Signal(object, bool, object)
    result = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, work, parent=None, done_label="Complete"):
        super().__init__(parent)
        self.work = work
        self.done_label = done_label
        self.cancel_callback = None
        self.preview_paused = False
        self.preview_visible = True
        self.media_progress = {}
        self.live_faces = {}
        self.live_segments = {}

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


class FramePreviewDecoder(QThread):
    """Latest-request-wins decoder with a bounded frame cache for editing."""
    frameReady = Signal(str, int, QImage)

    def __init__(self, parent=None, cache_size=12):
        super().__init__(parent)
        self._condition = threading.Condition()
        self._request = None
        self._serial = 0
        self._stopping = False
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self._container = None
        self._stream = None
        self._source = None
        self._decode_iter = None
        self._last_frame = None

    def request(self, media_id, source, frame, fps):
        with self._condition:
            self._serial += 1
            self._request = (
                self._serial, str(media_id), str(source),
                max(0, int(frame)), max(.001, float(fps)))
            self._condition.notify()

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify()

    def _is_current(self, serial):
        with self._condition:
            return not self._stopping and serial == self._serial

    def _open(self, source):
        if self._source == source and self._container is not None:
            return
        if self._container is not None:
            self._container.close()
        self._container = av.open(source)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._source = source
        self._decode_iter = None
        self._last_frame = None

    @staticmethod
    def _image(frame):
        width, height = frame.width, frame.height
        scale = min(1.0, 960 / max(1, width), 540 / max(1, height))
        width = max(2, round(width * scale) // 2 * 2)
        height = max(2, round(height * scale) // 2 * 2)
        rgb = frame.reformat(width=width, height=height, format="rgb24")
        array = rgb.to_ndarray()
        return QImage(
            array.data, width, height, array.strides[0],
            QImage.Format.Format_RGB888).copy()

    def _remember(self, key, image):
        self._cache[key] = image
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _decode(self, serial, source, target, fps):
        key = (source, target)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        self._open(source)
        can_continue = (self._decode_iter is not None and
                        self._last_frame is not None and
                        self._last_frame < target <= self._last_frame + 48)
        if not can_continue:
            seconds = max(0.0, (target - 2) / fps)
            offset = int(seconds / float(self._stream.time_base))
            self._container.seek(
                offset, stream=self._stream, backward=True, any_frame=False)
            self._decode_iter = iter(self._container.decode(self._stream))
            self._last_frame = None
        for decoded in self._decode_iter:
            if not self._is_current(serial):
                return None
            if decoded.pts is None:
                continue
            index = max(0, round(
                float(decoded.pts * self._stream.time_base) * fps))
            self._last_frame = index
            if index < target:
                continue
            image = self._image(decoded)
            self._remember((source, index), image)
            return image
        return None

    def run(self):
        while True:
            with self._condition:
                while self._request is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    break
                request = self._request
                self._request = None
            serial, media_id, source, frame, fps = request
            try:
                image = self._decode(serial, source, frame, fps)
                if image is not None and self._is_current(serial):
                    self.frameReady.emit(media_id, frame, image)
            except Exception:
                self._decode_iter = None
                self._last_frame = None
        if self._container is not None:
            self._container.close()


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
        return (getattr(self.task, "preview_visible", True) and
                not self.task.preview_paused and
                not self.task.isInterruptionRequested())

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
        self.task.metrics.emit(detected, cut)
    def append_message(self, text, kind="info"): self.task.progress.emit(-1.0, str(text))
    def write_finished(self, text): self.task.progress.emit(1.0, str(text))


class CutHandle(QGraphicsLineItem):
    def __init__(self, event, timeline):
        super().__init__(0, 0, 0, 64)
        self.event, self.timeline = event, timeline
        self._press_frame = int(event.frame)
        self._drag_bounds = None
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
            lower, upper = (self._drag_bounds or
                            self.timeline.bounds_for(self.event))
            return QPointF(
                self.timeline.x_for_frame(max(lower, min(frame, upper))), self.y())
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self.scene():
            self.timeline.handle_moved(self)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._drag_bounds = None
        if int(self.event.frame) != self._press_frame:
            selected_event = self.event
            QTimer.singleShot(
                0, lambda event=selected_event:
                self.timeline.finish_handle_move(event))

    def mousePressEvent(self, event):
        self._press_frame = int(self.event.frame)
        self._drag_bounds = self.timeline.bounds_for(self.event)
        self.timeline.editStarted.emit()
        super().mousePressEvent(event)

class PlayheadItem(QGraphicsItem):
    """Playhead whose hit area follows only its visible needle and head."""
    def __init__(self, timeline):
        super().__init__()
        self.timeline = timeline
        self.length = max(170.0, timeline.scene().sceneRect().height() - 5.0)
        self.dragging = False
        self._last_scrub_emit = 0.0
        self.setZValue(40)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

    def boundingRect(self):
        return QRectF(-10, 0, 20, self.length)

    def set_timeline_height(self, scene_height):
        length = max(170.0, float(scene_height) - 5.0)
        if abs(length - self.length) < 0.5:
            return
        self.prepareGeometryChange()
        self.length = length
        self.update()

    def shape(self):
        path = QPainterPath()
        path.addPolygon(QPolygonF(
            [QPointF(-8, 0), QPointF(8, 0), QPointF(0, 13)]))
        path.closeSubpath()
        path.addRect(QRectF(-1, 12, 2, max(0.0, self.length - 12)))
        return path

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#2d8cff")
        painter.setPen(QPen(color, 2))
        painter.drawLine(QPointF(0, 12), QPointF(0, self.length))
        painter.setPen(QPen(QColor("#bfdbfe"), 1))
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([QPointF(-8, 0), QPointF(8, 0), QPointF(0, 13)]))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            point = QPointF(value)
            return QPointF(max(0, min(point.x(), self.timeline.content_width)), 0)
        if (change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and
                self.scene()):
            self.timeline._update_playhead_indicators(self.x())
            if self.dragging and self.timeline.media:
                now = time.monotonic()
                if now - self._last_scrub_emit >= 1 / 30:
                    self._last_scrub_emit = now
                    self.timeline.seekRequested.emit(
                        self.timeline.seconds_for_x(self.x()))
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        self.dragging = True
        self._last_scrub_emit = 0.0
        self.timeline.scrubStarted.emit()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.dragging = False
        if self.timeline.media:
            self.timeline.scrubFinished.emit(
                self.timeline.seconds_for_x(self.x()))

class FaceChip(QGraphicsRectItem):
    def __init__(self, event, start_x, color, thumbnail=None, live_preview=False):
        super().__init__(start_x, 100, 44, 30)
        self.event = event
        self.live_preview = bool(live_preview)
        self.hovered = False
        self.playhead_active = False
        self.base_color = QColor(color)
        self.thumbnail = (thumbnail if thumbnail is not None else
                          load_fresh_pixmap(event.crop_file))
        if not self.live_preview:
            self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            self.setAcceptHoverEvents(True)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        state = ("Included" if event.enabled else
                 event.rejection_reason or "Not included")
        self.setToolTip(
            f"Face · {state} · available when analysis is complete"
            if self.live_preview else
            f"Face · {state} · click to inspect the best frame")
        self.setZValue(12)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setOpacity(
            .38 if self.live_preview or not self.event.enabled else 1.0)
        selected = self.isSelected()
        active = selected or self.hovered or self.playhead_active
        border = (QColor("#ffffff") if selected else
                  self.base_color.lighter(135 if active else 112))
        painter.setPen(QPen(border, 1))
        painter.setBrush(self.base_color.lighter(112) if active else
                         self.base_color.darker(128))
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
            painter.setBrush(self.base_color.lighter(185))
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

    def set_playhead_active(self, active):
        active = bool(active)
        if active != self.playhead_active:
            self.playhead_active = active
            self.update()


class SceneSegmentItem(QGraphicsRectItem):
    def __init__(self, rect, color, tooltip):
        super().__init__(*rect)
        self.base_color = QColor(color)
        self.direct_hover = False
        self.linked_hover = False
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setToolTip(tooltip)
        self._refresh()

    def _refresh(self):
        active = self.isSelected() or self.direct_hover or self.linked_hover
        brush = QBrush(self.base_color.lighter(120) if active else self.base_color)
        brush.setStyle(Qt.BrushStyle.Dense7Pattern)
        self.setBrush(brush)
        self.setPen(QPen(QColor("#f8fafc" if active else "#50545b"), 1))
        self.setZValue(1)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        super().paint(painter, option, widget)
        header = QRectF(
            self.rect().left() + .5, self.rect().top() + .5,
            max(0.0, self.rect().width() - 1), 5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.base_color.lighter(118))
        painter.drawRect(header)
        if not (self.isSelected() or self.direct_hover or self.linked_hover):
            return
        inner = self.rect().adjusted(1, 1, -1, -1)
        glow = QLinearGradient(inner.topLeft(), inner.bottomLeft())
        glow.setColorAt(0.0, QColor(255, 255, 255, 48))
        glow.setColorAt(0.42, QColor(255, 255, 255, 15))
        glow.setColorAt(1.0, QColor(255, 255, 255, 5))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawRect(inner)

        top_glow = QLinearGradient(
            inner.topLeft(), inner.topRight())
        top_glow.setColorAt(0.0, QColor(255, 255, 255, 55))
        top_glow.setColorAt(0.18, QColor(255, 255, 255, 125))
        top_glow.setColorAt(0.5, QColor(255, 255, 255, 185))
        top_glow.setColorAt(0.82, QColor(255, 255, 255, 125))
        top_glow.setColorAt(1.0, QColor(255, 255, 255, 55))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QBrush(top_glow), 2.5))
        painter.drawLine(
            QPointF(inner.left() + 1, inner.top() + 1),
            QPointF(inner.right() - 1, inner.top() + 1))

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 45), 1.5))
        painter.drawRect(self.rect().adjusted(1.5, 1.5, -1.5, -1.5))

    def set_linked_hover(self, active):
        self.linked_hover = bool(active)
        self._refresh()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._refresh()
        return super().itemChange(change, value)

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
                 thumbnail=None, enabled=True, color=None, live_preview=False):
        super().__init__(x, 101, 40, 25)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.index = index
        self.timeline = timeline
        self.segment = segment
        self.selected = False
        self.hovered = False
        self.playhead_active = False
        self.enabled = enabled
        self.live_preview = bool(live_preview)
        self.base_color = QColor(color or "#426f96")
        self.thumbnail = thumbnail or QPixmap()
        self.setAcceptHoverEvents(not self.live_preview)
        self.setBrush(self.base_color)
        self.setPen(QPen(self.base_color.lighter(140), 1))
        if not self.live_preview:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(
            f"Clip {index:02d} · available when analysis is complete"
            if self.live_preview else f"Preview clip {index:02d}")
        self.setZValue(15)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setOpacity(
            .38 if self.live_preview or not self.enabled else 1.0)
        selected = self.selected
        active = selected or self.hovered or self.playhead_active
        painter.setPen(QPen(
            QColor("#ffffff") if selected else
            self.base_color.lighter(135) if active else self.base_color, 1))
        painter.setBrush(self.base_color.lighter(112) if active else
                         self.base_color.darker(128))
        painter.drawRoundedRect(self.rect(), 4, 4)
        if not self.thumbnail.isNull():
            inner = self.rect().adjusted(2, 2, -2, -2)
            clip = QPainterPath()
            clip.addRoundedRect(inner, 3, 3)
            painter.save()
            painter.setClipPath(clip)
            painter.drawPixmap(inner.toRect(), self.thumbnail)
            painter.restore()
            if active:
                painter.fillRect(inner, QColor(255, 255, 255, 24))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#ffffff"))
            painter.drawPolygon(QPolygonF(
                INDICATOR_PLAY_POINTS).translated(self.rect().center()))
        if self.hovered and not self.thumbnail.isNull():
            center = self.rect().center()
            painter.save()
            painter.setOpacity(1.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#ffffff"))
            painter.drawPolygon(QPolygonF(
                INDICATOR_PLAY_POINTS).translated(center))
            painter.restore()

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

    def set_playhead_active(self, active):
        active = bool(active)
        if active != self.playhead_active:
            self.playhead_active = active
            self.update()

    def mousePressEvent(self, event):
        if self.live_preview:
            event.ignore()
            return
        if self.timeline.media:
            self.timeline.scene().clearSelection()
            for _left, _right, item in self.timeline._indicator_items:
                if isinstance(item, ClipPreviewChip) and item is not self:
                    item.selected = False
                    item.update()
            self.selected = True
            self.update()
            self.timeline.show_clip_duration(
                self.start_frame, self.end_frame, self.index)
            self.timeline.segmentActivated.emit(
                self.timeline.media.seconds_for_frame(self.start_frame),
                self.timeline.media.seconds_for_frame(self.end_frame),
                self.index)
            event.accept()
            return
        super().mousePressEvent(event)


class TimelineZoomScrollBar(QScrollBar):
    """Horizontal navigator with draggable end grips for timeline zoom."""

    def __init__(self, timeline):
        super().__init__(Qt.Orientation.Horizontal, timeline)
        self.timeline = timeline
        self._zoom_edge = None
        self._press_x = 0.0
        self._start_pps = 0.0
        self._anchor_seconds = 0.0
        self._last_zoom_rebuild = 0.0
        self._hover_edge = None
        self.setFixedHeight(16)
        self.setMouseTracking(True)
        self.setStyleSheet("""
            QScrollBar:horizontal {
                height: 16px;
                background: transparent;
                border: 0;
            }
            QScrollBar::groove:horizontal {
                height: 4px;
                background: transparent;
                border: 0;
            }
            QScrollBar::handle:horizontal {
                min-width: 52px;
                background: transparent;
                border: 0;
            }
            QScrollBar::handle:horizontal:hover,
            QScrollBar::handle:horizontal:pressed {
                background: transparent;
                border: 0;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0;
                background: transparent;
                border: 0;
            }
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: transparent;
            }
        """)

    def _slider_rect(self):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        return self.style().subControlRect(
            QStyle.ComplexControl.CC_ScrollBar, option,
            QStyle.SubControl.SC_ScrollBarSlider, self)

    def paintEvent(self, event):
        super().paintEvent(event)
        slider = self._slider_rect()
        if slider.isEmpty():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left_center, right_center, body = self._grip_geometry(slider)
        center_y = float(slider.center().y())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#303339"))
        painter.drawRoundedRect(
            QRectF(4, center_y - 2, max(1, self.width() - 8), 4), 2, 2)
        active = self._zoom_edge is not None
        fill = QLinearGradient(0, body.top(), 0, body.bottom())
        if active:
            fill.setColorAt(0, QColor("#25282c"))
            fill.setColorAt(1, QColor("#25282c"))
        else:
            fill.setColorAt(0, QColor("#464a50"))
            fill.setColorAt(1, QColor("#303339"))
        painter.setPen(QPen(
            QColor("#ffffff") if active else QColor("#555a62"), 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(body, 3, 3)
        radius = 4.25
        painter.drawEllipse(left_center, radius, radius)
        painter.drawEllipse(right_center, radius, radius)

    @staticmethod
    def _grip_geometry(slider):
        radius = 4.25
        gap = 3.0
        center_y = float(slider.center().y())
        left_center = QPointF(slider.left() + radius, center_y)
        right_center = QPointF(slider.right() - radius, center_y)
        body_left = left_center.x() + radius + gap
        body_right = right_center.x() - radius - gap
        body = QRectF(
            body_left, center_y - 3.5,
            max(2.0, body_right - body_left), 7.0)
        return left_center, right_center, body

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            slider = self._slider_rect()
            x = event.position().x()
            left_center, right_center, _body = self._grip_geometry(slider)
            grip = 11
            if abs(x - left_center.x()) <= grip:
                self._begin_zoom("left", x)
                event.accept()
                return
            if abs(x - right_center.x()) <= grip:
                self._begin_zoom("right", x)
                event.accept()
                return
        super().mousePressEvent(event)

    def _begin_zoom(self, edge, x):
        self._zoom_edge = edge
        self._press_x = float(x)
        self._start_pps = float(self.timeline.pixels_per_second)
        self._last_zoom_rebuild = 0.0
        view_width = max(1, self.timeline.viewport().width())
        scene_left = float(self.value())
        anchor_x = scene_left + view_width if edge == "left" else scene_left
        self._anchor_seconds = self.timeline.seconds_for_x(anchor_x)
        self.setCursor(Qt.CursorShape.SizeHorCursor)

    def mouseMoveEvent(self, event):
        if not self._zoom_edge:
            slider = self._slider_rect()
            x = event.position().x()
            left_center, right_center, _body = self._grip_geometry(slider)
            edge = ("left" if abs(x - left_center.x()) <= 11 else
                    "right" if abs(x - right_center.x()) <= 11 else None)
            self._hover_edge = edge
            near_grip = edge is not None
            self.setCursor(Qt.CursorShape.SizeHorCursor if near_grip else
                           Qt.CursorShape.ArrowCursor)
            super().mouseMoveEvent(event)
            return
        delta = float(event.position().x()) - self._press_x
        exponent = delta / 150.0 if self._zoom_edge == "left" else -delta / 150.0
        self.timeline.pixels_per_second = max(
            self.timeline.minimum_pixels_per_second(),
            min(500.0, self._start_pps * (2.0 ** exponent)))
        now = time.monotonic()
        if now - self._last_zoom_rebuild < 1 / 30:
            event.accept()
            return
        self._apply_zoom_frame()
        self._last_zoom_rebuild = now
        event.accept()

    def _apply_zoom_frame(self):
        self.timeline.rebuild()
        anchor_x = self.timeline.x_for_seconds(self._anchor_seconds)
        if self._zoom_edge == "left":
            anchor_x -= self.timeline.viewport().width()
        self.setValue(round(anchor_x))

    def mouseReleaseEvent(self, event):
        if self._zoom_edge and event.button() == Qt.MouseButton.LeftButton:
            self._apply_zoom_frame()
            self._zoom_edge = None
            self.unsetCursor()
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if not self._zoom_edge:
            self._hover_edge = None
            self.unsetCursor()
        super().leaveEvent(event)


class TimelineView(QGraphicsView):
    seekRequested = Signal(float)
    scrubStarted = Signal()
    scrubFinished = Signal(float)
    editStarted = Signal()
    cutSelected = Signal(object)
    segmentActivated = Signal(float, float, int)
    segmentSelected = Signal(float, float, int)
    contentChanged = Signal()
    faceRemoveRequested = Signal(object)
    segmentRemoveRequested = Signal(int)
    cutAddRequested = Signal(float)
    cutRemoveRequested = Signal(object)

    def __init__(self):
        super().__init__()
        self.setObjectName("timelineView")
        self.setHorizontalScrollBar(TimelineZoomScrollBar(self))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setScene(QGraphicsScene(self))
        self.media = None
        self.current_seconds = 0.0
        self.content_width = 1000.0
        self.pixels_per_second = 70.0
        self.playhead = None
        self.show_detected_faces = True
        self.show_saved_faces = True
        self.show_milliseconds = True
        self._indicator_items = []
        self._indicator_lefts = []
        self._active_indicator_items = set()
        self._clip_thumbnail_cache = {}
        self._face_thumbnail_cache = {}
        self._live_face_media_id = None
        self._live_face_ids = set()
        self._live_face_lane_right_edges = []
        self._live_segment_media_id = None
        self._clip_duration_label = None
        self._hover_guide = None
        self._hover_time_bubble = QLabel(None, Qt.WindowType.ToolTip)
        self._hover_time_bubble.setAttribute(
            Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._hover_time_bubble.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hover_time_bubble.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground)
        self._hover_time_bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hover_time_bubble.setFixedHeight(19)
        self._hover_time_bubble.setStyleSheet(
            "QLabel { background:transparent; color:rgba(229,231,235,180); "
            "border:0; padding:1px 3px; font-size:9px; font-weight:600; }")
        self._clip_duration_timer = QTimer(self)
        self._clip_duration_timer.setSingleShot(True)
        self._clip_duration_timer.timeout.connect(self.clear_clip_duration)
        self.setMinimumHeight(205)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setBackgroundBrush(QColor(BG_MAIN))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        # Extra face lanes normally fit in the unused lower area.  In the rare
        # case where many faces share the same moment, keep every lane reachable.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scene().selectionChanged.connect(self._selection_changed)
        # Render the empty state immediately so the import prompt is visible
        # on startup, not only after an action (such as Reset Layout) rebuilds
        # the timeline.
        self.rebuild()

    def set_media(self, media):
        changed = media is not self.media
        if changed:
            self.current_seconds = 0.0
            self._clip_thumbnail_cache.clear()
            self._face_thumbnail_cache.clear()
        self.media = media
        if changed and media:
            self.fit_to_view()
        else:
            self.rebuild()

    def fit_to_view(self):
        if not self.media:
            self.rebuild()
            return
        removed = set(self.media.removed_segments)
        visible_frames = sum(
            end - start
            for index, (start, end) in enumerate(self.media.segments(), 1)
            if index not in removed)
        duration = visible_frames / max(self.media.fps, 0.001)
        self.pixels_per_second = max(
            0.05, (self.viewport().width() - 20) / max(0.001, duration))
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
                        target_width = SCENE_THUMBNAIL_SIZE.width()
                        target_height = SCENE_THUMBNAIL_SIZE.height()
                        target_ratio = target_width / target_height
                        if width / max(1, height) > target_ratio:
                            crop_width = max(1, round(height * target_ratio))
                            left = max(0, (width - crop_width) // 2)
                            image = image[:, left:left + crop_width]
                        else:
                            crop_height = max(1, round(width / target_ratio))
                            top = max(0, (height - crop_height) // 2)
                            image = image[top:top + crop_height, :]
                        image = cv2.resize(
                            image, (target_width, target_height),
                            interpolation=cv2.INTER_AREA)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        qimage = QImage(
                            image.data, target_width, target_height,
                            target_width * 3, QImage.Format.Format_RGB888).copy()
                        self._clip_thumbnail_cache[(path, frame)] = QPixmap.fromImage(qimage)
            finally:
                cap.release()
        return {frame: self._clip_thumbnail_cache.get((path, int(frame)), QPixmap())
                for frame in frames}

    def _face_thumbnails(self, events):
        thumbnails = {}
        for event in events:
            path = os.path.abspath(event.crop_file) if event.crop_file else ""
            if path not in self._face_thumbnail_cache:
                pixmap = load_fresh_pixmap(path)
                if not pixmap.isNull():
                    scaled = pixmap.scaled(
                        FACE_THUMBNAIL_SIZE,
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation)
                    left = max(0, (scaled.width() - FACE_THUMBNAIL_SIZE.width()) // 2)
                    top = max(0, (scaled.height() - FACE_THUMBNAIL_SIZE.height()) // 2)
                    pixmap = scaled.copy(
                        left, top, FACE_THUMBNAIL_SIZE.width(),
                        FACE_THUMBNAIL_SIZE.height())
                self._face_thumbnail_cache[path] = pixmap
            thumbnails[event.event_id] = self._face_thumbnail_cache[path]
        return thumbnails

    def invalidate_face_thumbnail(self, path):
        if path:
            self._face_thumbnail_cache.pop(os.path.abspath(path), None)

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

    def _face_colors(self, events):
        """Give each face a varied color that remains stable across rebuilds."""
        media_identity = (getattr(self.media, "media_id", "") or
                          os.path.abspath(self.media.path))
        colors = []
        previous = -1
        for event in events:
            identity = f"{media_identity}:{event.event_id}"
            seed = sum((index + 1) * ord(char)
                       for index, char in enumerate(identity))
            palette_index = seed % len(SCENE_PALETTE)
            if palette_index == previous:
                palette_index = (palette_index + 1) % len(SCENE_PALETTE)
            colors.append(QColor(SCENE_PALETTE[palette_index]))
            previous = palette_index
        return colors

    def rebuild(self):
        self._live_face_media_id = None
        self._live_face_ids = set()
        self._live_face_lane_right_edges = []
        self._live_segment_media_id = None
        self._clip_duration_label = None
        self._hover_guide = None
        self._hover_time_bubble.hide()
        self.scene().clear()
        self._indicator_items = []
        self._indicator_lefts = []
        self._active_indicator_items = set()
        if not self.media:
            text = self.scene().addSimpleText("Import a video to begin")
            text.setBrush(QColor(TEXT_MUTED))
            return
        boundaries = [0, *(event.frame for event in sorted(
            self.media.cuts, key=lambda cut: cut.frame)), self.media.total_frames]
        removed = set(self.media.removed_segments)
        self._visible_layout = []
        display_start = 0.0
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
            if index in removed:
                continue
            self._visible_layout.append((start, end, index, display_start))
            display_start += max(
                0.0,
                self.media.seconds_for_frame(end) -
                self.media.seconds_for_frame(start),
            )
        self._display_duration = display_start
        self.content_width = max(self.viewport().width() - 20,
                                 self._display_duration * self.pixels_per_second)
        scene_height = 175
        self.scene().setSceneRect(0, 0, self.content_width + 44, scene_height)
        self._draw_ruler()
        clip_frames = [(start + end) // 2 for start, end, _index, _display
                       in self._visible_layout]
        has_sequences = self.media.analyzed or bool(self.media.cuts)
        clip_thumbnails = (self._clip_thumbnails(clip_frames)
                           if self.media.analyzed and
                           self.media.mode != "faces" else {})
        segment_colors = (self._segment_colors(len(boundaries) - 1)
                          if has_sequences else
                          [QColor("#383b40")] * (len(boundaries) - 1))
        segment_height = 57
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
            if index in removed:
                continue
            segment_enabled = index not in self.media.excluded_segments
            x = self.x_for_frame(start)
            width = max(1, self.x_for_frame(end) - x)
            tooltip = (f"Scene {index:02d} · "
                       f"{timecode(self.media.seconds_for_frame(start), self.media.duration)} – "
                       f"{timecode(self.media.seconds_for_frame(end), self.media.duration)}")
            segment = SceneSegmentItem(
                (x, 35, width, segment_height), segment_colors[index - 1], tooltip)
            segment.start_frame = start
            segment.end_frame = end
            segment.index = index
            if not has_sequences:
                # The initial full-length clip keeps the standard timeline
                # treatment without pretending an analysis result exists.
                segment.setToolTip("")
            elif not self.media.analyzed:
                segment.setToolTip(
                    f"Clip {index:02d} · "
                    f"{timecode(self.media.seconds_for_frame(start), self.media.duration)} – "
                    f"{timecode(self.media.seconds_for_frame(end), self.media.duration)}")
            segment.setOpacity(1.0 if segment_enabled else .38)
            self.scene().addItem(segment)
            if has_sequences and width >= 64:
                label = self.scene().addSimpleText(f"{index:02d}")
                label.setBrush(QColor("#cbd5e1"))
                label.setPos(x + 8, 48)
                label.setOpacity(1.0 if segment_enabled else .38)
                label.setZValue(3)
            if (self.media.analyzed and self.media.mode != "faces" and
                    index not in self.media.removed_segments):
                chip = ClipPreviewChip(
                    start, end, index, x, self, segment,
                    clip_thumbnails.get((start + end) // 2),
                    segment_enabled,
                    segment_colors[index - 1])
                self.scene().addItem(chip)
                self._indicator_items.append((x, x + 40, chip))
        name = self.scene().addSimpleText(Path(self.media.path).name)
        name.setFont(QFont("JetBrains Mono", 8))
        name.setBrush(QColor(TEXT_MUTED))
        name.setPos(12, 68)
        name.setZValue(2)
        drawn_handle_positions = set()
        ordered_cuts = sorted(self.media.cuts, key=lambda cut: cut.frame)
        for boundary_index, event in enumerate(ordered_cuts, 1):
            # A removed interval is already a ripple join.  Its source-side
            # boundaries collapse onto the same display point and must not be
            # draggable as if they were an ordinary visible cut.
            if (boundary_index in removed or
                    boundary_index + 1 in removed):
                continue
            handle_x = self.x_for_frame(event.frame)
            position_key = round(handle_x, 3)
            if position_key in drawn_handle_positions:
                continue
            drawn_handle_positions.add(position_key)
            handle = CutHandle(event, self)
            handle.setPos(handle_x, 28)
            handle.setToolTip(
                f"Cut · {timecode(self.media.seconds_for_frame(event.frame), self.media.duration)}")
            self.scene().addItem(handle)
        if self.media.faces:
            face_y = 136 if self.media.mode != "faces" else 100
            lane_right_edges = []
            face_events = sorted(
                (face for face in self.media.faces
                 if (face_event_is_included(self.media, face) and
                     ((face.enabled and self.show_saved_faces) or
                      (not face.enabled and self.show_detected_faces)))),
                key=lambda face: face.best_frame)
            face_colors = self._face_colors(face_events)
            face_thumbnails = self._face_thumbnails(face_events)
            for event, face_color in zip(face_events, face_colors):
                start_x = self.x_for_frame(face_event_source_frame(event))
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
                chip = FaceChip(
                    event, start_x, face_color,
                    face_thumbnails.get(event.event_id))
                chip.setY(face_y - 100 + lane * 28)
                self.scene().addItem(chip)
                self._indicator_items.append((start_x, start_x + 44, chip))
            scene_height = max(scene_height, face_y + len(lane_right_edges) * 28 + 8)
            self.scene().setSceneRect(0, 0, self.content_width + 44, scene_height)
        self.playhead = PlayheadItem(self)
        self.scene().addItem(self.playhead)
        self.playhead.setX(self.x_for_seconds(self.current_seconds))
        self._indicator_items.sort(key=lambda entry: entry[0])
        self._indicator_lefts = [entry[0] for entry in self._indicator_items]
        self._update_playhead_indicators(self.playhead.x())

    def begin_live_faces(self, media):
        """Prepare a transient result layer without touching project data."""
        if media is not self.media:
            return
        for item in list(self.scene().items()):
            if isinstance(item, FaceChip):
                self.scene().removeItem(item)
        self._indicator_items = [
            entry for entry in self._indicator_items
            if not isinstance(entry[2], FaceChip)]
        self._indicator_items.sort(key=lambda entry: entry[0])
        self._indicator_lefts = [entry[0] for entry in self._indicator_items]
        self._active_indicator_items = {
            item for item in self._active_indicator_items
            if not isinstance(item, FaceChip)}
        self._live_face_media_id = media.media_id
        self._live_face_ids = set()
        self._live_face_lane_right_edges = []

    def append_live_face(self, media, event):
        """Append one finalized face track; never rebuild the full timeline."""
        if media is not self.media:
            return
        if self._live_face_media_id != media.media_id:
            self.begin_live_faces(media)
        if event.event_id in self._live_face_ids:
            return
        self._live_face_ids.add(event.event_id)
        if not face_event_is_included(media, event):
            return
        if ((event.enabled and not self.show_saved_faces) or
                (not event.enabled and not self.show_detected_faces)):
            return
        start_x = self.x_for_frame(face_event_source_frame(event))
        right = start_x + 44
        lane = next((
            index for index, occupied_right in
            enumerate(self._live_face_lane_right_edges)
            if start_x >= occupied_right - 4),
            len(self._live_face_lane_right_edges))
        if lane == len(self._live_face_lane_right_edges):
            self._live_face_lane_right_edges.append(right)
        else:
            self._live_face_lane_right_edges[lane] = right
        thumbnail = self._face_thumbnails([event]).get(event.event_id)
        chip = FaceChip(
            event, start_x, self._face_colors([event])[0], thumbnail,
            live_preview=True)
        chip.setY(lane * 28)
        self.scene().addItem(chip)
        insert_at = bisect_right(self._indicator_lefts, start_x)
        self._indicator_lefts.insert(insert_at, start_x)
        self._indicator_items.insert(insert_at, (start_x, right, chip))
        scene_rect = self.sceneRect()
        required_height = max(175, 108 + len(self._live_face_lane_right_edges) * 28)
        if required_height > scene_rect.height():
            self.scene().setSceneRect(
                scene_rect.x(), scene_rect.y(), scene_rect.width(), required_height)
        if self.playhead:
            self.playhead.set_timeline_height(self.scene().sceneRect().height())
            self._update_playhead_indicators(self.playhead.x())

    def begin_live_segments(self, media):
        """Clear old clip indicators while preserving the project timeline."""
        if media is not self.media:
            return
        for item in list(self.scene().items()):
            if isinstance(item, ClipPreviewChip):
                self.scene().removeItem(item)
        self._indicator_items = [
            entry for entry in self._indicator_items
            if not isinstance(entry[2], ClipPreviewChip)]
        self._indicator_items.sort(key=lambda entry: entry[0])
        self._indicator_lefts = [entry[0] for entry in self._indicator_items]
        self._active_indicator_items = {
            item for item in self._active_indicator_items
            if not isinstance(item, ClipPreviewChip)}
        self._live_segment_media_id = media.media_id

    def show_live_segments(self, media, cuts, removed, load_thumbnails=True):
        """Draw finalized analysis clips once, reusing their cached thumbnails."""
        if media is not self.media:
            return
        self.begin_live_segments(media)
        boundaries = [0, *(cut.frame for cut in sorted(
            cuts, key=lambda event: event.frame)), media.total_frames]
        removed = set(removed)
        visible = [
            (start, end, index)
            for index, (start, end) in enumerate(
                zip(boundaries, boundaries[1:]), 1)
            if index not in removed]
        if not visible:
            return
        # Live results can precede the project commit, so derive their ripple
        # layout from the supplied cuts while keeping the same canonical media
        # time mapping used by the committed timeline.
        display_starts = {}
        display_total = 0.0
        for start, end, index in visible:
            display_starts[index] = display_total
            display_total += max(
                0.0,
                media.seconds_for_frame(end) - media.seconds_for_frame(start),
            )
        frames = [(start + end) // 2 for start, end, _index in visible]
        thumbnails = self._clip_thumbnails(frames) if load_thumbnails else {}
        colors = self._segment_colors(len(boundaries) - 1)
        for start, end, index in visible:
            x = (display_starts[index] / max(0.001, display_total) *
                 self.content_width)
            chip = ClipPreviewChip(
                start, end, index, x, self, None,
                thumbnails.get((start + end) // 2), True,
                colors[index - 1], live_preview=True)
            self.scene().addItem(chip)
            self._indicator_items.append((x, x + 40, chip))
        self._indicator_items.sort(key=lambda entry: entry[0])
        self._indicator_lefts = [entry[0] for entry in self._indicator_items]
        if self.playhead:
            self._update_playhead_indicators(self.playhead.x())

    def clear_clip_duration(self):
        self._clip_duration_timer.stop()
        label = self._clip_duration_label
        self._clip_duration_label = None
        if label is None:
            return
        try:
            if label.scene() is self.scene():
                self.scene().removeItem(label)
        except RuntimeError:
            pass

    def show_clip_duration(self, start_frame, end_frame, index):
        """Show the exact current clip duration inside its timeline segment."""
        self.clear_clip_duration()
        if not self.media or self.media.mode == "faces":
            return
        start_frame = max(0, int(start_frame))
        end_frame = min(self.media.total_frames, int(end_frame))
        if end_frame <= start_frame:
            return
        start_seconds = self.media.seconds_for_frame(start_frame)
        end_seconds = self.media.seconds_for_frame(end_frame)
        duration = max(0.0, end_seconds - start_seconds)
        label = self.scene().addSimpleText(
            transport_timecode(duration, self.media.duration))
        label.setFont(QFont("JetBrains Mono", 8))
        label.setBrush(QColor(TEXT_MUTED))
        indicator_center = self.x_for_frame(start_frame) + 20
        label_width = label.boundingRect().width()
        label_x = indicator_center - label_width / 2
        label.setPos(label_x, 130)
        label.setToolTip(f"Clip {int(index):02d} exact duration")
        label.setZValue(24)
        self._clip_duration_label = label
        self._clip_duration_timer.start(5000)

    def _draw_ruler(self):
        # Labels stay on whole-second boundaries. Each labeled interval is
        # split into quarters, with a stronger midpoint like professional NLEs.
        candidates = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600,
                      900, 1200, 1800, 3600, 7200, 10800)
        duration = getattr(self, "_display_duration", self.media.duration)
        effective_pps = self.content_width / max(duration, 0.001)
        label_spacing = 72.0
        if timecode_has_hours(duration):
            sample_width = QFontMetrics(QFont()).horizontalAdvance(
                transport_timecode(duration, duration))
            # The first and last labels are clamped inside the ruler rather
            # than centered.  Extra room prevents their neighbours from
            # overlapping them at wide, multi-hour zoom levels.
            label_spacing = max(label_spacing, sample_width * 1.5 + 12.0)
        step = next(
            (value for value in candidates
             if value * effective_pps >= label_spacing),
            None,
        )
        if step is None:
            step = candidates[-1]
            while step * effective_pps < label_spacing:
                step *= 2
        moment = 0.0
        minor_step = step / 4
        minor_index = 0
        minor = 0.0
        while minor <= duration + .001:
            x = minor / max(duration, .001) * self.content_width
            phase = minor_index % 4
            if phase:
                midpoint = phase == 2
                tick = QGraphicsLineItem(
                    x, 25 if midpoint else 28, x, 31)
                tick.setPen(QPen(
                    QColor("#59616c" if midpoint else "#3d424a"), 1))
                self.scene().addItem(tick)
            minor_index += 1
            minor += minor_step
        while moment <= duration + .001:
            x = moment / max(duration, .001) * self.content_width
            tick = QGraphicsLineItem(x, 22, x, 31)
            tick.setPen(QPen(QColor("#7d8794"), 1.25))
            self.scene().addItem(tick)
            label = self.scene().addSimpleText(
                transport_timecode(moment, duration))
            label.setBrush(QColor(TEXT_MUTED))
            label_width = label.boundingRect().width()
            label_x = max(
                2.0,
                min(x - label_width / 2,
                    max(2.0, self.content_width - label_width - 2.0)),
            )
            label.setPos(label_x, 4)
            moment += step

    def x_for_frame(self, frame):
        if not self.media or not getattr(self, "_visible_layout", None):
            return 0.0
        frame = max(0, min(int(frame), self.media.total_frames))
        return self._x_for_source_seconds(self.media.seconds_for_frame(frame))

    def _x_for_source_seconds(self, source_seconds):
        if not self.media or not getattr(self, "_visible_layout", None):
            return 0.0
        source_seconds = max(
            0.0, min(self.media.duration, float(source_seconds)))
        display_seconds = 0.0
        for start, end, _index, display_start in self._visible_layout:
            start_seconds = self.media.seconds_for_frame(start)
            end_seconds = self.media.seconds_for_frame(end)
            display_seconds = display_start
            if source_seconds < start_seconds:
                break
            if source_seconds <= end_seconds:
                display_seconds = display_start + max(
                    0.0, source_seconds - start_seconds,
                )
                break
            display_seconds = display_start + max(
                0.0, end_seconds - start_seconds,
            )
        return (display_seconds / max(.001, self._display_duration) *
                self.content_width)

    def minimum_pixels_per_second(self):
        duration = getattr(self, "_display_duration", 0.0)
        if duration <= 0 and self.media:
            duration = self.media.duration
        return max(0.05, (self.viewport().width() - 20) /
                   max(0.001, duration))

    def x_for_seconds(self, seconds):
        if not self.media:
            return 0.0
        return self._x_for_source_seconds(seconds)

    def seconds_for_x(self, x):
        if not self.media or not getattr(self, "_visible_layout", None):
            return 0.0
        ratio = max(0.0, min(1.0, float(x) / max(1.0, self.content_width)))
        display_seconds = ratio * self._display_duration
        for start, end, _index, display_start in self._visible_layout:
            start_seconds = self.media.seconds_for_frame(start)
            end_seconds = self.media.seconds_for_frame(end)
            length = max(0.0, end_seconds - start_seconds)
            if display_seconds < display_start + length - 1e-9:
                return min(
                    end_seconds,
                    start_seconds + display_seconds - display_start,
                )
        return self.media.seconds_for_frame(self._visible_layout[-1][1])

    def frame_for_x(self, x):
        return self.media.frame_for_seconds(self.seconds_for_x(x))

    def bounds_for(self, event):
        cuts = sorted(self.media.cuts, key=lambda cut: (cut.frame, cut.event_id))
        index = next((position for position, cut in enumerate(cuts)
                      if cut.event_id == event.event_id), None)
        if index is None:
            return 1, max(1, self.media.total_frames - 1)
        previous = cuts[index - 1].frame if index > 0 else 0
        following = (cuts[index + 1].frame if index + 1 < len(cuts)
                     else self.media.total_frames)
        # Keep at least one real frame on both sides.  In particular, the
        # first generated sequence can never collapse into frame zero.
        lower = previous + 1
        upper = following - 1
        return lower, upper

    def handle_moved(self, handle):
        if not handle.event.original_frame:
            handle.event.original_frame = handle.event.frame
        lower, upper = (handle._drag_bounds or
                        self.bounds_for(handle.event))
        handle.event.frame = max(
            lower, min(upper, self.frame_for_x(handle.x())))
        handle.setToolTip(
            f"Cut · {timecode(self.media.seconds_for_frame(handle.event.frame), self.media.duration)}")

    def finish_handle_move(self, selected_event=None):
        if not self.media:
            return
        self.media.normalize_cuts()
        self.rebuild()
        self.contentChanged.emit()
        if selected_event is not None:
            selected_handle = next((
                item for item in self.scene().items()
                if isinstance(item, CutHandle) and
                item.event.event_id == selected_event.event_id), None)
            if selected_handle is not None:
                selected_handle.setSelected(True)
                self.cutSelected.emit(selected_event)

    def set_position(self, seconds):
        self.current_seconds = max(
            0.0, min(float(seconds), self.media.duration if self.media else 0.0))
        if self.playhead and self.media and not self.playhead.dragging:
            self.playhead.setX(self.x_for_seconds(self.current_seconds))
            self._update_playhead_indicators(self.playhead.x())

    def _update_playhead_indicators(self, playhead_x):
        if not self._indicator_items:
            return
        playhead_x = float(playhead_x)
        active = set()
        index = bisect_right(self._indicator_lefts, playhead_x)
        for position in range(index - 1, -1, -1):
            left, right, item = self._indicator_items[position]
            if left < playhead_x - 44:
                break
            if left <= playhead_x <= right:
                active.add(item)
        for item in self._active_indicator_items - active:
            item.set_playhead_active(False)
        for item in active - self._active_indicator_items:
            item.set_playhead_active(True)
        self._active_indicator_items = active

    def _selection_changed(self):
        selected = self.scene().selectedItems()
        item = selected[0] if selected else None
        # Cut handles emit directly on mouse press so selection does not wait
        # for Qt to decide whether the gesture was a click or a drag.
        if isinstance(item, CutHandle):
            return
        self.cutSelected.emit(item.event if isinstance(item, FaceChip) else None)

    def mouseMoveEvent(self, event):
        # Let movable scene items update first.  While dragging the playhead,
        # its clamped position is the canonical hover position (the mouse may
        # have grabbed the wide hit target a few pixels away from the needle).
        super().mouseMoveEvent(event)
        if self.media and getattr(self, "_display_duration", 0) > 0:
            scene_position = self.mapToScene(event.position().toPoint())
            dragging_playhead = bool(self.playhead and self.playhead.dragging)
            x = (float(self.playhead.x()) if dragging_playhead
                 else scene_position.x())
            if 0 <= x <= self.content_width:
                if dragging_playhead:
                    if self._hover_guide is not None:
                        self._hover_guide.hide()
                else:
                    if self._hover_guide is None:
                        self._hover_guide = QGraphicsLineItem()
                        self._hover_guide.setPen(QPen(
                            QColor(156, 163, 175, 105), 1,
                            Qt.PenStyle.DashLine))
                        self._hover_guide.setZValue(35)
                        self._hover_guide.setAcceptedMouseButtons(
                            Qt.MouseButton.NoButton)
                        self.scene().addItem(self._hover_guide)
                    self._hover_guide.setLine(
                        x, 0, x, self.scene().sceneRect().height())
                    self._hover_guide.show()

                duration = self._display_duration
                seconds = max(
                    0.0, min(duration, x / max(1.0, self.content_width) * duration))
                text = (player_timecode(seconds, duration)
                        if self.show_milliseconds else
                        transport_timecode(seconds, duration))
                if self.show_milliseconds:
                    main_text, millis_text = text[:-4], text[-4:]
                    self._hover_time_bubble.setText(
                        f"<span style='color:#e5e7eb'>{main_text}</span>"
                        f"<span style='color:#7d8794'>{millis_text}</span>")
                else:
                    self._hover_time_bubble.setText(text)
                text_width = self._hover_time_bubble.fontMetrics().horizontalAdvance(text)
                self._hover_time_bubble.setFixedWidth(text_width + 18)
                anchor_position = event.position().toPoint()
                if dragging_playhead:
                    anchor_position.setX(round(
                        self.mapFromScene(QPointF(x, scene_position.y())).x()))
                anchor = self.viewport().mapToGlobal(anchor_position)
                self._hover_time_bubble.move(
                    anchor.x() - self._hover_time_bubble.width() // 2,
                    anchor.y() - self._hover_time_bubble.height() - 14)
                self._hover_time_bubble.show()
            else:
                self._hide_hover_time()
        else:
            self._hide_hover_time()

    def _hide_hover_time(self):
        self._hover_time_bubble.hide()
        if self._hover_guide is not None:
            self._hover_guide.hide()

    def leaveEvent(self, event):
        self._hide_hover_time()
        super().leaveEvent(event)

    def hideEvent(self, event):
        self._hide_hover_time()
        super().hideEvent(event)

    def _timeline_item_at(self, position):
        """Return the real timeline item below visual-only hover overlays."""
        return next((
            item for item in self.items(position)
            if item is not self._hover_guide
        ), None)

    def mousePressEvent(self, event):
        item = self._timeline_item_at(event.position().toPoint())
        pressed_cut = (item.event if
                       event.button() == Qt.MouseButton.LeftButton and
                       isinstance(item, CutHandle) else None)
        if event.button() == Qt.MouseButton.LeftButton and not isinstance(
                item, (CutHandle, PlayheadItem, FaceChip, ClipPreviewChip)) and self.media:
            x = self.mapToScene(event.position().toPoint()).x()
            self.scrubFinished.emit(self.seconds_for_x(x))
        super().mousePressEvent(event)
        if pressed_cut is not None:
            self.cutSelected.emit(pressed_cut)
        if (event.button() == Qt.MouseButton.LeftButton and
                isinstance(item, SceneSegmentItem) and self.media and
                (not self.media.analyzed or self.media.mode == "faces")):
            self.segmentSelected.emit(
                self.media.seconds_for_frame(item.start_frame),
                self.media.seconds_for_frame(item.end_frame),
                item.index)

    def contextMenuEvent(self, event):
        item = self._timeline_item_at(event.pos())
        if isinstance(item, CutHandle):
            local_point = item.mapFromScene(self.mapToScene(event.pos()))
            if QRectF(-7, -5, 14, 18).contains(local_point):
                menu = QMenu(self)
                menu.setObjectName("mediaContextMenu")
                remove_item = CompactMenuItem(
                    themed_icon("fa6s.trash", scale_factor=.85),
                    "Remove marker", menu)
                remove_action = QWidgetAction(menu)
                remove_action.setDefaultWidget(remove_item)
                menu.addAction(remove_action)
                selected = {"remove": False}

                def select_marker_remove():
                    selected["remove"] = True
                    menu.close()

                remove_item.activated.connect(select_marker_remove)
                menu.exec(event.globalPos())
                if selected["remove"]:
                    self.cutRemoveRequested.emit(item.event)
                event.accept()
                return
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
        if self.media and not isinstance(item, (CutHandle, FaceChip)):
            # The context menu may be opened elsewhere, but the blue playhead is
            # always the precise source of truth for a newly created cut.
            seconds = self.current_seconds
            frame = self.media.frame_for_seconds(seconds)
            scene_x = self.mapToScene(event.pos()).x()
            pointed_frame = self.frame_for_x(scene_x)
            boundaries = [0, *(cut.frame for cut in sorted(
                self.media.cuts, key=lambda cut: cut.frame)),
                self.media.total_frames]
            pointed_index = next((
                index for index, (start, end) in enumerate(
                    zip(boundaries, boundaries[1:]), 1)
                if start <= pointed_frame < end and
                index not in self.media.removed_segments), None)
            menu = QMenu(self)
            menu.setObjectName("mediaContextMenu")
            selected = {"cut": False, "remove": False}
            if 0 < frame < self.media.total_frames:
                seconds = self.media.seconds_for_frame(frame)
                cut_item = CompactMenuItem(
                    themed_icon("fa6s.scissors", scale_factor=.85),
                    "Add new cut", menu, hover_color=ACCENT)
                cut_action = QWidgetAction(menu)
                cut_action.setDefaultWidget(cut_item)
                menu.addAction(cut_action)
                def select_cut():
                    selected["cut"] = True
                    menu.close()

                cut_item.activated.connect(select_cut)
            if pointed_index is not None:
                remove_item = CompactMenuItem(
                    themed_icon("fa6s.trash", scale_factor=.85),
                    "Remove", menu)
                remove_action = QWidgetAction(menu)
                remove_action.setDefaultWidget(remove_item)
                menu.addAction(remove_action)

                def select_remove():
                    selected["remove"] = True
                    menu.close()

                remove_item.activated.connect(select_remove)
            if not menu.actions():
                return
            menu.exec(event.globalPos())
            if selected["cut"]:
                self.cutAddRequested.emit(seconds)
            elif selected["remove"] and pointed_index is not None:
                self.segmentRemoveRequested.emit(pointed_index)
            event.accept()
            return
        super().contextMenuEvent(event)

    def wheelEvent(self, event):
        if self.media and event.modifiers() & primary_modifier():
            self.zoom_by_factor(
                1.18 if event.angleDelta().y() > 0 else .85,
                event.position().x())
            event.accept()
            return
        super().wheelEvent(event)

    def zoom_by_factor(self, factor, viewport_x=None):
        if not self.media:
            return
        viewport_x = (self.viewport().width() / 2 if viewport_x is None
                      else float(viewport_x))
        scene_x = self.mapToScene(QPoint(round(viewport_x), 0)).x()
        anchor_seconds = self.seconds_for_x(scene_x)
        previous = self.pixels_per_second
        self.pixels_per_second = max(
            self.minimum_pixels_per_second(),
            min(500.0, previous * float(factor)))
        if abs(self.pixels_per_second - previous) < 1e-9:
            return
        self.rebuild()
        anchor_x = self.x_for_seconds(anchor_seconds)
        self.horizontalScrollBar().setValue(round(anchor_x - viewport_x))


class ScenespyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("Scenespy", "Scenespy")
        self.show_milliseconds = self.settings.value(
            "view/show_milliseconds", True, type=bool)
        self.project = ScenespyProject()
        self.current_media: MediaItem | None = None
        self.selected_cut = None
        self.selected_segment = None
        self.task = None
        self.batch_mode = False
        self.batch_active = False
        self.processing_media = None
        self._batch_analyzed_media = set()
        self._batch_completed_media = set()
        self._batch_failed_media = set()
        self._manual_face_draft = None
        self.preview_end_ms = None
        self.preview_start_ms = None
        self.preview_armed = False
        self._pending_interactive_seek_ms = None
        self._interactive_seek_inflight_ms = None
        self._frame_step_target = None
        self._frame_step_media_id = None
        self._playback_proxy_tasks = {}
        self._pending_proxy_path = None
        self._pending_proxy_restore = None
        self._edit_preview_active = False
        self._edit_preview_frame = None
        self._edit_preview_media_id = None
        self._play_after_edit_sync = False
        self.available_encoder_accel = detect_available_encoder_accel()
        self.available_inference_accel = detect_available_inference_accel()
        self.available_accel = (self.available_encoder_accel |
                                self.available_inference_accel)
        self.setWindowTitle(f"Scenespy {__version__}")
        icon = Path(__file__).parent / "assets" / "images" / "x.png"
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))
        self.resize(1440, 900)
        self.setMinimumSize(1080, 700)
        self._build_player()
        self._interactive_seek_timer = QTimer(self)
        self._interactive_seek_timer.setSingleShot(True)
        self._interactive_seek_timer.setInterval(120)
        self._interactive_seek_timer.timeout.connect(
            self._interactive_seek_timeout)
        self._build_ui()
        self._frame_decoder = None
        self._ensure_frame_decoder()
        self._build_actions()
        self._set_milliseconds_visible(self.show_milliseconds)
        QApplication.instance().installEventFilter(self)
        self._apply_theme()
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()
        self._update_transport_access()

    def eventFilter(self, watched, event):
        """Provide Premiere-style unmodified arrow-key frame stepping."""
        if hasattr(self, "github_button") and watched is self.github_button:
            if event.type() == QEvent.Type.Enter:
                button_position = self.github_button.mapTo(
                    self.menuBar(), QPoint(0, 0))
                label_size = self.github_label.sizeHint()
                self.github_label.setGeometry(
                    button_position.x() - label_size.width() - 5,
                    button_position.y() +
                    (self.github_button.height() - label_size.height()) // 2,
                    label_size.width(), label_size.height())
                self.github_label.show()
                self.github_label.raise_()
            elif event.type() == QEvent.Type.Leave:
                self.github_label.hide()
        if (hasattr(self, "volume_menu") and
                self.volume_menu.isVisible() and
                event.type() == QEvent.Type.MouseButtonPress and
                event.button() == Qt.MouseButton.LeftButton):
            global_position = event.globalPosition().toPoint()
            button_position = self.volume_button.mapFromGlobal(global_position)
            if self.volume_button.rect().contains(button_position):
                self.volume_menu.hide()
                self.toggle_mute()
                event.accept()
                return True
        if (hasattr(self, "volume_button") and
                watched is self.volume_button and
                event.type() == QEvent.Type.Enter):
            self._volume_hover_timer.start()
        elif (hasattr(self, "volume_button") and
              watched is self.volume_button and
              event.type() == QEvent.Type.Leave):
            self._volume_hover_timer.stop()
        elif (hasattr(self, "volume_button") and
              watched is self.volume_button and
              event.type() == QEvent.Type.MouseButtonPress):
            self._volume_hover_timer.stop()
        if (event.type() == QEvent.Type.KeyPress and
                event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right) and
                event.modifiers() == Qt.KeyboardModifier.NoModifier and
                self.current_media and
                not self._current_media_is_processing()):
            focus = QApplication.focusWidget()
            if not isinstance(focus, (QLineEdit, QSpinBox, QComboBox, QMenu)):
                self.step_video_frame(
                    -1 if event.key() == Qt.Key.Key_Left else 1)
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _show_volume_menu(self):
        if self.volume_menu.isVisible():
            return
        self.volume_menu.ensurePolished()
        menu_size = self.volume_menu.sizeHint()
        position = self.volume_button.mapToGlobal(QPoint(
            (self.volume_button.width() - menu_size.width()) // 2,
            -menu_size.height() - 4,
        ))
        self.volume_menu.popup(position)

    def _set_volume(self, value):
        volume = max(0, min(100, int(value))) / 100
        self.audio.setVolume(volume)
        if volume > 0 and self.audio.isMuted():
            self.audio.setMuted(False)

    def _sync_volume_slider(self, volume):
        value = round(max(0.0, min(1.0, float(volume))) * 100)
        if self.volume_slider.value() != value:
            self.volume_slider.setValue(value)

    def _build_player(self):
        self.audio = QAudioOutput(self)
        self.audio.setVolume(.5)
        self.audio.setMuted(False)
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
        media_footer.setContentsMargins(1, 1, 1, 1)
        media_footer.setSpacing(6)
        self.import_button = QPushButton("Input")
        self.import_button.setObjectName("mediaFooterButton")
        self.import_button.setFixedHeight(34)
        self.import_button.clicked.connect(self.import_videos)
        self.output_button = QPushButton("Output")
        self.output_button.setObjectName("mediaFooterButton")
        self.output_button.setFixedHeight(34)
        self.output_button.setToolTip("Choose where the processed folders will be saved.")
        self.output_button.clicked.connect(self.choose_batch_output)
        media_footer.addWidget(self.import_button, 1)
        media_footer.addWidget(self.output_button, 1)
        layout.addLayout(media_footer)
        root.addWidget(left)

        center = QSplitter(Qt.Orientation.Vertical)
        center.setChildrenCollapsible(False)
        self.workspace_splitter = center
        viewer = QFrame(objectName="viewer")
        viewer.setMinimumHeight(350)
        self.viewer_panel = viewer
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
            button.setFixedSize(42, 30)
            button.setIconSize(QSize(14, 14))
        self.play_button.setFixedSize(44, 34)
        self.play_button.setIconSize(QSize(15, 15))
        self.volume_button.setIconSize(QSize(15, 15))
        self.volume_menu = QMenu(self)
        self.volume_menu.setObjectName("volumeMenu")
        self._volume_hover_timer = QTimer(self)
        self._volume_hover_timer.setSingleShot(True)
        self._volume_hover_timer.setInterval(600)
        self._volume_hover_timer.timeout.connect(self._show_volume_menu)
        self.volume_slider = QSlider(Qt.Orientation.Vertical)
        self.volume_slider.setObjectName("volumeSlider")
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setPageStep(10)
        self.volume_slider.setValue(round(self.audio.volume() * 100))
        self.volume_slider.setFixedSize(32, 130)
        self.volume_slider.setToolTip("Volume")
        self.volume_slider.valueChanged.connect(self._set_volume)
        self.audio.volumeChanged.connect(self._sync_volume_slider)
        volume_action = QWidgetAction(self.volume_menu)
        volume_action.setDefaultWidget(self.volume_slider)
        self.volume_menu.addAction(volume_action)
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
        self.seek.sliderMoved.connect(self.seek_timeline_ms)
        self.clock = MillisecondTimeEdit("00:00.000")
        self.clock.setObjectName("editableTimecode")
        self.clock.setFixedWidth(92)
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clock.editingFinished.connect(self.seek_from_timecode)
        self.clock.timeStepped.connect(self._step_from_timecode)
        self.time_separator = QLabel("/")
        self.time_separator.setObjectName("timeSeparator")
        self.time_separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.duration_clock = MillisecondTimeEdit("00:00.000")
        self.duration_clock.setObjectName("timecode")
        self.duration_clock.setFixedWidth(92)
        self.duration_clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.duration_clock.setReadOnly(True)
        self.duration_clock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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
        self.timeline.show_milliseconds = self.show_milliseconds
        self.timeline.seekRequested.connect(self._request_timeline_seek)
        self.timeline.scrubStarted.connect(self._begin_timeline_scrub)
        self.timeline.scrubFinished.connect(self._finish_timeline_scrub)
        self.timeline.editStarted.connect(self._begin_timeline_edit)
        self.timeline.cutSelected.connect(self.show_cut)
        self.timeline.segmentActivated.connect(self.show_segment)
        self.timeline.segmentSelected.connect(self.show_segment)
        self.timeline.contentChanged.connect(self._timeline_content_changed)
        self.timeline.faceRemoveRequested.connect(self.remove_face_permanently)
        self.timeline.segmentRemoveRequested.connect(self.remove_segment_permanently)
        self.timeline.cutAddRequested.connect(self.add_cut_at)
        self.timeline.cutRemoveRequested.connect(self.remove_cut_marker)
        center.addWidget(self.timeline)
        center.setSizes([650, 210])
        QTimer.singleShot(0, self._restore_workspace_splitter)
        root.addWidget(center)

        right = QFrame(objectName="analysisPanel")
        # The scroll area owns the panel's width constraint.  Giving the inner
        # widget a second minimum makes a vertical scrollbar increase the
        # splitter minimum in short windows.
        right.setMinimumWidth(0)
        layout = QVBoxLayout(right)
        layout.setContentsMargins(13, 12, 13, 12)
        layout.setSpacing(7)
        analysis_title = QLabel("ANALYSIS", objectName="analysisTitle")
        analysis_title.setIndent(0)
        layout.addWidget(analysis_title)
        self.analysis_field_labels = {}
        for title, widget in self._analysis_fields():
            label = QLabel(title, objectName="muted")
            self.analysis_field_labels[title] = label
            layout.addWidget(label)
            layout.addWidget(widget)
        layout.addSpacing(7)
        selection_section_title = QLabel("SELECTION", objectName="analysisTitle")
        selection_section_title.setIndent(0)
        layout.addWidget(selection_section_title)
        self.selection_card = QFrame(objectName="selectionCard")
        selection_card_layout = QVBoxLayout(self.selection_card)
        selection_card_layout.setContentsMargins(9, 9, 9, 9)
        selection_card_layout.setSpacing(7)
        self.selection_title = QLabel("No event selected")
        selection_header = QHBoxLayout()
        selection_header.setContentsMargins(0, 0, 0, 0)
        selection_header.setSpacing(6)
        selection_header.addWidget(self.selection_title, 1)
        self.selection_close_button = QToolButton()
        self.selection_close_button.setObjectName("selectionClose")
        self.selection_close_button.setIcon(themed_icon("fa6s.xmark"))
        self.selection_close_button.setIconSize(QSize(10, 10))
        self.selection_close_button.setFixedSize(16, 16)
        self.selection_close_button.setToolTip("Close selection")
        self.selection_close_button.clicked.connect(self.clear_selection)
        self.selection_close_button.hide()
        selection_header.addWidget(self.selection_close_button)
        self.selection_detail = QLabel("Select a marker to inspect it.", objectName="muted")
        self.selection_detail.setWordWrap(True)
        self.selection_preview = ProcessingPreviewLabel()
        self.selection_preview.setObjectName("selectionPreview")
        self.selection_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selection_preview.setFixedHeight(150)
        self.selection_preview.hide()
        selection_actions = QHBoxLayout()
        selection_actions.setContentsMargins(1, 1, 1, 1)
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
        self.face_save_button = QToolButton()
        self.face_save_button.setIcon(themed_icon("fa6s.floppy-disk"))
        self.face_save_button.setToolTip("Save selected face as")
        self.face_save_button.clicked.connect(self.save_selected_face)
        self.face_save_button.hide()
        self.face_detected_filter_button = QToolButton()
        self.face_detected_filter_button.setCheckable(True)
        self.face_detected_filter_button.setChecked(True)
        self.face_detected_filter_button.setIcon(themed_icon("fa6s.eye"))
        self.face_detected_filter_button.setToolTip(
            "Hide detected faces that are not included")
        self.face_detected_filter_button.toggled.connect(
            lambda visible: self._toggle_face_indicators("detected", visible))
        self.face_saved_filter_button = QToolButton()
        self.face_saved_filter_button.setCheckable(True)
        self.face_saved_filter_button.setChecked(True)
        self.face_saved_filter_button.setIcon(themed_icon("fa6s.eye"))
        self.face_saved_filter_button.setToolTip("Hide saved faces")
        self.face_saved_filter_button.toggled.connect(
            lambda visible: self._toggle_face_indicators("saved", visible))
        self.remove_button = QToolButton()
        self.remove_button.setIcon(themed_icon("fa6s.trash"))
        self.remove_button.setToolTip("Remove selected event")
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self.remove_cut)
        self.selection_result_actions = QWidget()
        self.selection_result_actions.setObjectName("selectionActions")
        self.selection_result_actions.setLayout(selection_actions)
        for button in (self.face_previous_button, self.face_next_button,
                       self.face_reset_button, self.face_add_button,
                       self.face_cancel_button, self.face_save_button,
                       self.remove_button):
            button.setObjectName("selectionAction")
            button.setMinimumSize(26, 30)
            button.setMaximumSize(34, 30)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setIconSize(QSize(14, 14))
            selection_actions.addWidget(button, 1)
        selection_actions.addStretch(1)
        self.preanalysis_actions = QWidget()
        self.preanalysis_actions.setObjectName("selectionActions")
        preanalysis_layout = QHBoxLayout(self.preanalysis_actions)
        preanalysis_layout.setContentsMargins(1, 1, 1, 1)
        preanalysis_layout.setSpacing(6)
        self.preanalysis_cut_button = QToolButton()
        self.preanalysis_cut_button.setObjectName("selectionAction")
        self.preanalysis_cut_button.setIcon(themed_icon("fa6s.scissors"))
        self.preanalysis_cut_button.setToolTip("Add cut at playhead (Ctrl+K)")
        self.preanalysis_cut_button.clicked.connect(self.add_cut)
        self.preanalysis_disable_button = QToolButton()
        self.preanalysis_disable_button.setObjectName("selectionAction")
        self.preanalysis_disable_button.setIcon(themed_icon("fa6s.trash"))
        self.preanalysis_disable_button.setToolTip("Disable selected clip")
        self.preanalysis_disable_button.clicked.connect(
            self._toggle_preanalysis_segment)
        for button in (self.preanalysis_cut_button,
                       self.preanalysis_disable_button):
            button.setMinimumSize(26, 30)
            button.setMaximumSize(34, 30)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setIconSize(QSize(14, 14))
            preanalysis_layout.addWidget(button, 1)
        preanalysis_layout.addStretch(1)
        selection_card_layout.addLayout(selection_header)
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
        self.progress_animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.processing_position_animation = QVariantAnimation(self)
        self.processing_position_animation.setEasingCurve(
            QEasingCurve.Type.InOutCubic)
        self.processing_position_animation.valueChanged.connect(
            lambda value: self._sync_processing_position(float(value)))
        self.queue_metrics = QLabel(objectName="muted")
        self.queue_metrics.hide()
        self.progress_metrics = QLabel(objectName="muted")
        self.face_metrics = QWidget()
        self.face_metrics.setObjectName("faceMetrics")
        face_metrics_layout = QVBoxLayout(self.face_metrics)
        face_metrics_layout.setContentsMargins(0, 0, 0, 0)
        face_metrics_layout.setSpacing(3)
        self.face_detected_metric = QLabel(objectName="muted")
        self.face_saved_metric = QLabel(objectName="muted")
        for label, button in (
                (self.face_detected_metric, self.face_detected_filter_button),
                (self.face_saved_metric, self.face_saved_filter_button)):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(label.fontMetrics().horizontalAdvance(" "))
            row.addWidget(label)
            button.setObjectName("selectionAction")
            button.setFixedSize(20, 20)
            button.setIconSize(QSize(11, 11))
            row.addWidget(button)
            row.addStretch(1)
            face_metrics_layout.addLayout(row)
        self.face_metrics.hide()
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
        layout.addWidget(self.queue_metrics)
        layout.addWidget(self.progress_metrics)
        layout.addWidget(self.face_metrics)
        layout.addLayout(progress_status_row)
        layout.addWidget(self.progress)
        layout.addWidget(self.action_button)
        analysis_scroll = QScrollArea()
        analysis_scroll.setObjectName("analysisScroll")
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setFrameShape(QFrame.Shape.NoFrame)
        analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        analysis_scroll.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        analysis_scroll.viewport().setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground)
        analysis_scroll.viewport().setAutoFillBackground(False)
        analysis_scroll.setWidget(right)
        analysis_scroll.setMinimumWidth(220)
        root.addWidget(analysis_scroll)
        root.setSizes([225, 950, 275])
        root.splitterMoved.connect(lambda _position, _index: self._position_media_info())
        self.setStatusBar(QStatusBar())
        self._set_progress_metrics(stage=None)
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

    def _restore_workspace_splitter(self):
        saved_sizes = self.settings.value(
            "layout/workspace_splitter_sizes", [], type=list)
        try:
            sizes = [int(value) for value in saved_sizes]
        except (TypeError, ValueError):
            return
        if len(sizes) == 2 and all(value > 0 for value in sizes):
            self.workspace_splitter.setSizes(sizes)

    def _save_workspace_splitter(self):
        sizes = self.workspace_splitter.sizes()
        if len(sizes) == 2 and all(value > 0 for value in sizes):
            self.settings.setValue(
                "layout/workspace_splitter_sizes", sizes)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._position_media_info)

    def _analysis_fields(self):
        self.mode = FramedComboBox()
        self.mode.setObjectName("analysisField")
        self.mode.setProperty("protectLeftBorder", True)
        self.mode.addItem("Scene Detection", "scene")
        self.mode.addItem("Every Seconds", "interval")
        self.mode.addItem("Detect Faces", "faces")
        for index in range(self.mode.count()):
            description = MODE_TOOLTIPS.get(self.mode.itemData(index), "")
            self.mode.setItemData(
                index, description, Qt.ItemDataRole.ToolTipRole)
        self.sensitivity = FramedComboBox()
        self.sensitivity.setObjectName("analysisField")
        self.sensitivity.setProperty("protectLeftBorder", True)
        self.sensitivity.addItems(PROFILES.keys())
        self.sensitivity.setCurrentText("Normal")
        self.interval = IntervalSpinBox()
        self.interval.setObjectName("analysisField")
        self.interval.setRange(1, 18000)
        self.interval.setValue(10)
        self.interval.setKeyboardTracking(False)
        self.accel = FramedComboBox()
        self.accel.setObjectName("analysisField")
        self.accel.setProperty("protectLeftBorder", True)
        for value in ACCEL_OPTIONS:
            if value in self.available_accel: self.accel.addItem(value.upper(), value)
        return (("Mode", self.mode), ("Sensitivity", self.sensitivity),
                ("Interval", self.interval), ("Acceleration", self.accel))

    def _build_actions(self):
        self.import_action = QAction("Import Video", self)
        self.import_action.setShortcut(QKeySequence.StandardKey.Open)
        self.import_action.triggered.connect(self.import_videos)
        save_action = QAction("Save Project", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_project)
        self.load_action = QAction("Open Project", self)
        self.load_action.setShortcut(primary_shortcut(Qt.Key.Key_O, shift=True))
        self.load_action.triggered.connect(self.open_project)
        self.add_cut_action = QAction("Add Cut at Playhead", self)
        self.add_cut_action.setShortcut(primary_shortcut(Qt.Key.Key_K))
        self.add_cut_action.triggered.connect(self.add_cut)
        previous_frame_action = QAction("Previous Frame", self)
        previous_frame_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        previous_frame_action.setShortcutContext(
            Qt.ShortcutContext.WidgetShortcut)
        previous_frame_action.triggered.connect(lambda: self.step_video_frame(-1))
        next_frame_action = QAction("Next Frame", self)
        next_frame_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        next_frame_action.setShortcutContext(
            Qt.ShortcutContext.WidgetShortcut)
        next_frame_action.triggered.connect(lambda: self.step_video_frame(1))
        playback_action = QAction("Play/Pause", self)
        playback_action.setShortcut(QKeySequence(Qt.Key.Key_Space))
        playback_action.triggered.connect(self.toggle_playback)
        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.setShortcut(primary_shortcut(Qt.Key.Key_R, shift=True))
        reset_layout_action.triggered.connect(self.reset_layout)
        zoom_modifier = "⌘" if sys.platform == "darwin" else "Ctrl"
        zoom_in_action = QAction(
            f"Zoom In\t{zoom_modifier}+Scroll Up", self)
        zoom_in_action.triggered.connect(
            lambda: self.timeline.zoom_by_factor(1.18))
        zoom_out_action = QAction(
            f"Zoom Out\t{zoom_modifier}+Scroll Down", self)
        zoom_out_action.triggered.connect(
            lambda: self.timeline.zoom_by_factor(.85))
        file_menu = ShortcutMenu("&File", self)
        self.menuBar().addMenu(file_menu)
        file_menu.addActions([self.import_action, self.load_action, save_action])
        edit_menu = ShortcutMenu("&Edit", self)
        self.menuBar().addMenu(edit_menu)
        edit_menu.addAction(playback_action)
        edit_menu.addActions([previous_frame_action, next_frame_action])
        edit_menu.addSeparator()
        edit_menu.addAction(self.add_cut_action)
        view_menu = ShortcutMenu("&View", self)
        self.menuBar().addMenu(view_menu)
        view_menu.addAction(reset_layout_action)
        view_menu.addActions([zoom_in_action, zoom_out_action])
        self.milliseconds_menu_item = CompactMenuItem(
            themed_icon(
                "fa6s.toggle-on" if self.show_milliseconds else "fa6s.toggle-off",
                ACCENT, scale_factor=.85),
            "Show Milliseconds", view_menu, hover_color=ACCENT)
        milliseconds_action = QWidgetAction(view_menu)
        milliseconds_action.setDefaultWidget(self.milliseconds_menu_item)
        view_menu.addAction(milliseconds_action)

        def toggle_milliseconds():
            self._set_milliseconds_visible(not self.show_milliseconds)
            view_menu.close()

        self.milliseconds_menu_item.activated.connect(toggle_milliseconds)
        self.github_label = QLabel("guilhermejaques", self.menuBar())
        self.github_label.setObjectName("githubUsername")
        self.github_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.github_label.hide()
        self.github_button = QToolButton(self.menuBar())
        self.github_button.setObjectName("githubLink")
        self.github_button.setIcon(qta.icon(
            "fa6b.github", color="#aeb4bd", color_active="#ffffff"))
        self.github_button.setIconSize(QSize(14, 14))
        self.github_button.setFixedSize(22, 22)
        self.github_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.github_button.setToolTip("Scenespy on GitHub")
        self.github_button.installEventFilter(self)
        self.github_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl("https://github.com/guilhermejaques/scenespy")))
        self.menuBar().setCornerWidget(
            self.github_button, Qt.Corner.TopRightCorner)
        self.addAction(playback_action)

    def _set_milliseconds_visible(self, visible):
        self.show_milliseconds = bool(visible)
        self.settings.setValue(
            "view/show_milliseconds", self.show_milliseconds)
        self.timeline.show_milliseconds = self.show_milliseconds
        self.timeline._hover_time_bubble.hide()
        if hasattr(self, "milliseconds_menu_item"):
            self.milliseconds_menu_item.setIcon(themed_icon(
                "fa6s.toggle-on" if self.show_milliseconds else "fa6s.toggle-off",
                ACCENT, scale_factor=.85))

        duration = self._timeline_duration()
        minimum_width = 92 if self.show_milliseconds else 72
        duration_text = self._player_timecode(duration, duration)
        width = max(
            minimum_width,
            self.clock.fontMetrics().horizontalAdvance(duration_text) + 24)
        self.clock.setFixedWidth(width)
        self.duration_clock.setFixedWidth(width)
        if self.current_media:
            if self._processing_preview_selected():
                self._sync_processing_position(self.timeline.current_seconds)
            else:
                self._refresh_transport_timeline()
        else:
            zero = self._player_timecode(0, 0)
            self.clock.setText(zero)
            self.duration_clock.setText(zero)

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
        self._save_workspace_splitter()
        self.timeline.fit_to_view()
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
        QFrame#selectionCard {{
            background:#292b2f;
            border-top:1px solid #151619;
            border-left:1px solid #151619;
            border-bottom:1px solid #4b4e54;
            border-right:1px solid #4b4e54;
            border-radius:7px;
        }}
        QWidget#selectionActions {{ background:transparent; border:0; }}
        QWidget#faceMetrics {{ background:transparent; border:0; }}
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
        QListWidget#mediaList {{
            border-top:1px solid #151619;
            border-left:1px solid #151619;
            border-bottom:1px solid #4b4e54;
            border-right:1px solid #4b4e54;
        }}
        QListWidget#mediaList::item:hover {{ background:#353941; color:white; border-radius:0; }}
        QListWidget#mediaList::item:selected {{ background:#4a4e55; color:white; border-radius:0; }}
        QListWidget#mediaList QScrollBar:vertical, QListWidget#mediaList QWidget#qt_scrollarea_vcontainer {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #323438, stop:1 #292b2f); border:0; }}
        QListWidget#mediaList QScrollBar:horizontal, QListWidget#mediaList QWidget#qt_scrollarea_hcontainer {{ background:#292b2f; border:0; }}
        QListWidget#mediaList QScrollBar::add-page:horizontal, QListWidget#mediaList QScrollBar::sub-page:horizontal, QListWidget#mediaList QScrollBar::add-page:vertical, QListWidget#mediaList QScrollBar::sub-page:vertical {{ background:transparent; border:0; }}
        QListWidget#mediaList QScrollBar::add-line:horizontal, QListWidget#mediaList QScrollBar::sub-line:horizontal {{ background:transparent; border:0; width:0; }}
        QListWidget#mediaList QScrollBar::add-line:vertical, QListWidget#mediaList QScrollBar::sub-line:vertical {{ background:transparent; border:0; height:0; }}
        QListWidget#mediaList::corner {{ background:#292b2f; border:0; }}
        QScrollArea#analysisScroll,
        QScrollArea#analysisScroll QWidget#qt_scrollarea_viewport {{ background:transparent; border:0; }}
        QScrollArea#analysisScroll QScrollBar:vertical,
        QScrollArea#analysisScroll QWidget#qt_scrollarea_vcontainer {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #323438, stop:1 #292b2f); border:0; }}
        QScrollArea#analysisScroll QScrollBar::add-page:vertical,
        QScrollArea#analysisScroll QScrollBar::sub-page:vertical {{ background:transparent; border:0; }}
        QScrollArea#analysisScroll QScrollBar::add-line:vertical,
        QScrollArea#analysisScroll QScrollBar::sub-line:vertical {{ background:transparent; border:0; height:0; }}
        QScrollArea#analysisScroll::corner {{ background:#292b2f; border:0; }}
        QComboBox, QSpinBox {{ min-height:20px; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #292c31, stop:1 #202226); color:#e5e7eb; border:1px solid #4a4e55; border-radius:7px; padding:6px 9px; selection-background-color:{ACCENT}; }}
        QComboBox#analysisField, QSpinBox#analysisField {{ margin-left:1px; }}
        QComboBox:hover, QSpinBox:hover {{ border-color:#69717c; background:#2d3136; }}
        QComboBox:focus, QSpinBox:focus {{ border-color:#ffffff; }}
        QComboBox::drop-down {{ width:25px; border:0; border-left:1px solid #444950; }}
        QComboBox QAbstractItemView {{ background:#25282c; color:#e5e7eb; border:1px solid #555a62; selection-background-color:#286da9; padding:3px; outline:0; }}
        QSpinBox {{ padding-right:26px; }}
        QSpinBox::up-button {{ subcontrol-origin:border; subcontrol-position:top right; width:24px; background:transparent; border:0; }}
        QSpinBox::down-button {{ subcontrol-origin:border; subcontrol-position:bottom right; width:24px; background:transparent; border:0; }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background:transparent; border:0; }}
        QSpinBox::up-arrow {{ image:url("{spin_up}"); width:8px; height:5px; }}
        QSpinBox::down-arrow {{ image:url("{spin_down}"); width:8px; height:5px; }}
        QSpinBox QLabel#intervalMinutesHint {{ color:#9ca3af; background:transparent; border:0; padding:0; }}
        QPushButton {{ min-height:20px; color:#dbe1e8; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #464a50, stop:1 #303339); border:1px solid #555a62; border-radius:8px; padding:7px 12px; font-weight:700; }}
        QPushButton:hover {{ color:white; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #59606a, stop:1 #3c4148); border-color:#737b87; }}
        QPushButton:pressed {{ background:#25282c; border-color:#ffffff; padding-top:9px; padding-bottom:5px; }}
        QPushButton#primary {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #327fc4, stop:1 #1f538d); border-color:#4b91cf; color:white; }}
        QPushButton#primary:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4295dd, stop:1 #2869ad); border-color:#70b3ed; }}
        QPushButton#warning {{ color:#211b08; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #f3d45f, stop:1 #c99f28); border-color:#f6df7c; }}
        QPushButton#warning:hover {{ color:#171204; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ffe782, stop:1 #ddb43a); border-color:#fff0a5; }}
        QPushButton#danger {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #b84b55, stop:1 #7f2932); border-color:#d26a73; color:white; }}
        QPushButton#danger:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #d25a65, stop:1 #97343e); border-color:#ef8991; }}
        QMessageBox QPushButton#confirmRemoveAll:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #d25a65, stop:1 #97343e); border-color:#ef8991; color:white; }}
        QMessageBox QPushButton#confirmRemoveAll:pressed {{ background:#7f2932; border-color:#d26a73; color:white; }}
        QMessageBox QLabel#qt_msgbox_informativelabel {{ color:#9ca3af; font-size:9px; }}
        QMessageBox QDialogButtonBox {{ padding:5px 4px 4px 4px; }}
        QMessageBox QDialogButtonBox QPushButton {{ margin:1px; }}
        QPushButton#secondary {{ color:#cbd5e1; }}
        QPushButton#mediaFooterButton {{ min-height:0; color:#cbd5e1; margin:1px; padding:6px 12px; }}
        QPushButton#mediaFooterButton:pressed {{ padding:8px 12px 4px 12px; }}
        QCheckBox#batchSwitch {{ color:#d1d5db; spacing:7px; font-size:9px; padding:4px 8px; border:1px solid #4a4e55; border-radius:7px; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #292c31, stop:1 #202226); }}
        QCheckBox#batchSwitch:hover {{ color:white; border-color:#69717c; background:#2d3136; }}
        QCheckBox#batchSwitch:checked {{ border-color:#555a62; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #292c31, stop:1 #202226); }}
        QCheckBox#batchSwitch:checked:hover {{ border-color:#69717c; background:#2d3136; }}
        QCheckBox#batchSwitch:disabled {{ color:#5d626a; border-color:#34373c; background:#222428; }}
        QCheckBox#batchSwitch::indicator {{ width:28px; height:14px; border-radius:7px; border:1px solid #555a62; background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #aeb5bf, stop:0.39 #aeb5bf, stop:0.40 #24272b, stop:1 #24272b); }}
        QCheckBox#batchSwitch::indicator:checked {{ border-color:#ffffff; background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #214f7b, stop:0.60 #214f7b, stop:0.61 #dbe7f2, stop:1 #dbe7f2); image:none; }}
        QCheckBox#batchSwitch::indicator:disabled {{ border-color:#34373c; background:#292b2f; }}
        QToolButton#mediaListMenu {{ background:transparent; border:0; border-radius:5px; padding:0; }}
        QToolButton#mediaListMenu:hover {{ background:#41454b; }}
        QToolButton#mediaListMenu:pressed {{ background:#25282c; }}
        QToolButton#selectionAction {{ background:#303339; border:1px solid #555a62; border-radius:7px; margin:1px; padding:5px; }}
        QToolButton#selectionAction:hover {{ background:#464a50; border-color:#737b87; }}
        QToolButton#selectionAction:pressed {{ background:#25282c; border-color:#ffffff; }}
        QToolButton#selectionClose {{ background:transparent; border:0; padding:0; }}
        QToolButton#selectionClose:hover {{ background:transparent; border:0; }}
        QToolButton#selectionClose:pressed {{ background:transparent; border:0; }}
        QPushButton:disabled {{ color:#5d626a; background:#222428; border-color:#34373c; }}
        QPushButton#primary:disabled,
        QPushButton#warning:disabled,
        QPushButton#danger:disabled,
        QPushButton#success:disabled {{ color:#555a62; background:#202226; border-color:#30343a; }}
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
        QFrame#transportGroup QToolButton {{ color:#d1d5db; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #464a50, stop:1 #303339); border:1px solid #555a62; border-radius:7px; margin:1px; padding:0; font-weight:700; }}
        QFrame#transportGroup QToolButton:hover {{ color:white; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #59606a, stop:1 #3c4148); border-color:#737b87; }}
        QFrame#transportGroup QToolButton:pressed {{ background:#24272b; border-color:#ffffff; padding-top:2px; }}
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
        QMenu#volumeMenu {{ padding:8px 6px; }}
        QMenu#volumeMenu::item {{ margin:0; padding:0; background:transparent; }}
        QMenu#volumeMenu::item:selected {{ background:transparent; }}
        QSlider#volumeSlider {{ background:transparent; border:0; padding:5px 0; }}
        QSlider#volumeSlider::groove:vertical {{ width:4px; background:#565b63; border:0; border-radius:2px; }}
        QSlider#volumeSlider::sub-page:vertical {{ background:#565b63; border:0; border-radius:2px; }}
        QSlider#volumeSlider::add-page:vertical {{ background:#d25a65; border:0; border-radius:2px; }}
        QSlider#volumeSlider::handle:vertical {{ background:#e5edf6; border:0; width:14px; height:14px; margin:0 -5px; border-radius:7px; }}
        QSlider#volumeSlider::handle:vertical:hover {{ background:white; border:0; }}
        QFrame#transportGroup QLineEdit#editableTimecode, QFrame#transportGroup QLineEdit#timecode {{ background:transparent; color:#f3f4f6; border:0; border-radius:0; padding:5px 4px; font-weight:700; }}
        QFrame#transportGroup QLineEdit#editableTimecode:hover, QFrame#transportGroup QLineEdit#editableTimecode:focus {{ background:#2d3136; color:white; border:1px solid #69717c; border-radius:7px; padding:4px 8px; }}
        QFrame#transportGroup QLabel#timeSeparator {{ background:transparent; color:#737b87; border:0; min-width:16px; max-width:16px; padding:5px 0; font-weight:700; }}
        QProgressBar {{ background:#202226; border:1px solid #41454b; border-radius:4px; height:7px; text-align:center; }}
        QProgressBar::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #188a51, stop:1 {SUCCESS}); border-radius:3px; }}
        QMenuBar {{ background:#2b2d31; color:#d1d5db; border-bottom:1px solid #45484e; padding:2px; }}
        QMenuBar::item {{ padding:5px 9px; border-radius:5px; }}
        QMenuBar::item:selected {{ background:#41454b; color:white; }}
        QMenuBar QToolButton#githubLink {{ color:#aeb4bd; background:transparent; border:0; border-radius:10px; margin:1px; padding:2px; }}
        QMenuBar QToolButton#githubLink:hover {{ color:white; background:#41454b; border-radius:10px; }}
        QMenuBar QToolButton#githubLink:pressed {{ background:#24272b; border-radius:10px; }}
        QMenuBar QLabel#githubUsername {{ color:#9ca3af; background:transparent; border:0; padding:0; font-weight:400; }}
        QMenu {{ background:#292c30; color:#e5e7eb; border:1px solid #50545b; padding:5px; }}
        QMenu::item {{ padding:7px 28px 7px 12px; border-radius:0; }}
        QMenu::item:selected {{ background:#286da9; color:white; }}
        QMenu#mediaContextMenu {{ padding:2px; }}
        QStatusBar {{ background:#2b2d31; color:{TEXT_MUTED}; border-top:1px solid #45484e; }}
        QAbstractScrollArea::corner {{ background:transparent; border:0; }}
        QGraphicsView#timelineView QWidget#qt_scrollarea_hcontainer {{ background:{BG_MAIN}; border:0; }}
        QGraphicsView#timelineView::corner {{ background:{BG_MAIN}; border:0; }}
        QScrollBar:horizontal {{ background:transparent; border:0; height:16px; margin:2px 4px; }}
        QScrollBar::handle:horizontal {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #464a50, stop:1 #303339); border:1px solid #555a62; min-width:52px; border-radius:6px; margin:2px 0; }}
        QScrollBar::handle:horizontal:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #59606a, stop:1 #3c4148); border-color:#737b87; }}
        QScrollBar::handle:horizontal:pressed {{ background:#25282c; border-color:#ffffff; }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background:transparent; border:0; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ background:transparent; border:0; width:0; }}
        QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {{ width:0; height:0; }}
        QScrollBar:vertical {{ background:transparent; border:0; width:16px; margin:4px 2px; }}
        QScrollBar::handle:vertical {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #464a50, stop:1 #303339); border:1px solid #555a62; min-height:52px; border-radius:6px; margin:0 2px; }}
        QScrollBar::handle:vertical:hover {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #59606a, stop:1 #3c4148); border-color:#737b87; }}
        QScrollBar::handle:vertical:pressed {{ background:#25282c; border-color:#ffffff; }}
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
        save_item = CompactMenuItem(
            themed_icon("fa6s.floppy-disk", scale_factor=.85),
            "Save Project", menu, hover_color=ACCENT)
        save_action = QWidgetAction(menu)
        save_action.setDefaultWidget(save_item)
        menu.addAction(save_action)
        selected = {"remove": False, "folder": False, "save": False}

        def select_remove():
            selected["remove"] = True
            menu.close()

        def select_folder():
            selected["folder"] = True
            menu.close()

        def select_save():
            selected["save"] = True
            menu.close()

        remove_item.activated.connect(select_remove)
        folder_item.activated.connect(select_folder)
        save_item.activated.connect(select_save)
        menu.exec(self.media_list.viewport().mapToGlobal(position))
        if selected["remove"]:
            self._remove_media_from_list(row)
        elif selected["folder"]:
            self._reveal_media_file(self.project.media[row].path)
        elif selected["save"]:
            self.media_list.setCurrentRow(row)
            self.save_project()

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
            zero = self._player_timecode(0, 0)
            self.clock.setText(zero)
            self.duration_clock.setText(zero)
            self.media_info.clear()
            self.media_info.hide()
            self._restore_selected_media_progress()

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
        zero = self._player_timecode(0, 0)
        self.clock.setText(zero)
        self.duration_clock.setText(zero)
        self.media_info.clear()
        self.media_info.hide()
        self._restore_selected_media_progress()
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
            self._refresh_queue_mode_label()
            self.statusBar().showMessage(
                "A queue requires at least two videos in Media.", 6000)
            self._refresh_action_button()
            return
        self.batch_mode = bool(enabled)
        self._refresh_queue_mode_label()
        self._refresh_action_button()
        self.statusBar().showMessage(
            "Automatic queue enabled · processing remains sequential" if enabled
            else "Automatic queue disabled", 4000)

    def _refresh_queue_mode_label(self):
        label = getattr(self, "analysis_field_labels", {}).get("Mode")
        if label is not None:
            label.setText("Mode (Queue)" if self.batch_mode else "Mode")

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
        self.mode.setToolTip(MODE_TOOLTIPS.get(mode, ""))
        show_interval = mode == "interval"
        self.interval.setVisible(show_interval)
        self.analysis_field_labels["Interval"].setVisible(show_interval)
        self.interval.setEnabled(show_interval)
        self.sensitivity.setEnabled(mode != "interval")
        auto_index = self.sensitivity.findText("Auto")
        if mode == "faces":
            if self.sensitivity.currentText() == "Auto":
                self.sensitivity.setCurrentText("Normal")
            if auto_index >= 0:
                self.sensitivity.removeItem(auto_index)
        elif auto_index < 0:
            self.sensitivity.addItem("Auto")
        self._sync_acceleration_options(mode)
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
        self._refresh_media_metrics()
        self._refresh_action_button()
        self._update_analysis_controls_access()

    def _update_analysis_controls_access(self):
        """Analysis settings become immutable once this media has results."""
        locked = bool(self._current_media_is_processing() or (
            self.current_media and self.current_media.analyzed))
        editable = not locked
        mode = self.mode.currentData()
        self.mode.setEnabled(editable)
        self.sensitivity.setEnabled(editable and mode != "interval")
        self.interval.setEnabled(editable and mode == "interval")
        self.accel.setEnabled(editable)
        for title in ("Mode", "Sensitivity", "Interval", "Acceleration"):
            label = self.analysis_field_labels.get(title)
            if label is not None:
                label.setEnabled(editable)

    def _current_media_is_processing(self):
        return bool(
            self.task and self.current_media is not None and
            self.current_media is self.processing_media)

    def _sync_acceleration_options(self, mode):
        compat = MODE_ACCEL_COMPAT.get(mode, {})
        allowed = ({"cpu"} |
                   (compat.get("encoder", set()) &
                    self.available_encoder_accel) |
                   (compat.get("inference", set()) &
                    self.available_inference_accel))
        selected = self.accel.currentData() or "cpu"
        self.accel.blockSignals(True)
        self.accel.clear()
        for value in ACCEL_OPTIONS:
            if value in allowed:
                self.accel.addItem(value.upper(), value)
        selected_index = self.accel.findData(selected)
        self.accel.setCurrentIndex(
            selected_index if selected_index >= 0 else self.accel.findData("cpu"))
        self.accel.blockSignals(False)

    @staticmethod
    def _playback_proxy_path(media):
        return os.path.join(
            USER_DATA_DIR, "cache", media.media_id, "playback_proxy_v2.mp4")

    def _playback_source(self, media):
        proxy = self._playback_proxy_path(media)
        try:
            if (os.path.getsize(proxy) > 0 and
                    os.path.getmtime(proxy) >= os.path.getmtime(media.path)):
                return proxy
        except OSError:
            pass
        return media.path

    def _ensure_playback_proxy(self, media):
        if (not media or media.media_id in self._playback_proxy_tasks or
                (media.width <= 1280 and media.height <= 720)):
            return
        target = self._playback_proxy_path(media)
        if self._playback_source(media) == target:
            return

        def work(task):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            temporary = os.path.join(
                os.path.dirname(target), "playback_proxy_v2.part.mp4")
            try:
                create_playback_proxy(
                    media.path, temporary, media.fps, media.total_frames,
                    stop_cb=task.isInterruptionRequested)
                if task.isInterruptionRequested():
                    return None
                os.replace(temporary, target)
                return target
            finally:
                if os.path.exists(temporary):
                    try:
                        os.remove(temporary)
                    except OSError:
                        pass

        task = Task(work, self)
        self._playback_proxy_tasks[media.media_id] = task
        task.result.connect(
            lambda path, m=media, t=task:
            self._playback_proxy_ready(m, path, t))
        task.failed.connect(
            lambda _message, m=media, t=task:
            self._playback_proxy_finished(m, t))
        task.cancelled.connect(
            lambda m=media, t=task:
            self._playback_proxy_finished(m, t))
        task.start()
        self.statusBar().showMessage(
            "Preparing a lightweight playback proxy in the background", 5000)

    def _playback_proxy_finished(self, media, task):
        if self._playback_proxy_tasks.get(media.media_id) is task:
            self._playback_proxy_tasks.pop(media.media_id, None)
        task.deleteLater()

    def _playback_proxy_ready(self, media, path, task):
        self._playback_proxy_finished(media, task)
        if not path or self.current_media is not media:
            return
        if (self.player.playbackState() ==
                QMediaPlayer.PlaybackState.PlayingState or
                (self.timeline.playhead and self.timeline.playhead.dragging)):
            self._pending_proxy_path = (media.media_id, path)
            return
        self._activate_playback_proxy(media, path)

    def _activate_playback_proxy(self, media, path):
        if self.current_media is not media or not os.path.exists(path):
            return
        current = self.player.source().toLocalFile()
        if current and os.path.normcase(os.path.abspath(current)) == os.path.normcase(
                os.path.abspath(path)):
            return
        self._pending_proxy_path = None
        self._pending_proxy_restore = (media.media_id, self.player.position())
        self.player.setSource(QUrl.fromLocalFile(path))

    def _activate_pending_playback_proxy(self):
        pending = self._pending_proxy_path
        if not pending or not self.current_media:
            return
        media_id, path = pending
        if self.current_media.media_id == media_id:
            self._activate_playback_proxy(self.current_media, path)
        else:
            self._pending_proxy_path = None

    def _select_media(self, row):
        if (self.task and self.batch_active and self.processing_media is not None and
                0 <= row < len(self.project.media) and
                self.project.media[row] is not self.processing_media):
            # Keep the list enabled so its scrollbar remains usable, but the
            # active queue item continues to own the workspace selection.
            try:
                processing_row = self.project.media.index(self.processing_media)
            except ValueError:
                return
            self.media_list.blockSignals(True)
            self.media_list.setCurrentRow(processing_row)
            self.media_list.blockSignals(False)
            return
        if not 0 <= row < len(self.project.media): return
        self._end_edit_preview()
        self._pending_proxy_path = None
        self._pending_proxy_restore = None
        self.selected_cut = None
        self.selected_segment = None
        self._manual_face_draft = None
        self.current_media = self.project.media[row]
        mode_index = self.mode.findData(self.current_media.mode)
        if mode_index >= 0:
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(mode_index)
            self.mode.blockSignals(False)
        self._mode_changed()
        sensitivity = self.current_media.sensitivity
        if self.current_media.mode == "faces" and sensitivity == "Auto":
            sensitivity = "Normal"
            self.current_media.sensitivity = sensitivity
        if self.sensitivity.findText(sensitivity) >= 0:
            self.sensitivity.setCurrentText(sensitivity)
        self.interval.setValue(max(1, int(getattr(
            self.current_media, "interval", 10))))
        acceleration = getattr(self.current_media, "acceleration", "cpu")
        acceleration_index = self.accel.findData(acceleration)
        if acceleration_index >= 0:
            self.accel.setCurrentIndex(acceleration_index)
        self.preview_end_ms = None
        self.preview_start_ms = None
        self.preview_armed = False
        self.timeline.show_detected_faces = self.current_media.show_detected_faces
        self.timeline.show_saved_faces = self.current_media.show_saved_faces
        for button, checked in (
                (self.face_detected_filter_button,
                 self.current_media.show_detected_faces),
                (self.face_saved_filter_button,
                 self.current_media.show_saved_faces)):
            button.blockSignals(True)
            button.setChecked(checked)
            button.setIcon(themed_icon(
                "fa6s.eye" if checked else "fa6s.eye-slash"))
            button.blockSignals(False)
        self.face_detected_filter_button.setToolTip(
            "Hide detected faces that are not included"
            if self.current_media.show_detected_faces else
            "Show detected faces that are not included")
        self.face_saved_filter_button.setToolTip(
            "Hide saved faces" if self.current_media.show_saved_faces else
            "Show saved faces")
        self.timeline.set_media(self.current_media)
        self._restore_live_results_for_current()
        self._set_selection_available(self.current_media.analyzed)
        playback_source = self._playback_source(self.current_media)
        self.player.setSource(QUrl.fromLocalFile(playback_source))
        if playback_source == self.current_media.path:
            self._ensure_playback_proxy(self.current_media)
        if self._processing_preview_selected():
            self.player.pause()
            self.viewer_stack.setCurrentWidget(self.processing_preview)
            self.video.hide()
            self.processing_preview.show()
        else:
            self.viewer_stack.setCurrentWidget(self.video)
            self.video.show()
        if self.task:
            self.task.preview_visible = self._processing_preview_selected()
        self._update_transport_access()
        self._refresh_action_button()
        self._update_media_info_text()
        timeline_duration = self._timeline_duration()
        duration_text = self._player_timecode(timeline_duration, timeline_duration)
        time_width = max(
            92, self.clock.fontMetrics().horizontalAdvance(duration_text) + 24)
        self.clock.setFixedWidth(time_width)
        self.duration_clock.setFixedWidth(time_width)
        self.media_info.setToolTip(self.current_media.path)
        self._position_media_info()
        self.media_info.show()
        self._refresh_progress_state_icon()
        self._refresh_transport_timeline()
        self._refresh_media_metrics()
        self._restore_selected_media_progress()
        self.show_cut(None)

    def _restore_selected_media_progress(self):
        """Show progress belonging to the selected media, not the previous task."""
        media = self.current_media
        if self._current_media_is_processing():
            value, label = getattr(self.task, "media_progress", {}).get(
                media.media_id, (0, "Processing"))
        elif media and media.exported:
            value, label = 1000, "Complete"
        elif media and media.analyzed:
            value, label = 1000, "Video analyzed"
        else:
            value, label = 0, "Ready"
        self._set_progress_value(value, immediate=True)
        self.progress_label.setText(label)
        self._refresh_progress_state_icon()
        self._update_transport_access()

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
        if not self.current_media:
            return
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
        if self._edit_preview_active:
            target = self._edit_preview_frame
            if target is not None:
                self._play_after_edit_sync = True
                self._flush_interactive_seek(round(
                    self.current_media.seconds_for_frame(target) * 1000))
                return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState: self.player.pause()
        else: self.player.play()

    def _update_play_button(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        if playing:
            self._reset_frame_step_target()
        elif self._pending_proxy_path:
            QTimer.singleShot(0, self._activate_pending_playback_proxy)
        self.play_button.setIcon(self._pause_icon if playing else self._play_icon)

    def toggle_mute(self):
        self.audio.setMuted(not self.audio.isMuted())

    def _update_volume_button(self, muted):
        self.volume_button.setIcon(self._muted_icon if muted else self._volume_icon)
        self.volume_button.setToolTip("Unmute audio" if muted else "Mute audio")

    def relative_seek(self, amount):
        current = self._source_to_timeline_seconds(self.player.position() / 1000)
        self.seek_timeline_ms(round(current * 1000) + amount)

    def _reset_frame_step_target(self):
        self._frame_step_target = None
        self._frame_step_media_id = None

    def _queue_interactive_seek(self, value):
        """Keep one decoder seek in flight; the newest pending one wins."""
        self._pending_interactive_seek_ms = int(value)
        if self._interactive_seek_inflight_ms is None:
            self._apply_pending_interactive_seek()

    def _apply_pending_interactive_seek(self):
        value = self._pending_interactive_seek_ms
        if value is None:
            return
        self._pending_interactive_seek_ms = None
        self._interactive_seek_inflight_ms = value
        self.seek_player_ms(value, reset_frame_step=False)
        self._interactive_seek_timer.start()

    def _interactive_seek_timeout(self):
        # Some backends do not report the exact requested millisecond.  The
        # watchdog releases the slot without allowing a queue to accumulate.
        self._interactive_seek_inflight_ms = None
        if self._pending_interactive_seek_ms is not None:
            self._apply_pending_interactive_seek()

    def _flush_interactive_seek(self, value=None):
        if value is not None:
            self._pending_interactive_seek_ms = int(value)
        self._interactive_seek_timer.stop()
        self._interactive_seek_inflight_ms = None
        self._apply_pending_interactive_seek()

    def _request_timeline_seek(self, seconds):
        self._reset_frame_step_target()
        self._request_edit_preview(float(seconds))

    def _begin_timeline_scrub(self):
        self.player.pause()
        self._interactive_seek_timer.stop()
        self._pending_interactive_seek_ms = None
        self._interactive_seek_inflight_ms = None
        self._reset_frame_step_target()
        self.preview_end_ms = None
        self.preview_start_ms = None
        self.preview_armed = False
        if self.current_media:
            self._request_edit_preview(self.player.position() / 1000)

    def _finish_timeline_scrub(self, seconds):
        self._reset_frame_step_target()
        self._request_edit_preview(float(seconds))
        self._flush_interactive_seek(round(float(seconds) * 1000))

    def _request_edit_preview(self, seconds):
        media = self.current_media
        if (not media or self._current_media_is_processing() or
                self._processing_preview_selected()):
            return
        seconds = max(0.0, min(float(seconds), media.duration))
        frame = min(media.total_frames - 1,
                    media.frame_for_seconds(seconds))
        self._edit_preview_active = True
        self._edit_preview_frame = frame
        self._edit_preview_media_id = media.media_id
        timeline_seconds = self._source_to_timeline_seconds(seconds)
        if not self.seek.isSliderDown():
            self.seek.setValue(round(timeline_seconds * 1000))
        self.timeline.set_position(seconds)
        duration = self._timeline_duration()
        if not self.clock.hasFocus():
            self.clock.setText(self._player_timecode(timeline_seconds, duration))
        self._ensure_frame_decoder()
        self._frame_decoder.request(
            media.media_id, self._playback_source(media), frame, media.fps)

    def _ensure_frame_decoder(self):
        if self._frame_decoder is not None:
            return
        self._frame_decoder = FramePreviewDecoder(self)
        self._frame_decoder.frameReady.connect(self._show_edit_preview_frame)
        self._frame_decoder.start()

    def _release_frame_decoder(self):
        decoder = self._frame_decoder
        if decoder is None:
            return
        self._frame_decoder = None
        decoder.stop()
        decoder.wait(3000)
        decoder.deleteLater()

    def _show_edit_preview_frame(self, media_id, frame, image):
        if (not self._edit_preview_active or not self.current_media or
                self.current_media.media_id != media_id or
                self._edit_preview_media_id != media_id or
                self._edit_preview_frame != frame or image.isNull()):
            return
        self.processing_preview.setStyleSheet("background:#000000; border:0;")
        self.processing_preview.set_source_pixmap(QPixmap.fromImage(image))
        self.viewer_stack.setCurrentWidget(self.processing_preview)
        self.video.hide()
        self.processing_preview.show()

    def _end_edit_preview(self):
        if not self._edit_preview_active:
            return
        self._edit_preview_active = False
        self._edit_preview_frame = None
        self._edit_preview_media_id = None
        if not self._processing_preview_selected():
            self.processing_preview.clear_source()
            self.processing_preview.hide()
            self.viewer_stack.setCurrentWidget(self.video)
            self.video.show()

    def step_video_frame(self, direction):
        media = self.current_media
        if not media or self._current_media_is_processing() or not direction:
            return
        ranges = self._timeline_ranges(media)
        if not ranges:
            return
        same_sequence = self._frame_step_media_id == media.media_id
        current = (self._frame_step_target if same_sequence and
                   self._frame_step_target is not None else
                   media.frame_for_seconds(self.player.position() / 1000))
        target = max(0, min(media.total_frames - 1,
                            current + (1 if direction > 0 else -1)))
        if not any(start <= target < end for start, end in ranges):
            if direction > 0:
                target = next((start for start, _end in ranges
                               if start >= target), ranges[-1][1] - 1)
            else:
                target = next((end - 1 for start, end in reversed(ranges)
                               if end - 1 <= target), ranges[0][0])
        self._frame_step_target = target
        self._frame_step_media_id = media.media_id
        self.player.pause()
        self._request_edit_preview(media.seconds_for_frame(target))

    def _timeline_ranges(self, media=None):
        media = media or self.current_media
        if not media:
            return []
        removed = set(media.removed_segments)
        return [segment for index, segment in enumerate(media.segments(), 1)
                if index not in removed]

    def _timeline_duration(self, media=None):
        media = media or self.current_media
        if not media:
            return 0.0
        return sum(
            max(0.0,
                media.seconds_for_frame(end) - media.seconds_for_frame(start))
            for start, end in self._timeline_ranges(media))

    def _player_timecode(self, seconds, duration=None):
        formatter = player_timecode if self.show_milliseconds else transport_timecode
        return formatter(seconds, duration)

    def _source_to_timeline_seconds(self, seconds, media=None):
        media = media or self.current_media
        if not media:
            return max(0.0, float(seconds))
        source_seconds = max(0.0, min(media.duration, float(seconds)))
        cursor = 0.0
        for start, end in self._timeline_ranges(media):
            start_seconds = media.seconds_for_frame(start)
            end_seconds = media.seconds_for_frame(end)
            if source_seconds < start_seconds:
                return cursor
            if source_seconds <= end_seconds:
                return cursor + source_seconds - start_seconds
            cursor += max(0.0, end_seconds - start_seconds)
        return cursor

    def _timeline_to_source_seconds(self, seconds, media=None):
        media = media or self.current_media
        if not media:
            return max(0.0, float(seconds))
        timeline_seconds = max(0.0, min(
            self._timeline_duration(media), float(seconds)))
        ranges = self._timeline_ranges(media)
        cursor = 0.0
        for start, end in ranges:
            start_seconds = media.seconds_for_frame(start)
            end_seconds = media.seconds_for_frame(end)
            length = max(0.0, end_seconds - start_seconds)
            if timeline_seconds < cursor + length - 1e-9:
                return min(
                    end_seconds,
                    start_seconds + timeline_seconds - cursor)
            cursor += length
        return media.seconds_for_frame(ranges[-1][1]) if ranges else 0.0

    def _refresh_transport_timeline(self):
        if not self.current_media or self._processing_preview_selected():
            return
        duration = self._timeline_duration()
        self.seek.setRange(0, max(0, round(duration * 1000)))
        position = self._source_to_timeline_seconds(self.player.position() / 1000)
        if not self.seek.isSliderDown():
            self.seek.setValue(round(position * 1000))
        if not self.clock.hasFocus():
            self.clock.setText(self._player_timecode(position, duration))
        self.duration_clock.setText(self._player_timecode(duration, duration))

    def _timeline_content_changed(self):
        self._mark_current_unexported()
        self._refresh_transport_timeline()
        self._refresh_media_metrics()
        self._refresh_action_button()

    def _begin_timeline_edit(self):
        """Stop transient previews before a cut boundary is edited."""
        self.player.pause()
        self._interactive_seek_timer.stop()
        self._pending_interactive_seek_ms = None
        self._interactive_seek_inflight_ms = None
        self._reset_frame_step_target()
        self.preview_end_ms = None
        self.preview_start_ms = None
        self.preview_armed = False
        self.timeline.clear_clip_duration()

    def seek_timeline_ms(self, value):
        duration = self._timeline_duration()
        timeline_seconds = max(0.0, min(duration, int(value) / 1000))
        self.seek_player_ms(round(
            self._timeline_to_source_seconds(timeline_seconds) * 1000))

    def seek_player_ms(self, value, reset_frame_step=True):
        if reset_frame_step:
            self._reset_frame_step_target()
            self._end_edit_preview()
            self._interactive_seek_timer.stop()
            self._pending_interactive_seek_ms = None
            self._interactive_seek_inflight_ms = None
        if self.viewer_stack.currentWidget() is self.face_review_preview:
            self.show_cut(None)
        self.timeline.clear_clip_duration()
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
        editable = has_media and not self._current_media_is_processing()
        self.selection_card.setEnabled(editable)
        self.selection_result_actions.setVisible(available)
        self.preanalysis_actions.setVisible(not available)
        self.preanalysis_cut_button.setEnabled(editable)
        self.preanalysis_disable_button.setEnabled(False)
        if available:
            if self.selected_cut is None and self.selected_segment is None:
                self.selection_title.setText("No event selected")
                self.selection_detail.setText("Select a marker to inspect it.")
            self._sync_progress_metrics_visibility()
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
        self._sync_progress_metrics_visibility()

    def clear_selection(self):
        if self._manual_face_draft is not None:
            self.cancel_manual_face()
            return
        self.timeline.scene().clearSelection()
        self.selected_cut = None
        self.selected_segment = None
        if self.current_media and self.current_media.analyzed:
            self.show_cut(None)
        else:
            self._set_selection_available(False)

    def _toggle_face_indicators(self, category, visible):
        visible = bool(visible)
        if category == "detected":
            self.timeline.show_detected_faces = visible
            if self.current_media:
                self.current_media.show_detected_faces = visible
            button = self.face_detected_filter_button
            button.setToolTip(
                "Hide detected faces that are not included" if visible else
                "Show detected faces that are not included")
        else:
            self.timeline.show_saved_faces = visible
            if self.current_media:
                self.current_media.show_saved_faces = visible
            button = self.face_saved_filter_button
            button.setToolTip("Hide saved faces" if visible else
                              "Show saved faces")
        button.setIcon(themed_icon("fa6s.eye" if visible else "fa6s.eye-slash"))
        selected = self.selected_cut
        hides_selected = isinstance(selected, FaceEvent) and (
            (selected.enabled and not self.timeline.show_saved_faces) or
            (not selected.enabled and not self.timeline.show_detected_faces))
        if hides_selected:
            self.clear_selection()
        self.timeline.rebuild()

    def show_segment(self, start_seconds, end_seconds, index):
        if not self.current_media or end_seconds <= start_seconds:
            return
        if not self.current_media.analyzed:
            self.select_preanalysis_segment(start_seconds, end_seconds, index)
            self.play_segment(start_seconds, end_seconds)
            return
        self._set_selection_available(True)
        self._manual_face_draft = None
        self._set_manual_face_controls(False)
        self.selected_cut = None
        self.selected_segment = (int(index), float(start_seconds), float(end_seconds))
        self._sync_progress_metrics_visibility()
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
            f"Start: {transport_timecode(start_seconds, self.current_media.duration)}\n"
            f"End: {transport_timecode(end_seconds, self.current_media.duration)}\n"
            f"Duration: {transport_timecode(duration, self.current_media.duration)}\n"
            f"Status: {'Excluded' if excluded else 'Included'}\n"
            "Move the boundary markers to adjust this range.")
        self.selection_preview.clear_source()
        frame_number = self.current_media.frame_for_seconds(midpoint)
        self._show_segment_still(frame_number)
        self.play_segment(start_seconds, end_seconds)

    def select_preanalysis_segment(self, start_seconds, end_seconds, index):
        if (not self.current_media or self.current_media.analyzed or
                end_seconds <= start_seconds):
            return
        self._set_selection_available(False)
        self.selected_segment = (int(index), float(start_seconds), float(end_seconds))
        self._sync_progress_metrics_visibility()
        excluded = index in self.current_media.excluded_segments
        self.selection_title.setText(f"Clip {index:02d}")
        self.selection_detail.setText(
            "This clip is hidden from analysis and can still be previewed or restored."
            if excluded else
            "This clip will be included in the analysis.")
        self.preanalysis_disable_button.setEnabled(
            not self._current_media_is_processing())
        self.preanalysis_disable_button.setIcon(themed_icon(
            "fa6s.rotate-left" if excluded else "fa6s.trash"))
        self.preanalysis_disable_button.setToolTip(
            "Restore selected clip to analysis" if excluded else
            "Hide selected clip from analysis")

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
            f"Previewing clip · "
            f"{transport_timecode(start_seconds, self.current_media.duration)} – "
            f"{transport_timecode(end_seconds, self.current_media.duration)}")

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
        if self._interactive_seek_inflight_ms is not None:
            fps = self.current_media.fps if self.current_media else 30.0
            tolerance = max(20, round(1000 / max(.001, fps)))
            if abs(int(value) - self._interactive_seek_inflight_ms) <= tolerance:
                self._interactive_seek_timer.stop()
                self._interactive_seek_inflight_ms = None
                if self._pending_interactive_seek_ms is not None:
                    QTimer.singleShot(0, self._apply_pending_interactive_seek)
        if (self._edit_preview_active and self._edit_preview_frame is not None and
                self.current_media and
                not (self.timeline.playhead and self.timeline.playhead.dragging)):
            target_ms = round(self.current_media.seconds_for_frame(
                self._edit_preview_frame) * 1000)
            tolerance = max(
                20, round(1000 / max(.001, self.current_media.fps)))
            if abs(int(value) - target_ms) <= tolerance:
                play_after_sync = self._play_after_edit_sync
                self._play_after_edit_sync = False
                self._end_edit_preview()
                if play_after_sync:
                    QTimer.singleShot(0, self.player.play)
        source_seconds = value / 1000
        if self.current_media and self.current_media.removed_segments:
            ranges = self._timeline_ranges()
            source_frame = self.current_media.frame_for_seconds(source_seconds)
            inside = any(start <= source_frame < end for start, end in ranges)
            at_end = source_frame >= self.current_media.total_frames
            if not inside and not at_end:
                next_start = next((start for start, _end in ranges
                                   if start > source_frame), None)
                if next_start is not None:
                    target = round(
                        self.current_media.seconds_for_frame(next_start) * 1000)
                    if abs(target - value) > 1:
                        self.player.setPosition(target)
                        return
                elif ranges:
                    self.player.pause()
                    target = round(self.current_media.seconds_for_frame(
                        ranges[-1][1]) * 1000)
                    if abs(target - value) > 1:
                        self.player.setPosition(target)
                        return
                else:
                    self.player.pause()
        timeline_seconds = self._source_to_timeline_seconds(source_seconds)
        timeline_duration = self._timeline_duration()
        if not self.seek.isSliderDown():
            self.seek.setValue(round(timeline_seconds * 1000))
        self.timeline.set_position(source_seconds)
        if not self.clock.hasFocus():
            self.clock.setText(self._player_timecode(
                timeline_seconds, timeline_duration))
        self.duration_clock.setText(self._player_timecode(
            timeline_duration, timeline_duration))
        if (self.preview_armed and self.preview_end_ms is not None and
                value >= self.preview_end_ms - 10):
            end = self.preview_end_ms
            self.preview_end_ms = None
            self.preview_start_ms = None
            self.preview_armed = False
            self.player.pause()
            self.player.setPosition(end)
            self.timeline.clear_clip_duration()
            self.statusBar().showMessage("Clip preview complete", 3000)

    def _duration_changed(self, value):
        if self.current_media and value > 0:
            source = self.player.source().toLocalFile()
            same_source = bool(source) and os.path.normcase(os.path.abspath(source)) == os.path.normcase(
                os.path.abspath(self.current_media.path))
            actual_duration = value / 1000
            if same_source and abs(self.current_media.duration - actual_duration) >= .001:
                self.current_media.duration = actual_duration
                self.timeline.rebuild()
                duration_text = self._player_timecode(actual_duration, actual_duration)
                self._update_media_info_text()
                time_width = max(
                    92, self.clock.fontMetrics().horizontalAdvance(duration_text) + 24)
                self.clock.setFixedWidth(time_width)
                self.duration_clock.setFixedWidth(time_width)
        duration = self._timeline_duration() if self.current_media else value / 1000
        self.seek.setRange(0, max(0, round(duration * 1000)))
        self.duration_clock.setText(self._player_timecode(duration, duration))

    def _media_status_changed(self, status):
        visible_statuses = {
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.BufferingMedia,
            QMediaPlayer.MediaStatus.StalledMedia,
            QMediaPlayer.MediaStatus.EndOfMedia,
        }
        if (status in visible_statuses and self._pending_proxy_restore and
                self.current_media):
            media_id, position = self._pending_proxy_restore
            if self.current_media.media_id == media_id:
                self._pending_proxy_restore = None
                self.player.setPosition(max(0, int(position)))
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
        self._processing_preview_started_at = time.monotonic()
        self._processing_preview_first_frame_at = None
        self.processing_preview.clear_source()
        self.processing_preview.setText("Preparing preview")
        self.processing_preview.setStyleSheet(
            "background:#000000; color:#8b949e; border:0; font-size:12px;")
        if self.task:
            self.task.preview_paused = False
            self.task.media_progress.setdefault(
                media.media_id, (0, "Processing"))
        if self._processing_preview_selected():
            self.player.pause()
            self.viewer_stack.setCurrentWidget(self.processing_preview)
            self.video.hide()
            self.processing_preview.show()
        if self.task:
            self.task.preview_visible = self._processing_preview_selected()
        self._update_media_info_text()
        self._sync_processing_position(0.0)
        self._update_transport_access()

    def _processing_preview_selected(self):
        return bool(
            self._processing_preview_active and self.current_media is not None and
            self.current_media is getattr(self, "_processing_preview_media", None))

    def _update_transport_access(self):
        has_media = self.current_media is not None
        locked = self._current_media_is_processing()
        controls_enabled = has_media and not locked
        self.timeline.setEnabled(controls_enabled)
        self.seek.setEnabled(controls_enabled)
        self.clock.setEnabled(controls_enabled)
        self.duration_clock.setEnabled(controls_enabled)
        for button in (self.previous_button, self.back_button, self.forward_button,
                       self.following_button, self.volume_button):
            button.setEnabled(controls_enabled)
        self.selection_card.setEnabled(controls_enabled)
        self.add_cut_action.setEnabled(controls_enabled)
        if hasattr(self, "preanalysis_cut_button"):
            self.preanalysis_cut_button.setEnabled(controls_enabled)
        # Keep the locked transport visually consistent with processing preview:
        # the main control stays blue while its action remains guarded when
        # there is no selected media.
        self.play_button.setEnabled(has_media and not locked)
        if locked and self.task:
            paused = self.task.preview_paused
            self.play_button.setIcon(self._play_icon if paused else self._pause_icon)
            self.play_button.setToolTip(
                "Continue processing preview" if paused else "Pause processing preview")
        elif not has_media:
            self.play_button.setIcon(self._play_icon)
            self.play_button.setToolTip("Import a video to use the player")
        else:
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
            if self._processing_preview_first_frame_at is None:
                self._processing_preview_first_frame_at = time.monotonic()
            if seconds >= 0:
                self._animate_processing_position(seconds)
        except Exception:
            pass

    def _sync_processing_position(self, seconds):
        media = getattr(self, "_processing_preview_media", None)
        if not self._processing_preview_selected() or not media:
            return
        seconds = max(0.0, min(float(media.duration), float(seconds)))
        timeline_seconds = self._source_to_timeline_seconds(seconds, media)
        timeline_duration = self._timeline_duration(media)
        if not self.seek.isSliderDown():
            self.seek.setValue(round(timeline_seconds * 1000))
        self.timeline.set_position(seconds)
        self.clock.setText(self._player_timecode(timeline_seconds, timeline_duration))
        self.duration_clock.setText(self._player_timecode(
            timeline_duration, timeline_duration))

    def _animate_processing_position(self, seconds):
        media = getattr(self, "_processing_preview_media", None)
        if not self._processing_preview_selected() or not media:
            return
        target = max(0.0, min(float(media.duration), float(seconds)))
        current = max(0.0, min(
            float(media.duration), self.timeline.current_seconds))
        if target < current:
            return
        if abs(target - current) < 0.001:
            self.processing_position_animation.stop()
            self._sync_processing_position(target)
            return
        span_ratio = (target - current) / max(float(media.duration), 0.001)
        self.processing_position_animation.stop()
        self.processing_position_animation.setStartValue(current)
        self.processing_position_animation.setEndValue(target)
        self.processing_position_animation.setDuration(
            max(180, min(900, round(220 + span_ratio * 2400))))
        self.processing_position_animation.start()

    def _end_processing_preview(self):
        if not self._processing_preview_active:
            return
        was_selected = self._processing_preview_selected()
        restore = getattr(self, "_processing_restore_position", self.player.position())
        self.processing_position_animation.stop()
        self._processing_preview_active = False
        self._processing_preview_media = None
        if self.task:
            self.task.preview_visible = False
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
            duration = self._timeline_duration()
            position = self._source_to_timeline_seconds(
                self.player.position() / 1000)
            self.clock.setText(self._player_timecode(position, duration))
            return
        self.seek_timeline_ms(round(seconds * 1000))

    def _step_from_timecode(self, seconds):
        duration = self._timeline_duration()
        seconds = max(0.0, min(duration, float(seconds)))
        self.clock.setText(self._player_timecode(seconds, duration))
        self.seek_timeline_ms(round(seconds * 1000))

    def add_cut(self):
        if self._current_media_is_processing():
            return
        self.add_cut_at(self.player.position() / 1000)

    def add_cut_at(self, seconds):
        if self._current_media_is_processing() or not self.current_media:
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
            self._refresh_media_metrics()
            self._refresh_action_button()

    def _mark_current_unexported(self):
        if self.current_media and self.current_media.exported:
            self.current_media.exported = False
            self._refresh_media_item_states()
            self._refresh_action_button()

    def show_video_context_menu(self, point):
        media = self.current_media
        if (not media or self._current_media_is_processing() or
                self._processing_preview_selected()):
            return
        target_frame = max(0, min(
            media.total_frames - 1,
            media.frame_for_seconds(self.player.position() / 1000)))
        # Parent the popup to the main window so the video's transparent local
        # stylesheet cannot override the standard compact-menu background.
        menu = QMenu(self)
        menu.setObjectName("mediaContextMenu")
        selected = {"crop": False, "save": False}

        if media.mode == "faces" and media.analyzed:
            crop_item = CompactMenuItem(
                themed_icon("fa6s.scissors", scale_factor=.85),
                "Crop new face", menu, hover_color="#286da9")
            crop_action = QWidgetAction(menu)
            crop_action.setDefaultWidget(crop_item)
            menu.addAction(crop_action)

            def select_crop():
                selected["crop"] = True
                menu.close()

            crop_item.activated.connect(select_crop)

        save_item = CompactMenuItem(
            themed_icon("fa6s.floppy-disk", scale_factor=.85),
            "Save Frame", menu, hover_color=ACCENT)
        save_action = QWidgetAction(menu)
        save_action.setDefaultWidget(save_item)
        menu.addAction(save_action)

        def select_save():
            selected["save"] = True
            menu.close()

        save_item.activated.connect(select_save)
        menu.exec(self.video.mapToGlobal(point.toPoint()))
        if selected["crop"]:
            self.open_manual_face_editor(point)
        elif selected["save"]:
            self.save_video_frame(target_frame)

    @staticmethod
    def _decode_original_video_frame(path, target_frame, fps):
        """Decode one source-resolution frame, preferring the lightest path."""
        cap = cv2.VideoCapture(path)
        try:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_frame))
                ok, frame = cap.read()
                if ok and frame is not None:
                    return frame
        finally:
            cap.release()

        container = av.open(path)
        try:
            stream = container.streams.video[0]
            seconds = max(0.0, (int(target_frame) - 2) / max(float(fps), .001))
            container.seek(
                int(seconds / float(stream.time_base)), stream=stream,
                backward=True, any_frame=False)
            for decoded in container.decode(stream):
                if decoded.pts is None:
                    continue
                index = max(0, round(
                    float(decoded.pts * stream.time_base) * float(fps)))
                if index >= int(target_frame):
                    return decoded.to_ndarray(format="bgr24")
        finally:
            container.close()
        return None

    def save_video_frame(self, frame_number=None):
        media = self.current_media
        if (not media or self._current_media_is_processing() or
                self._processing_preview_selected()):
            return
        frame_number = (media.frame_for_seconds(self.player.position() / 1000)
                        if frame_number is None else int(frame_number))
        frame_number = max(0, min(media.total_frames - 1, frame_number))
        digits = max(6, len(str(max(1, media.total_frames))))
        suggested = Path(media.path).with_name(
            f"{Path(media.path).stem}_frame_{frame_number:0{digits}d}.png")
        target, selected_filter = QFileDialog.getSaveFileName(
            self, "Save exact video frame", str(suggested),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)")
        if not target:
            return
        suffix = Path(target).suffix.lower()
        jpeg_selected = selected_filter.startswith("JPEG")
        if jpeg_selected and suffix not in {".jpg", ".jpeg"}:
            suffix = ".jpg"
            target = str(Path(target).with_suffix(suffix))
        elif not jpeg_selected and suffix != ".png":
            suffix = ".png"
            target = str(Path(target).with_suffix(suffix))
        try:
            frame = self._decode_original_video_frame(
                media.path, frame_number, media.fps)
            if frame is None:
                raise RuntimeError("The selected frame could not be decoded.")
            encode_options = ([cv2.IMWRITE_JPEG_QUALITY, 100]
                              if suffix in {".jpg", ".jpeg"} else
                              [cv2.IMWRITE_PNG_COMPRESSION, 3])
            encoded, data = cv2.imencode(suffix, frame, encode_options)
            if not encoded:
                raise RuntimeError("The selected frame could not be encoded.")
            data.tofile(target)
        except Exception as error:
            QMessageBox.critical(self, "Could not save frame", str(error))
            return
        self.statusBar().showMessage(
            f"Frame {frame_number} saved · {Path(target).name}", 5000)

    def _set_manual_face_controls(self, active):
        for button in (self.face_previous_button, self.face_next_button,
                       self.face_reset_button, self.face_save_button,
                       self.remove_button):
            button.setVisible(not active)
        self.face_add_button.setVisible(active)
        self.face_cancel_button.setVisible(active)

    def open_manual_face_editor(self, point):
        media = self.current_media
        if (not media or self._current_media_is_processing() or
                media.mode != "faces" or not media.analyzed):
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
        self._sync_progress_metrics_visibility()
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
        self._refresh_media_metrics()
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
        self.face_save_button.setVisible(isinstance(event, FaceEvent))
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

    def save_selected_face(self):
        event, media = self.selected_cut, self.current_media
        if not isinstance(event, FaceEvent) or not media:
            return
        suggested = os.path.join(
            os.path.dirname(media.path),
            f"{Path(media.path).stem}_face_{event.best_frame:06d}.png")
        target, selected_filter = QFileDialog.getSaveFileName(
            self, "Save selected face", suggested,
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg)")
        if not target:
            return
        suffix = Path(target).suffix.lower()
        if not suffix:
            suffix = ".jpg" if selected_filter.startswith("JPEG") else ".png"
            target += suffix
        if suffix not in (".png", ".jpg", ".jpeg"):
            QMessageBox.warning(
                self, "Unsupported image format",
                "Choose PNG or JPEG for the selected face image.")
            return

        cap = cv2.VideoCapture(media.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(event.best_frame) - 1))
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None or len(event.crop_box or ()) != 4:
            QMessageBox.critical(
                self, "Could not save face",
                "The original video frame could not be read.")
            return
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(value) for value in event.crop_box)
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            QMessageBox.critical(
                self, "Could not save face",
                "The edited crop is outside the original video frame.")
            return
        params = ([int(cv2.IMWRITE_PNG_COMPRESSION), 1] if suffix == ".png"
                  else [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        extension = ".jpg" if suffix in (".jpg", ".jpeg") else ".png"
        encoded, data = cv2.imencode(extension, crop, params)
        try:
            if not encoded:
                raise OSError("image encoding failed")
            data.tofile(target)
        except OSError as error:
            QMessageBox.critical(
                self, "Could not save face", f"The image could not be saved:\n{error}")
            return
        self.statusBar().showMessage(f"Face saved · {target}", 6000)

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
        self.timeline.invalidate_face_thumbnail(event.crop_file)
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
        self._sync_progress_metrics_visibility()
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
                f"Confidence: {event.confidence:.2f} · Quality: {event.quality:.2f}\n"
                + ("Status: Included manually"
                   if event.enabled and not event.auto_included else
                   "Status: Included automatically" if event.enabled else
                   f"Status: Not included · "
                   f"{event.rejection_reason or 'Automatic filter'}"))
            self.selection_detail.setText(
                self.selection_detail.text() +
                f"\nFinal crop: {crop_width} x {crop_height} px" +
                (f"\nCandidate: {review_index + 1}/{len(review_frames)}"
                 if review_frames else ""))
            self.remove_button.setIcon(themed_icon(
                "fa6s.trash" if event.enabled else "fa6s.arrow-up"))
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
        if (self._current_media_is_processing() or not self.current_media or
                self.current_media.analyzed or
                not self.selected_segment):
            return
        self.remove_cut()

    def remove_cut(self):
        if self._current_media_is_processing():
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
            self._refresh_media_metrics()
            self.timeline.rebuild()
            self.show_segment(start_seconds, end_seconds, index)
            self._refresh_action_button()
            return
        if not self.selected_cut:
            return
        if isinstance(self.selected_cut, FaceEvent):
            event = self.selected_cut
            event.enabled = not event.enabled
            self._mark_current_unexported()
            self._refresh_media_metrics()
            hidden_by_filter = (
                (event.enabled and not self.timeline.show_saved_faces) or
                (not event.enabled and not self.timeline.show_detected_faces))
            if hidden_by_filter:
                self.clear_selection()
                self.timeline.rebuild()
                return
            self.timeline.rebuild()
            self.show_cut(event)
            return
        self.remove_cut_marker(self.selected_cut)

    def remove_cut_marker(self, event):
        """Remove one boundary and make the resulting merged segment active."""
        if (self._current_media_is_processing() or not self.current_media or
                not isinstance(event, CutEvent)):
            return
        ordered_cuts = sorted(self.current_media.cuts, key=lambda cut: cut.frame)
        boundary_index = next(
            (index for index, cut in enumerate(ordered_cuts)
             if cut.event_id == event.event_id), None)
        if boundary_index is None:
            return
        left_index = boundary_index + 1
        right_index = left_index + 1

        def merged_included_state(values):
            values = set(values)
            merged = [value for value in values if value < left_index]
            # Both source intervals become one explicitly included interval.
            merged.extend(value - 1 for value in values if value > right_index)
            return sorted(set(merged))

        self.current_media.excluded_segments = merged_included_state(
            self.current_media.excluded_segments)
        self.current_media.removed_segments = merged_included_state(
            self.current_media.removed_segments)
        self.current_media.cuts = [
            cut for cut in self.current_media.cuts
            if cut.event_id != event.event_id]
        self.current_media.normalize_cuts()
        self._mark_current_unexported()
        self.selected_cut = None
        self.selected_segment = None
        self.timeline.rebuild()
        self.show_cut(None)
        self._refresh_transport_timeline()
        self._refresh_media_metrics()
        self._refresh_action_button()
        self.statusBar().showMessage(
            "Marker removed · sequences merged and included.", 4000)

    def remove_face_permanently(self, event):
        if (self._current_media_is_processing() or not self.current_media or
                not isinstance(event, FaceEvent)):
            return
        before = len(self.current_media.faces)
        self.current_media.faces = [
            face for face in self.current_media.faces
            if face.event_id != event.event_id]
        if len(self.current_media.faces) == before:
            return
        self._mark_current_unexported()
        self._refresh_media_metrics()
        self.selected_cut = None
        self.timeline.rebuild()
        self.show_cut(None)

    def remove_segment_permanently(self, index):
        if self._current_media_is_processing() or not self.current_media:
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
        elif (isinstance(self.selected_cut, FaceEvent) and
              not face_event_is_included(self.current_media, self.selected_cut)):
            self.selected_cut = None
            self.show_cut(None)
        self.timeline.rebuild()
        self.statusBar().showMessage("Sequence removed from the project.", 4000)
        self._refresh_transport_timeline()
        self._refresh_media_metrics()
        self._refresh_action_button()

    @staticmethod
    def _media_has_processable_frames(media):
        if not media:
            return False
        unavailable = (set(media.excluded_segments) |
                       set(media.removed_segments))
        return any(
            end > start and index not in unavailable
            for index, (start, end) in enumerate(media.segments(), 1))

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
        boundaries, unavailable = mapping.rebuild_segments(analysis_boundaries)
        segments = list(zip(boundaries, boundaries[1:]))
        # Pre-analysis exclusions are editing instructions for the input, not
        # review items in the result.  Once analysis succeeds, every range that
        # was omitted from the proxy must disappear from the ripple timeline.
        # Users can still exclude newly detected result segments afterwards;
        # those remain visible/dimmed until export as before.
        omitted_ranges = [*excluded_ranges, *removed_ranges]
        removed = set(unavailable)
        removed.update(
            index for index, (start, end) in enumerate(segments, 1)
            if any(a <= start and end <= b for a, b in omitted_ranges))
        cuts = [CutEvent(frame, mode, original_frame=frame)
                for frame in boundaries[1:-1]]
        return cuts, [], sorted(removed)

    @staticmethod
    def _interval_analysis_boundaries(total_frames, fps, interval):
        fps = max(float(fps or 0), 0.001)
        duration = max(int(total_frames) / fps, 1.0)
        boundaries = []
        moment = float(interval)
        while moment < duration:
            frame = int(round(moment * fps))
            if 0 < frame < total_frames:
                boundaries.append(frame)
            moment += float(interval)
        return boundaries

    @staticmethod
    def _mapped_face_event(face, mapping, output_root, output_dir=None):
        def source_frame(value):
            mapped = mapping.to_source(max(0, int(value) - 1))
            return max(1, int(mapped or 0) + 1)

        review_frames = []
        for candidate in face.get("review_frames") or []:
            candidate = dict(candidate)
            candidate["frame"] = source_frame(candidate.get("frame", 0))
            review_frames.append(candidate)
        best_frame = source_frame(face.get("best_frame", 0))
        return FaceEvent(
            start_frame=source_frame(face.get("first_frame", 0)),
            end_frame=source_frame(face.get("last_frame", 0)),
            best_frame=best_frame,
            crop_file=os.path.join(
                output_dir or output_root, face.get("file", ""))
                if face.get("file") else "",
            confidence=float(face.get("confidence", 0)),
            quality=float(face.get("quality", 0)),
            box=list(face.get("box") or []),
            crop_box=list(face.get("crop_box") or []),
            review_frames=review_frames,
            original_best_frame=best_frame,
            original_box=list(face.get("box") or []),
            original_crop_box=list(face.get("crop_box") or []),
            enabled=bool(face.get("accepted", True)),
            rejection_reason=str(face.get("rejection_reason") or ""),
            auto_included=bool(face.get("accepted", True)),
        )

    @classmethod
    def _mapped_face_events(cls, engine, mapping, output_root):
        return [cls._mapped_face_event(
            face, mapping, output_root, engine._output_dir)
            for face in engine._face_candidates]

    def analyze(self):
        if not self.current_media or self.task: return
        if self.batch_mode:
            self.process_batch()
            return
        mode = self.mode.currentData()
        self.player.pause()
        media, interval = self.current_media, self.interval.value()
        media.mode, media.sensitivity = mode, self.sensitivity.currentText()
        media.interval = interval
        media.acceleration = self.accel.currentData() or "cpu"
        frame_map, excluded_ranges, removed_ranges = self._analysis_edit_plan(media)
        if not frame_map.ranges:
            self.statusBar().showMessage(
                "Include at least one sequence before starting analysis.", 5000)
            return
        self._set_progress_metrics(0, 0, mode, stage="detected")
        media.analyzed = False
        media.exported = False
        # Keep the edit plan intact while the worker runs.  Successful results
        # replace it atomically; cancellation or failure leaves it recoverable.
        self.selected_cut = None
        self._set_selection_available(False)
        self.timeline.set_media(media)
        if mode == "faces":
            self.timeline.begin_live_faces(media)
        else:
            self.timeline.begin_live_segments(media)
            if mode == "interval":
                preview_boundaries = self._interval_analysis_boundaries(
                    frame_map.analysis_frames, media.fps, interval)
                preview_cuts, _preview_excluded, preview_removed = (
                    self._mapped_timeline(
                        frame_map, preview_boundaries, mode,
                        excluded_ranges, removed_ranges))
                self.timeline.show_live_segments(
                    media, preview_cuts, preview_removed,
                    load_thumbnails=False)
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
                        reset_frames=frame_map.join_frames, keep_rejected=True,
                        face_callback=lambda face: task.face_found.emit(
                            media, self._mapped_face_event(
                                face, frame_map, cache, engine._output_dir)))
                    task.register_cancel(engine.stop)
                    if not engine.run():
                        raise RuntimeError("Face analysis stopped before producing results")
                    events = self._mapped_face_events(engine, frame_map, cache)
                    cuts, excluded, removed = final_timeline(frame_map.join_frames)
                    return ("faces", events, cuts, excluded, removed)
                cfg = PROFILES[media.sensitivity].copy()
                cfg.update(ENCODER=accel, INFERENCE="cpu", DEBUG=False)
                if mode == "interval": cfg["FIXED_INTERVAL"] = interval
                live_timeline = []
                def publish_scenes(scenes, finalized=True):
                    analysis_boundaries = [
                        start for start, _end in scenes if start > 0]
                    mapped = final_timeline(analysis_boundaries)
                    if finalized:
                        live_timeline[:] = mapped
                    task.segments_found.emit(
                        media, (mapped[0], mapped[2], finalized))
                engine = SceneEngine(analysis_path, cache, cfg,
                    LogAdapter(task), ProgressAdapter(task),
                    previewer=PreviewAdapter(task, frame_map, media.fps),
                    preview_enabled=True, preview_pool=[analysis_path],
                    export_results=False,
                    analysis_callback=publish_scenes,
                    preview_analysis_callback=(
                        (lambda scenes: publish_scenes(scenes, False))
                        if mode == "scene" else None))
                task.register_cancel(engine.stop)
                if not engine.run(scene_mode=mode == "scene"):
                    raise RuntimeError("Analysis stopped before producing results")
                if live_timeline:
                    cuts, excluded, removed = live_timeline
                else:
                    analysis_boundaries = [
                        start for start, _end in engine.analysis_scenes if start > 0]
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
            "Analyzing video", preview_media=media,
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
                    preview_enabled=True, reset_frames=mapping.join_frames,
                    keep_rejected=True,
                    face_callback=lambda face: task.face_found.emit(
                        media, self._mapped_face_event(
                            face, mapping, output, engine._output_dir)))
                task.register_cancel(engine.stop)
                if not engine.run():
                    raise RuntimeError("Face processing did not finish")
                events = self._mapped_face_events(engine, mapping, output)
                cuts, excluded, removed = self._mapped_timeline(
                    mapping, mapping.join_frames, mode,
                    excluded_ranges, removed_ranges)
                task.media_analyzed.emit(media)
                return media, "faces", events, cuts, excluded, removed

            cfg = PROFILES[sensitivity].copy()
            cfg.update(ENCODER=accel, INFERENCE="cpu", DEBUG=False)
            if mode == "interval":
                cfg["FIXED_INTERVAL"] = interval
            live_timeline = []
            def publish_scenes(scenes, finalized=True):
                analysis_boundaries = [
                    start for start, _end in scenes if start > 0]
                mapped = self._mapped_timeline(
                    mapping, analysis_boundaries, mode,
                    excluded_ranges, removed_ranges)
                if finalized:
                    live_timeline[:] = mapped
                task.segments_found.emit(
                    media, (mapped[0], mapped[2], finalized))
            engine = SceneEngine(
                analysis_path, output, cfg, LogAdapter(task), ProgressAdapter(task),
                previewer=previewer, preview_enabled=True,
                preview_pool=[analysis_path], export_results=False,
                analysis_callback=publish_scenes,
                preview_analysis_callback=(
                    (lambda scenes: publish_scenes(scenes, False))
                    if mode == "scene" else None))
            task.register_cancel(engine.stop)
            if not engine.run(scene_mode=mode == "scene"):
                raise RuntimeError("Video processing did not finish")
            if live_timeline:
                cuts, excluded, removed = live_timeline
            else:
                analysis_boundaries = [
                    start for start, _end in engine.analysis_scenes if start > 0]
                cuts, excluded, removed = self._mapped_timeline(
                    mapping, analysis_boundaries, mode,
                    excluded_ranges, removed_ranges)
            boundaries = [0, *(cut.frame for cut in cuts), media.total_frames]
            unavailable = set(excluded) | set(removed)
            export_segments = [segment for segment_index, segment in enumerate(
                zip(boundaries, boundaries[1:]), 1)
                if segment_index not in unavailable]
            task.media_analyzed.emit(media)
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
            output = QFileDialog.getExistingDirectory(
                self, "Choose queue output folder",
                output or (os.path.dirname(self.current_media.path)
                           if self.current_media else ""))
            if not output:
                self.statusBar().showMessage(
                    "Queue cancelled · no output folder selected", 5000)
                return
            self.project.output_folder = output
            self.output_button.setToolTip(f"Output folder: {output}")
        # Resume checkpoints without repeating items already exported.
        media_queue = [media for media in self.project.media if not media.exported]
        if not media_queue:
            self.statusBar().showMessage("Queue is already complete.", 4000)
            return
        empty_media = [media for media in media_queue
                       if not self._media_has_processable_frames(media)]
        if empty_media:
            self.statusBar().showMessage(
                "Restore at least one timeline sequence in every queued video.",
                6000)
            return
        mode = self.mode.currentData()
        sensitivity = self.sensitivity.currentText()
        interval = self.interval.value()
        accel = self.accel.currentData() or "cpu"
        self.player.pause()
        self._set_progress_metrics(0, 0, mode, stage="detected")
        self._batch_analyzed_media.clear()
        self._batch_completed_media.clear()
        self._batch_failed_media.clear()
        self._queue_total = len(media_queue)
        self._queue_media_ids = [media.media_id for media in media_queue]
        self._queue_completed = 1
        self._queue_errors = 0
        self._refresh_queue_metrics()
        for media in media_queue:
            media.mode, media.sensitivity = mode, sensitivity
            media.interval, media.acceleration = interval, accel
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
                    # Snapshot this item only when its turn begins, so edits
                    # made while an earlier queue item runs are respected.
                    edit_plan = self._analysis_edit_plan(media)
                    item_result = self._process_batch_item(
                        task, media, edit_plan, output, mode,
                        sensitivity, interval, accel)
                    completed.append(item_result)
                    task.media_finished.emit(media, True, item_result)
                except Exception as error:
                    if task.isInterruptionRequested():
                        break
                    failures.append((Path(media.path).name, str(error) or type(error).__name__))
                    task.media_finished.emit(media, False, None)
            return completed, failures

        self.batch_active = True
        self._refresh_queue_metrics()
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
            self._refresh_media_metrics()
        self._batch_completed_media.clear()
        self._refresh_media_item_states()
        self._refresh_action_button()
        self._update_analysis_controls_access()
        attempted = len(completed) + len(failures)
        summary = f"Queue complete · {len(completed)}/{attempted} videos processed"
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
        self._refresh_media_metrics()
        self._refresh_media_item_states()
        if self.current_media is media:
            self.timeline.rebuild()
            self._set_selection_available(True)
            self._refresh_action_button()
            self._update_analysis_controls_access()
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
        if not segments:
            self.statusBar().showMessage(
                "Include at least one sequence before exporting.", 5000)
            return
        accel = self.accel.currentData() or "cpu"
        self._refresh_media_metrics()
        self._set_progress_metrics(
            getattr(self, "_metric_detected", None), 0, media.mode,
            stage="cut")
        def work(task):
            if media.mode == "faces":
                destination = os.path.join(folder, f"{Path(media.path).stem}_faces_reviewed")
                os.makedirs(destination, exist_ok=True)
                saved = 0
                available_faces = [event for event in media.faces
                                   if face_event_is_included(media, event)]
                export_faces = [event for event in available_faces
                                if event.enabled and os.path.isfile(event.crop_file)]
                task.metrics.emit(len(available_faces), saved)
                total_faces = max(1, len(export_faces))
                preview_cap = None
                try:
                    for event in export_faces:
                        if task.isInterruptionRequested():
                            return None
                        preview = None
                        if getattr(task, "preview_visible", True):
                            try:
                                if preview_cap is None:
                                    preview_cap = cv2.VideoCapture(media.path)
                                preview_cap.set(
                                    cv2.CAP_PROP_POS_FRAMES, event.best_frame)
                                ok, frame = preview_cap.read()
                                if ok and frame is not None:
                                    if len(event.box or []) == 4:
                                        x1, y1, x2, y2 = map(int, event.box)
                                        cv2.rectangle(
                                            frame, (x1, y1), (x2, y2),
                                            (0, 255, 0), 2)
                                    preview = resize_for_preview(Image.fromarray(
                                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                                if preview is None:
                                    with Image.open(event.crop_file) as source:
                                        preview = resize_for_preview(
                                            source.convert("RGB"))
                                if preview:
                                    task.preview.emit(
                                        preview,
                                        media.seconds_for_frame(event.best_frame))
                            except Exception:
                                pass
                        target = os.path.join(
                            destination, f"face_{saved + 1:04d}.png")
                        __import__("shutil").copy2(event.crop_file, target)
                        saved += 1
                        task.metrics.emit(len(available_faces), saved)
                        task.progress.emit(
                            saved / total_faces, "Exporting faces")
                finally:
                    if preview_cap is not None:
                        preview_cap.release()
                if not export_faces:
                    task.progress.emit(1.0, "Exporting faces")
                with open(os.path.join(destination, "faces.json"), "w", encoding="utf-8") as handle:
                    json.dump({"source": media.path,
                               "faces": [vars(event) for event in available_faces],
                               "faces_exported": saved}, handle, indent=2)
                return destination
            cfg = PROFILES[media.sensitivity].copy()
            cfg.update(ENCODER=accel, INFERENCE="cpu", DEBUG=False)
            engine = SceneEngine(
                media.path, folder, cfg, LogAdapter(task), ProgressAdapter(task),
                previewer=PreviewAdapter(task), preview_enabled=True,
                preview_frames=[
                    frame
                    for start, end in segments
                    for frame in (start, max(start, end - 1))])
            task.register_cancel(engine.stop)
            engine.scene_mode = media.mode == "scene"
            engine.detected, engine._start_time = len(segments), time.time()
            engine.start_preview()
            try:
                engine._cut_scenes(segments)
            finally:
                engine.finish_preview(
                    show_final=not task.isInterruptionRequested())
            return engine._cut_output_dir
        self.start_task(
            work, lambda output, target=media: self.export_finished(target, output),
            "Exporting", preview_media=media, minimum_preview_ms=800)

    def export_finished(self, media, output):
        media.exported = True
        if self.current_media is media:
            self._refresh_media_metrics()
        self._refresh_media_item_states()
        self._refresh_action_button()
        self._update_analysis_controls_access()
        confirmation = QMessageBox(self)
        complete_icon = qta.icon(
            "fa6s.circle-check", color="#38c982", scale_factor=.92)
        confirmation.setIconPixmap(complete_icon.pixmap(QSize(38, 38)))
        confirmation.setWindowTitle("Export complete")
        confirmation.setText("Files were exported successfully.")
        output_path = os.path.abspath(output)
        display_path = output_path.replace("\\", "\\\u200b").replace("/", "/\u200b")
        confirmation.setInformativeText(display_path)
        path_label = confirmation.findChild(
            QLabel, "qt_msgbox_informativelabel")
        if path_label is not None:
            path_label.setWordWrap(True)
            path_label.setMaximumWidth(520)
            path_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
        confirmation.setStandardButtons(QMessageBox.StandardButton.Ok)
        ok_button = confirmation.button(QMessageBox.StandardButton.Ok)
        ok_button.setStyleSheet("""
            QPushButton {
                color:white;
                background:qlineargradient(
                    x1:0,y1:0,x2:0,y2:1,
                    stop:0 #38c982, stop:1 #188a51);
                border:1px solid #52d99a;
                border-radius:8px;
            }
            QPushButton:hover {
                background:qlineargradient(
                    x1:0,y1:0,x2:0,y2:1,
                    stop:0 #4dda95, stop:1 #20a463);
                border-color:#7be8b5;
            }
            QPushButton:pressed {
                background:#147344;
                border-color:#38c982;
            }
        """)
        show_button = confirmation.addButton(
            "Show in Folder", QMessageBox.ButtonRole.ActionRole)
        confirmation.setDefaultButton(QMessageBox.StandardButton.Ok)
        confirmation.exec()
        if confirmation.clickedButton() is show_button:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(output_path))

    def start_task(self, work, callback, label, preview_media=None,
                   done_label="Complete", minimum_preview_ms=0):
        if self.task:
            return
        self._end_edit_preview()
        self._release_frame_decoder()
        self._set_working_ui(True)
        self._set_progress_value(0, immediate=True)
        self.progress_label.setText(label)
        self._refresh_progress_state_icon()
        task = Task(work, self, done_label=done_label)
        task.minimum_preview_ms = max(0, int(minimum_preview_ms))
        self.task = task
        task.progress.connect(self.task_progress)
        task.metrics.connect(self.task_metrics)
        task.preview.connect(self._show_processing_frame)
        task.face_found.connect(self._live_face_found)
        task.segments_found.connect(self._live_segments_found)
        task.media_started.connect(self.batch_media_started)
        task.media_analyzed.connect(self.batch_media_analyzed)
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
            confirmation = QMessageBox(self)
            stop_icon = qta.icon(
                "fa6s.circle-xmark", color="#d25a65", scale_factor=.92)
            confirmation.setIconPixmap(stop_icon.pixmap(QSize(38, 38)))
            confirmation.setWindowTitle("Stop current process")
            confirmation.setText(
                "Stop the current analysis/export?\n"
                "Partial results will be discarded.")
            confirmation.setStandardButtons(
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No)
            confirmation.setDefaultButton(QMessageBox.StandardButton.No)
            yes_button = confirmation.button(QMessageBox.StandardButton.Yes)
            yes_button.setObjectName("danger")
            yes_button.setStyleSheet("""
                QPushButton {
                    color:white;
                    background:qlineargradient(
                        x1:0,y1:0,x2:0,y2:1,
                        stop:0 #b84b55, stop:1 #7f2932);
                    border:1px solid #d26a73;
                    border-radius:8px;
                }
                QPushButton:hover {
                    background:qlineargradient(
                        x1:0,y1:0,x2:0,y2:1,
                        stop:0 #d25a65, stop:1 #97343e);
                    border-color:#ef8991;
                }
                QPushButton:pressed {
                    background:#7f2932;
                    border-color:#d26a73;
                }
            """)
            confirmation.exec()
            if confirmation.clickedButton() is not yes_button:
                return False
        self.progress_label.setText("Stopping")
        self._refresh_progress_state_icon()
        self.action_button.setEnabled(False)
        task.cancel()
        return True

    def _refresh_action_button(self):
        if not hasattr(self, "action_button"):
            return
        if self.task:
            if self._current_media_is_processing():
                label, button_style, enabled = "Stop", "danger", True
                tooltip = "Stop current process"
            else:
                label = ("Export" if self.current_media and
                         self.current_media.analyzed else "Start")
                button_style, enabled = (
                    "warning" if label == "Export" else "primary"), False
                tooltip = "Another video is currently being processed."
            self.action_button.setText(label)
            self.action_button.setEnabled(enabled)
            self.action_button.setToolTip(tooltip)
            self.action_button.setObjectName(button_style)
            self.action_button.style().unpolish(self.action_button)
            self.action_button.style().polish(self.action_button)
            return
        disabled_reason = ""
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
            has_content = bool(self.project.media) and all(
                self._media_has_processable_frames(media)
                for media in self.project.media)
            enabled = has_content
            if not has_content and self.project.media:
                disabled_reason = (
                    "Restore at least one timeline sequence in every queued video.")
            elif not (self.project.output_folder and
                      os.path.isdir(self.project.output_folder)):
                disabled_reason = "Choose the queue output folder when starting."
            button_style = "primary"
        elif self.current_media and self.current_media.analyzed:
            label = "Export"
            enabled = self._media_has_processable_frames(self.current_media)
            if not enabled:
                disabled_reason = (
                    "Restore at least one timeline sequence before exporting.")
            button_style = "warning"
        else:
            label = "Start"
            enabled = self._media_has_processable_frames(self.current_media)
            if self.current_media and not enabled:
                disabled_reason = (
                    "Restore at least one timeline sequence before starting.")
            button_style = "primary"
        self.action_button.setText(label)
        self.action_button.setEnabled(enabled)
        self.action_button.setToolTip(
            "Results exported." if button_style == "success" else
            disabled_reason)
        self.action_button.setObjectName(button_style)
        self.action_button.style().unpolish(self.action_button)
        self.action_button.style().polish(self.action_button)

    def _set_working_ui(self, working):
        # Keep scrolling available so long filenames remain inspectable.  In
        # auto-process mode _select_media blocks selection changes instead.
        self.media_list.setEnabled(True)
        self.import_button.setEnabled(not working)
        self.output_button.setEnabled(not working)
        self.auto_process.setEnabled(not working)
        self.clear_media_button.setEnabled(not working)
        self.import_action.setEnabled(not working)
        self.load_action.setEnabled(not working)
        self.add_cut_action.setEnabled(not working)
        # Processing preview is read-only in every analysis mode.  Keep the
        # visible scissors control dimmed just like the adjacent trash button;
        # add_cut() also retains its task guard as a second line of defense.
        if working:
            self.preanalysis_cut_button.setEnabled(False)
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
        was_batch = self.batch_active
        self.batch_active = False
        if was_batch:
            self.queue_metrics.clear()
            self.queue_metrics.hide()
        self._mark_processing_media(None)
        self._end_processing_preview()
        self._set_working_ui(False)
        self.progress_label.setText(label)
        self._refresh_progress_state_icon()

    def task_progress(self, value, label):
        media = self.processing_media
        if media and self.task:
            previous_value, previous_label = self.task.media_progress.get(
                media.media_id, (0, "Processing"))
            stored_value = (round(value * 1000) if value >= 0 else
                            previous_value)
            self.task.media_progress[media.media_id] = (
                stored_value, label or previous_label)
        if value >= 0 and self._current_media_is_processing():
            self._set_progress_value(round(value * 1000))
        if (value >= 0 and self._processing_preview_selected() and self.task and
                not self.task.preview_paused):
            media = getattr(self, "_processing_preview_media", None)
            if media:
                timeline_target = float(value) * self._timeline_duration(media)
                self._animate_processing_position(
                    self._timeline_to_source_seconds(timeline_target, media))
        if label and self._current_media_is_processing():
            self.progress_label.setText(label)
            self._refresh_progress_state_icon()

    def _set_progress_metrics(self, detected=None, cut=None, mode=None,
                              stage=None):
        mode = mode or (self.current_media.mode if self.current_media else
                        self.mode.currentData())
        self._metric_mode = mode
        self._metric_detected = detected
        self._metric_cut = cut
        self._metric_stage = stage
        detected_label, cut_label = status_labels(mode)
        if stage == "detected":
            value = 0 if detected is None else max(0, int(detected))
            if mode == "faces":
                selected = 0 if cut is None else max(0, int(cut))
                self.face_detected_metric.setText(f"{detected_label}: {value}")
                self.face_saved_metric.setText(f"{cut_label}: {selected}")
                self.progress_metrics.clear()
            else:
                self.progress_metrics.setText(f"{detected_label}: {value}")
        elif stage == "cut":
            value = 0 if cut is None else max(0, int(cut))
            if mode == "faces":
                detected_value = 0 if detected is None else max(0, int(detected))
                self.face_detected_metric.setText(
                    f"{detected_label}: {detected_value}")
                self.face_saved_metric.setText(f"{cut_label}: {value}")
                self.progress_metrics.clear()
            else:
                self.progress_metrics.setText(f"{cut_label}: {value}")
        else:
            self.progress_metrics.clear()
            self.face_detected_metric.clear()
            self.face_saved_metric.clear()
        self._sync_progress_metrics_visibility()

    def _sync_progress_metrics_visibility(self):
        if not hasattr(self, "progress_metrics"):
            return
        has_selection = bool(self.selected_cut is not None or
                             self.selected_segment is not None or
                             self._manual_face_draft is not None)
        if hasattr(self, "selection_close_button"):
            self.selection_close_button.setVisible(has_selection)
        if hasattr(self, "face_detected_filter_button"):
            face_results = bool(
                self.current_media and self.current_media.analyzed and
                self.current_media.mode == "faces" and
                not self._current_media_is_processing())
            show_face_metrics = bool(
                getattr(self, "_metric_stage", None) is not None and
                getattr(self, "_metric_mode", None) == "faces" and
                not has_selection)
            self.face_metrics.setVisible(show_face_metrics)
            self.face_detected_filter_button.setVisible(
                show_face_metrics and face_results)
            self.face_saved_filter_button.setVisible(
                show_face_metrics and face_results)
            if face_results and self._manual_face_draft is None:
                for button in (self.face_previous_button,
                               self.face_next_button,
                               self.face_reset_button,
                               self.remove_button):
                    button.setVisible(has_selection)
        self.progress_metrics.setVisible(
            getattr(self, "_metric_stage", None) is not None and
            getattr(self, "_metric_mode", None) != "faces" and
            not has_selection)

    def _refresh_media_metrics(self):
        media = self.current_media
        if not media or not media.analyzed:
            self._set_progress_metrics(
                mode=(media.mode if media else None), stage=None)
            return
        unavailable = set(media.excluded_segments) | set(media.removed_segments)
        detected_segments = sum(
            1 for index, _segment in enumerate(media.segments(), 1)
            if index not in set(media.removed_segments))
        available_segments = sum(
            1 for index, _segment in enumerate(media.segments(), 1)
            if index not in unavailable)
        if media.mode == "faces":
            available_faces = [event for event in media.faces
                               if face_event_is_included(media, event)]
            detected = len(available_faces)
            saved = sum(1 for event in available_faces if event.enabled)
            self._set_progress_metrics(
                detected, saved, media.mode,
                stage="cut" if media.exported else "detected")
        else:
            self._set_progress_metrics(
                detected_segments, available_segments if media.exported else 0,
                media.mode, stage="cut" if media.exported else "detected")

    def task_metrics(self, detected, cut):
        if self.task and not self._current_media_is_processing():
            return
        if detected is None:
            detected = getattr(self, "_metric_detected", None)
        if cut is None:
            cut = getattr(self, "_metric_cut", None)
        stage = getattr(self, "_metric_stage", None)
        if (self.batch_active and getattr(self, "_metric_mode", None) != "faces"
                and cut is not None and int(cut) > 0):
            stage = "cut"
        self._set_progress_metrics(
            detected, cut, getattr(self, "_metric_mode", None), stage=stage)

    def _live_face_found(self, media, event):
        """Receive finalized face tracks on the UI thread."""
        task = self.task
        if not task or not media or not isinstance(event, FaceEvent):
            return
        events = task.live_faces.setdefault(media.media_id, [])
        if any(existing.event_id == event.event_id for existing in events):
            return
        events.append(event)
        if self.current_media is media:
            self.timeline.append_live_face(media, event)

    def _live_segments_found(self, media, result):
        """Receive the definitive cut layout without committing project data."""
        task = self.task
        if not task or not media or not isinstance(result, tuple):
            return
        cuts, removed, finalized = result
        task.live_segments[media.media_id] = (
            list(cuts), list(removed), bool(finalized))
        if self.current_media is media:
            self.timeline.show_live_segments(
                media, cuts, removed, load_thumbnails=bool(finalized))

    def _restore_live_results_for_current(self):
        task = self.task
        media = self.current_media
        if not task or not media:
            return
        if media.mode == "faces":
            events = task.live_faces.get(media.media_id)
            if events is None:
                return
            self.timeline.begin_live_faces(media)
            for event in events:
                self.timeline.append_live_face(media, event)
            return
        result = task.live_segments.get(media.media_id)
        if result is None:
            return
        cuts, removed, finalized = result
        if cuts is None:
            self.timeline.begin_live_segments(media)
            return
        self.timeline.show_live_segments(
            media, cuts, removed, load_thumbnails=finalized)

    def _set_progress_value(self, value, immediate=False):
        target = max(self.progress.minimum(), min(self.progress.maximum(), int(value)))
        current = self.progress.value()
        self.progress_animation.stop()
        if immediate or target <= current or target == current:
            self.progress.setValue(target)
            return
        distance = target - current
        self.progress_animation.setDuration(
            max(140, min(550, round(130 + distance * 0.8))))
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
        if self.task and media.mode == "faces":
            self.task.live_faces.setdefault(media.media_id, [])
        elif self.task:
            self.task.live_segments.setdefault(
                media.media_id, (None, None, False))
        if self.task:
            self.task.media_progress[media.media_id] = (0, "Processing")
        try:
            self._queue_completed = self._queue_media_ids.index(media.media_id) + 1
        except ValueError:
            pass
        self._refresh_queue_metrics()
        self._mark_processing_media(media)
        try:
            row = next(index for index, item in enumerate(self.project.media) if item is media)
        except StopIteration:
            return
        if self.media_list.currentRow() != row:
            self.media_list.setCurrentRow(row)
        else:
            # setCurrentRow does not emit when the row is unchanged.
            self._restore_live_results_for_current()
            self._restore_selected_media_progress()
            self._update_transport_access()
        if getattr(self, "_processing_preview_media", None) is not media:
            self._begin_processing_preview(media)

    def batch_media_finished(self, media, succeeded, result=None):
        if succeeded and result:
            _media, kind, events, cuts, excluded, removed = result
            media.analyzed = True
            media.exported = True
            media.cuts = list(cuts)
            media.normalize_cuts()
            media.excluded_segments = list(excluded)
            media.removed_segments = list(removed)
            if kind == "faces":
                media.faces = list(events)
        if succeeded:
            self._batch_analyzed_media.discard(media.media_id)
            self._batch_failed_media.discard(media.media_id)
            self._batch_completed_media.add(media.media_id)
        else:
            # Preserve whether the item stopped before (white) or after
            # analysis (yellow); failures must never enter the green set.
            self._batch_failed_media.add(media.media_id)
            self._batch_completed_media.discard(media.media_id)
        if not succeeded:
            self._queue_errors = getattr(self, "_queue_errors", 0) + 1
        self._refresh_queue_metrics()
        self._refresh_media_item_states()

    def _refresh_queue_metrics(self):
        if not hasattr(self, "queue_metrics"):
            return
        total = max(0, int(getattr(self, "_queue_total", 0)))
        completed = max(0, min(
            total, int(getattr(self, "_queue_completed", 0))))
        errors = max(0, int(getattr(self, "_queue_errors", 0)))
        text = f"Queue: {completed}/{total}"
        if errors:
            text += f" · Errors: {errors}"
        self.queue_metrics.setText(text)
        self.queue_metrics.setVisible(bool(self.batch_active and total))

    def batch_media_analyzed(self, media):
        if not self.batch_active or not media:
            return
        self._batch_analyzed_media.add(media.media_id)
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
            if ((item_media is self.processing_media or
                 item_media.media_id in self._batch_failed_media) and
                    not item_media.analyzed and
                    item_media.media_id not in self._batch_analyzed_media):
                item.setIcon(qta.icon(
                    "fa6s.circle-arrow-right", color="#ffffff",
                    scale_factor=.9))
                item.setToolTip(f"Processing now\n{item_media.path}")
                item.setForeground(QColor("#ffffff"))
            elif item_media.exported or item_media.media_id in self._batch_completed_media:
                item.setIcon(qta.icon(
                    "fa6s.circle-check", color="#38c982",
                    scale_factor=.9))
                item.setToolTip(f"Exported\n{item_media.path}")
                item.setForeground(QColor("#b8e8cf"))
            elif (item_media.analyzed or
                  item_media.media_id in self._batch_analyzed_media):
                item.setIcon(qta.icon(
                    "fa6s.circle-info", color="#f2c94c",
                    scale_factor=.9))
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
        if (media and media.exported and status == "Complete" and
                not self._current_media_is_processing()):
            icon = qta.icon(
                "fa6s.circle-check", color="#38c982", scale_factor=.9)
            self.progress_state_icon.setPixmap(icon.pixmap(QSize(13, 13)))
            self.progress_state_icon.setToolTip(f"Exported\n{media.path}")
            self.progress_state_icon.show()
        elif (media and media.analyzed and not media.exported and
              status == "Video analyzed" and
              not self._current_media_is_processing()):
            icon = qta.icon(
                "fa6s.circle-info", color="#f2c94c", scale_factor=.9)
            self.progress_state_icon.setPixmap(icon.pixmap(QSize(13, 13)))
            self.progress_state_icon.setToolTip(
                f"Analyzed · ready to export\n{media.path}")
            self.progress_state_icon.show()
        else:
            self.progress_state_icon.clear()
            self.progress_state_icon.setToolTip("")
            self.progress_state_icon.hide()

    def task_done(self, task, callback, result):
        def finish():
            finished_media = self.processing_media
            self._finish_task_ui(task, task.done_label)
            self._set_progress_value(1000)
            callback(result)
            if self.current_media is not finished_media:
                self._restore_selected_media_progress()
            task.deleteLater()

        minimum_ms = getattr(task, "minimum_preview_ms", 0)
        started = (getattr(self, "_processing_preview_first_frame_at", None) or
                   getattr(self, "_processing_preview_started_at", None))
        elapsed_ms = ((time.monotonic() - started) * 1000
                      if started is not None else minimum_ms)
        remaining_ms = max(0, round(minimum_ms - elapsed_ms))
        if remaining_ms:
            QTimer.singleShot(remaining_ms, finish)
        else:
            finish()

    def task_failed(self, task, message):
        self._batch_analyzed_media.clear()
        self._batch_completed_media.clear()
        failed_media = self.processing_media
        self._finish_task_ui(task, "Failed")
        if self.current_media:
            self.timeline.rebuild()
        self._refresh_media_metrics()
        if self.current_media is not failed_media:
            self._restore_selected_media_progress()
        QMessageBox.critical(self, "Scenespy error", message)
        task.deleteLater()

    def task_cancelled(self, task):
        self._batch_analyzed_media.clear()
        self._batch_completed_media.clear()
        cancelled_media = self.processing_media
        self._finish_task_ui(task, "Stopped")
        if self.current_media:
            self.timeline.rebuild()
        self._refresh_media_metrics()
        self._set_progress_value(0, immediate=True)
        if self.current_media is not cancelled_media:
            self._restore_selected_media_progress()
        self.statusBar().showMessage("Process stopped. Partial results were discarded.", 6000)
        task.deleteLater()

    def save_project(self):
        if not self.current_media:
            return
        if self.task and not self.batch_active:
            return
        if self.batch_mode or self.batch_active:
            saved_media = copy.deepcopy(self.project.media)
            if self.batch_active:
                for media in saved_media:
                    if not media.exported:
                        # A running/failed/unstarted item is a pending job in
                        # the checkpoint, never a partial analysis result.
                        media.analyzed = False
                        media.exported = False
                        if media.mode == "faces":
                            media.faces = []
            selected_project = ScenespyProject(
                media=saved_media,
                output_folder=self.project.output_folder,
                queue_mode=True)
            suggested = str(
                Path(self.current_media.path).with_name("Scenespy Queue.scenespy"))
            dialog_title = "Save Scenespy queue"
        else:
            media = self.current_media
            media.mode = self.mode.currentData()
            media.sensitivity = self.sensitivity.currentText()
            media.interval = self.interval.value()
            media.acceleration = self.accel.currentData() or "cpu"
            selected_project = ScenespyProject(
                media=[copy.deepcopy(media)],
                output_folder=self.project.output_folder)
            suggested = str(Path(media.path).with_suffix(".scenespy"))
            dialog_title = "Save selected Scenespy project"
        path, _ = QFileDialog.getSaveFileName(
            self, dialog_title, suggested,
            "Scenespy Project (*.scenespy)")
        if not path:
            return
        saved_path = selected_project.save(path)
        self.statusBar().showMessage(
            f"Project saved · {Path(saved_path).name}", 5000)

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
        if not project.media:
            QMessageBox.warning(
                self, "Empty project", "This project does not contain any media.")
            return
        existing_ids = {media.media_id for media in self.project.media}
        first_row = len(self.project.media)
        for media in project.media:
            if media.media_id in existing_ids:
                media.media_id = uuid.uuid4().hex
            existing_ids.add(media.media_id)
            self.project.media.append(media)
            item = QListWidgetItem(Path(media.path).name)
            item.setToolTip(media.path)
            self.media_list.addItem(item)
        if not self.project.output_folder and project.output_folder:
            self.project.output_folder = project.output_folder
            self.output_button.setToolTip(
                f"Output folder: {project.output_folder}")
        self._refresh_media_item_states()
        self._update_batch_availability()
        self.media_list.setCurrentRow(first_row)
        if project.queue_mode and self.media_list.count() >= 2:
            self.auto_process.setChecked(True)
        self.statusBar().showMessage(
            f"Project imported · {len(project.media)} media item(s)", 5000)

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
        proxy_tasks = list(self._playback_proxy_tasks.values())
        for proxy_task in proxy_tasks:
            if proxy_task.isRunning():
                proxy_task.cancel()
        if proxy_tasks:
            terminate_all_child_processes(grace_seconds=.2)
            for proxy_task in proxy_tasks:
                proxy_task.wait(2000)
        self._release_frame_decoder()
        self.player.stop()
        terminate_all_child_processes(grace_seconds=.2)
        self._save_workspace_splitter()
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
