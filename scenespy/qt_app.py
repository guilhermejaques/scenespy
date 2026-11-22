"""Production-oriented Qt workspace for non-destructive Scenespy projects."""

from __future__ import annotations

import json
import copy
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import av
import cv2
import qtawesome as qta

from PySide6.QtCore import (QEasingCurve, QEvent, QPoint, QPointF, QProcess, QProcessEnvironment, QPropertyAnimation,
                            QRect, QRectF, QSettings, QSize, Qt, QThread, QTimer, QUrl, QVariantAnimation,
                            Signal)
from PySide6.QtGui import (QAction, QBrush, QColor, QDesktopServices, QFont, QFontDatabase,
                           QFontMetrics,
                           QIcon, QImage, QKeySequence, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGraphicsItem, QGraphicsLineItem,
    QGraphicsOpacityEffect,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
    QScrollBar, QSizePolicy, QSpinBox, QSplitter, QStackedLayout, QStatusBar, QStyle,
    QStyleOptionFrame,
    QStyleOptionSlider, QToolButton, QVBoxLayout, QWidget, QWidgetAction,
)

from .media import probe_media
from . import __version__
from . import shared as runtime_shared
from .project import CutEvent, FaceEvent, MediaItem, ScenespyProject
from .face_engine import FaceDetectionEngine, draw_face_marker
from .frame_map import FrameMap, create_analysis_proxy, create_playback_proxy
from .scene_engine import SceneEngine
from .shared import (ACCEL_OPTIONS, ACCENT, ALLOWED_VIDEO_EXTENSIONS, BG_CARD, BG_MAIN,
                     BG_PANEL, BORDER_SOFT2, CACHE_PROFILE, DANGER, MODE_ACCEL_COMPAT,
                     PROFILES, SUCCESS,
                     TEXT_MAIN, TEXT_MUTED,
                     USER_DATA_DIR, detect_available_encoder_accel, detect_available_inference_accel,
                     next_output_path,
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
FACE_THUMBNAIL_SIZE = QSize(72, 42)
TIMELINE_INDICATOR_WIDTH = 40
TIMELINE_INDICATOR_HEIGHT = 25
FACE_INDICATOR_LANE_STEP = 27
INDICATOR_PLAY_POINTS = (
    QPointF(-2.5, -4), QPointF(4, 0), QPointF(-2.5, 4))
QUEUE_RESULT_PREVIEW_MIN_MS = 900
UPDATE_API_URL = "https://api.github.com/repos/guilhermejaques/scenespy/releases/latest"
UPDATE_ALLOWED_HOSTS = {
    "api.github.com",
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
}


def version_tuple(value):
    numbers = [int(part) for part in re.findall(r"\d+", str(value))[:3]]
    return tuple((numbers + [0, 0, 0])[:3])


def update_asset_suffix(system=None, machine=None):
    system = (system or sys.platform).lower()
    machine = (machine or platform.machine()).lower()
    if system.startswith("win"):
        return "-windows-x64-setup.exe"
    if system.startswith("linux"):
        return "-linux-amd64.deb"
    if system == "darwin":
        arch = "arm64" if machine in {"arm64", "aarch64"} else "x64"
        return f"-macos-{arch}.pkg"
    return ""


def _trusted_update_url(url):
    parsed = urllib.parse.urlparse(str(url))
    return parsed.scheme == "https" and parsed.hostname in UPDATE_ALLOWED_HOSTS


def place_timeline_indicator_lane(left, occupied_right_edges):
    """Place a fixed-width indicator in the first non-conflicting lane."""
    right = float(left) + TIMELINE_INDICATOR_WIDTH
    lane = next(
        (index for index, occupied_right in enumerate(occupied_right_edges)
         if float(left) >= occupied_right - 4),
        len(occupied_right_edges),
    )
    if lane == len(occupied_right_edges):
        occupied_right_edges.append(right)
    else:
        occupied_right_edges[lane] = right
    return lane, right


def valid_faces_output_name(name):
    parts = str(name).split("_")
    if len(parts) not in (5, 6) or parts[0] != "FD":
        return False
    if not (len(parts[1]) == 8 and parts[1].isdigit() and
            len(parts[2]) == 6 and parts[2].isdigit()):
        return False
    return len(parts) == 5 or parts[-1].isdigit()


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


def format_file_size(byte_count):
    """Format an exact byte count using a bounded, correctly scaled unit."""
    try:
        size = max(0, int(byte_count))
    except (TypeError, ValueError, OverflowError):
        return ""
    units = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    value = float(size)
    unit_index = 0
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{size} B"
    decimals = 0 if value >= 100 else 1 if value >= 10 else 2
    if (round(value, decimals) >= 1024 and
            unit_index < len(units) - 1):
        value /= 1024
        unit_index += 1
        decimals = 0 if value >= 100 else 1 if value >= 10 else 2
    number = f"{value:.{decimals}f}"
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return f"{number} {units[unit_index]}"


def exported_path_size(path):
    """Return bytes actually stored by one exported file or directory."""
    if not path:
        return 0
    try:
        if os.path.isfile(path):
            return max(0, int(os.path.getsize(path)))
    except OSError:
        return 0
    total = 0
    try:
        for root, _directories, files in os.walk(path, followlinks=False):
            for filename in files:
                candidate = os.path.join(root, filename)
                try:
                    if not os.path.islink(candidate):
                        total += max(0, int(os.path.getsize(candidate)))
                except OSError:
                    continue
    except OSError:
        return total
    return total


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


def next_face_edit_path(previous_path, event_id):
    """Return a fixed-length unique crop path, independent of edit depth."""
    previous = Path(previous_path)
    suffix = previous.suffix or ".png"
    return previous.with_name(
        f"face_edit_{str(event_id)[:32]}_{uuid.uuid4().hex}{suffix}")


def clip_preview_frame(start_frame, _end_frame=None):
    """Use the exact first source frame as a clip's visual identity."""
    return max(0, int(start_frame))


def clip_preview_candidates(start_frame, end_frame, max_attempts=30):
    """Return the first frames to try without crossing the clip boundary."""
    start = max(0, int(start_frame))
    end = max(start, int(end_frame))
    return range(start, min(end, start + max(1, int(max_attempts))))


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


STATUS_ICON_SIZE = QSize(20, 20)
STATUS_ICON_SCALE = .648  # Same glyph size as .72 in an 18 px canvas.
STATUS_PROCESS_COLOR = "#ffffff"
STATUS_PENDING_COLOR = "#f2c94c"
STATUS_SUCCESS_COLOR = "#38c982"
STATUS_ERROR_COLOR = "#d25a65"


def status_indicator_icon(name, color):
    """Return a status glyph optically centered beside its text label."""
    # Font Awesome's circular glyphs sit slightly low in their square canvas.
    # QtAwesome applies this offset while painting the vector glyph, retaining
    # its native antialiasing and its colors in every QIcon state.
    return qta.icon(
        name,
        color=color,
        scale_factor=STATUS_ICON_SCALE,
        offset=(0, -1 / STATUS_ICON_SIZE.height()),
    )


def remove_owned_directory(path, parent):
    """Remove a generated directory only when it is below its expected root."""
    if not path or not parent:
        return False
    root = os.path.abspath(parent)
    target = os.path.abspath(path)
    try:
        contained = os.path.commonpath((root, target)) == root
    except ValueError:
        contained = False
    if not contained or target == root:
        return False
    for _attempt in range(5):
        if not os.path.exists(target):
            return True
        try:
            shutil.rmtree(target)
        except OSError:
            time.sleep(.05)
    return not os.path.exists(target)


class ScrubSlider(QSlider):
    """Seek slider whose entire groove is directly draggable."""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.show_milliseconds = True
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
        formatter = player_timecode if self.show_milliseconds else transport_timecode
        text = formatter(value / 1000, duration)
        self._time_bubble.setText(text)
        sample = formatter(duration, duration)
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
        if (self.property("uniformTimeText") or self.hasFocus() or
                len(text) < 5 or text[-4] != "."):
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
            option, self)
        main_text, millis_text = text[:-4], text[-4:]
        metrics = self.fontMetrics()
        main_width = metrics.horizontalAdvance(main_text)
        millis_width = metrics.horizontalAdvance(millis_text)
        total_width = main_width + millis_width
        if self.alignment() & Qt.AlignmentFlag.AlignLeft:
            x = contents.left()
        elif self.alignment() & Qt.AlignmentFlag.AlignRight:
            x = contents.right() - total_width
        else:
            x = contents.center().x() - total_width / 2
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
    """Four native marker edges stacked above the native video surface."""
    def __init__(self, host, parent=None):
        super().__init__(parent or host)
        self._host = host
        self._box = None
        self._source_size = QSize()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()
        edge_parent = parent or host
        self._edges = [QWidget(edge_parent) for _ in range(4)]
        self._shadows = [QWidget(edge_parent) for _ in range(4)]
        for marker in (*self._shadows, *self._edges):
            marker.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            # QVideoWidget can own a native rendering surface on Windows.
            # Native sibling edges participate in the same window Z-order and
            # cannot be swallowed by that surface like a transparent child.
            marker.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
            marker.hide()
        for shadow in self._shadows:
            shadow.setStyleSheet("background:#000000; border:0;")
        for edge in self._edges:
            edge.setStyleSheet("background:#00ff00; border:0;")

    def set_face(self, box, source_width, source_height, color="#00ff00"):
        if len(box or ()) != 4 or source_width <= 0 or source_height <= 0:
            self.clear_face()
            return
        for edge in self._edges:
            edge.setStyleSheet(f"background:{color}; border:0;")
        self._box = tuple(float(value) for value in box)
        self._source_size = QSize(int(source_width), int(source_height))
        self.refresh_geometry()

    def clear_face(self):
        self._box = None
        for marker in (*self._edges, *self._shadows):
            marker.hide()
        self.hide()

    def refresh_geometry(self):
        if not self._box or self._source_size.isEmpty() or not self._host:
            return
        parent = self.parentWidget()
        if parent is self._host:
            host_rect = self._host.rect()
        else:
            host_rect = QRect(self._host.mapTo(parent, QPoint(0, 0)),
                              self._host.size())
        source_width = self._source_size.width()
        source_height = self._source_size.height()
        scale = min(host_rect.width() / source_width,
                    host_rect.height() / source_height)
        draw_width = source_width * scale
        draw_height = source_height * scale
        offset_x = host_rect.left() + (host_rect.width() - draw_width) / 2
        offset_y = host_rect.top() + (host_rect.height() - draw_height) / 2
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
        marker_width = max(thickness, right - left)
        marker_height = max(thickness, bottom - top)
        self._shadows[0].setGeometry(left + 1, top + 1,
                                     marker_width, thickness)
        self._shadows[1].setGeometry(left + 1, bottom - thickness + 1,
                                     marker_width, thickness)
        self._shadows[2].setGeometry(left + 1, top + 1,
                                     thickness, marker_height)
        self._shadows[3].setGeometry(right - thickness + 1, top + 1,
                                     thickness, marker_height)
        self._edges[0].setGeometry(left, top, max(thickness, right - left), thickness)
        self._edges[1].setGeometry(left, max(top, bottom - thickness),
                                   max(thickness, right - left), thickness)
        self._edges[2].setGeometry(left, top, thickness, max(thickness, bottom - top))
        self._edges[3].setGeometry(max(left, right - thickness), top,
                                   thickness, max(thickness, bottom - top))
        for shadow in self._shadows:
            shadow.show()
            shadow.raise_()
        for edge in self._edges:
            edge.show()
            edge.raise_()

    def raise_edges(self):
        for edge in (*self._shadows, *self._edges):
            if edge.isVisible():
                edge.raise_()

    def dispose(self):
        for edge in (*self._edges, *self._shadows):
            edge.hide()
            edge.deleteLater()
        self._edges = []
        self._shadows = []
        self.deleteLater()


class FaceMarkerVideoWidget(QVideoWidget):
    rightClicked = Signal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.face_overlay = FaceBoxOverlay(self)

    def stack_face_overlay_on(self, parent):
        """Keep markers outside QVideoWidget's native rendering surface."""
        previous = self.face_overlay
        self.face_overlay = FaceBoxOverlay(self, parent)
        previous.dispose()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.face_overlay.refresh_geometry()

    def contextMenuEvent(self, event):
        self.rightClicked.emit(QPointF(event.pos()))
        event.accept()


class SoftwareVideoWidget(QLabel):
    """CPU-painted video surface for virtualized macOS graphics."""
    rightClicked = Signal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_sink = QVideoSink(self)
        self._source_pixmap = None
        self.face_overlay = FaceBoxOverlay(self)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_sink.videoFrameChanged.connect(self._show_video_frame)

    def videoSink(self):
        return self._video_sink

    def setAspectRatioMode(self, _mode):
        # This widget always letterboxes frames using KeepAspectRatio.
        pass

    def stack_face_overlay_on(self, parent):
        previous = self.face_overlay
        self.face_overlay = FaceBoxOverlay(self, parent)
        previous.dispose()

    def _show_video_frame(self, frame):
        image = frame.toImage()
        if image.isNull():
            return
        self._source_pixmap = QPixmap.fromImage(image)
        self._fit_pixmap()

    def _fit_pixmap(self):
        if self._source_pixmap and not self._source_pixmap.isNull():
            self.setPixmap(self._source_pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_pixmap()
        self.face_overlay.refresh_geometry()

    def contextMenuEvent(self, event):
        self.rightClicked.emit(QPointF(event.pos()))
        event.accept()


def use_software_video_output():
    """Return whether native Qt video rendering is unsafe on this machine."""
    override = os.environ.get("SCENESPY_SOFTWARE_VIDEO", "").strip().lower()
    if override:
        return override not in {"0", "false", "no", "off"}
    if sys.platform != "darwin":
        return False
    try:
        result = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True, text=True, timeout=2, check=False)
        identity = f"{result.stdout}\n{result.stderr}".lower()
        markers = ("virtualbox", "oracle vm", "innotek", "vbox")
        if any(marker in identity for marker in markers):
            return True
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5, check=False)
        identity = f"{result.stdout}\n{result.stderr}".lower()
        return any(marker in identity for marker in markers)
    except (OSError, subprocess.SubprocessError):
        return False


class ProcessingPreviewLabel(QLabel):
    """Letterboxed preview which always keeps the complete source frame visible."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_pixmap = None
        self.face_marker_box = []
        self.face_marker_source_size = QSize()
        self.face_marker_color = QColor("#00ff00")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_source_pixmap(self, pixmap):
        self.source_pixmap = pixmap
        self._fit_pixmap()

    def clear_source(self):
        self.source_pixmap = None
        self.clear_face_marker()
        self.clear()

    def set_face_marker(self, box, source_width, source_height,
                        color="#00ff00"):
        self.face_marker_box = [float(value) for value in (box or [])]
        self.face_marker_source_size = QSize(
            int(source_width), int(source_height))
        self.face_marker_color = QColor(color)
        self.update()

    def clear_face_marker(self):
        self.face_marker_box = []
        self.face_marker_source_size = QSize()
        self.update()

    def _fit_pixmap(self):
        if self.source_pixmap and not self.source_pixmap.isNull():
            self.setPixmap(self.source_pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_pixmap()

    def paintEvent(self, event):
        super().paintEvent(event)
        pixmap = self.pixmap()
        if (not pixmap or pixmap.isNull() or
                len(self.face_marker_box) != 4 or
                self.face_marker_source_size.isEmpty()):
            return
        image_rect = QRectF(
            (self.width() - pixmap.width()) / 2,
            (self.height() - pixmap.height()) / 2,
            pixmap.width(), pixmap.height())
        sx = image_rect.width() / self.face_marker_source_size.width()
        sy = image_rect.height() / self.face_marker_source_size.height()
        x1, y1, x2, y2 = self.face_marker_box
        marker = QRectF(
            image_rect.left() + x1 * sx,
            image_rect.top() + y1 * sy,
            max(1.0, (x2 - x1) * sx),
            max(1.0, (y2 - y1) * sy)).intersected(image_rect)
        if marker.isEmpty():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#000000"), 2))
        painter.drawRect(marker.translated(1, 1))
        painter.setPen(QPen(self.face_marker_color, 2))
        painter.drawRect(marker)


class FaceCropEditor(ProcessingPreviewLabel):
    cropChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.crop_box = []
        self.source_size = QSize()
        self._drag_mode = None
        self._drag_start = QPointF()
        self._start_rect = QRectF()
        self.marker_color = QColor("#00ff00")
        self.setMouseTracking(True)

    def set_face_frame(self, pixmap, crop_box, source_width, source_height,
                       color="#00ff00"):
        self.crop_box = [float(value) for value in crop_box]
        self.source_size = QSize(int(source_width), int(source_height))
        self.marker_color = QColor(color)
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
        painter.setPen(QPen(QColor("#000000"), 2))
        painter.drawRect(rect.translated(1, 1))
        painter.setPen(QPen(self.marker_color, 2))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor("#000000"), 1))
        painter.setBrush(self.marker_color)
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


class UpdateCheckTask(QThread):
    updateAvailable = Signal(object)

    def run(self):
        try:
            request = urllib.request.Request(
                UPDATE_API_URL,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": f"Scenespy/{__version__}",
                },
            )
            with urllib.request.urlopen(request, timeout=8) as response:
                if not _trusted_update_url(response.geturl()):
                    return
                release = json.loads(response.read().decode("utf-8"))
            latest = str(release.get("tag_name") or "").lstrip("vV")
            if not latest or version_tuple(latest) <= version_tuple(__version__):
                return
            suffix = update_asset_suffix()
            assets = release.get("assets") or []
            asset = next((item for item in assets
                          if str(item.get("name") or "").lower().endswith(suffix)), None)
            if not asset:
                return
            asset_name = str(asset.get("name") or "")
            checksum_name = asset_name + ".sha256"
            checksum = next((item for item in assets
                             if str(item.get("name") or "") == checksum_name), None)
            download_url = str(asset.get("browser_download_url") or "")
            checksum_url = str((checksum or {}).get("browser_download_url") or "")
            if (Path(asset_name).name != asset_name or
                    not _trusted_update_url(download_url) or
                    not _trusted_update_url(checksum_url)):
                return
            self.updateAvailable.emit({
                "version": latest,
                "name": asset_name,
                "url": download_url,
                "checksum_url": checksum_url,
                "size": max(0, int(asset.get("size") or 0)),
                "notes": str(release.get("body") or "").strip(),
            })
        except Exception:
            # Update checks are opportunistic and never interrupt app startup.
            return


class UpdateDownloadTask(QThread):
    progress = Signal(int)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, release, parent=None):
        super().__init__(parent)
        self.release = dict(release)

    @staticmethod
    def _open(url):
        request = urllib.request.Request(
            url, headers={"User-Agent": f"Scenespy/{__version__}"})
        response = urllib.request.urlopen(request, timeout=30)
        if not _trusted_update_url(response.geturl()):
            response.close()
            raise RuntimeError("GitHub redirected the update to an untrusted host.")
        return response

    def run(self):
        target = None
        partial = None
        try:
            with self._open(self.release["checksum_url"]) as response:
                checksum_text = response.read(4096).decode("ascii", "strict")
            match = re.search(r"\b([0-9a-fA-F]{64})\b", checksum_text)
            if not match:
                raise RuntimeError("The release checksum is missing or invalid.")
            expected_hash = match.group(1).lower()
            update_dir = Path(USER_DATA_DIR) / "updates"
            update_dir.mkdir(parents=True, exist_ok=True)
            target = update_dir / self.release["name"]
            partial = target.with_suffix(target.suffix + ".part")
            digest = hashlib.sha256()
            downloaded = 0
            expected_size = max(0, int(self.release.get("size") or 0))
            with self._open(self.release["url"]) as response, open(partial, "wb") as output:
                while not self.isInterruptionRequested():
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if expected_size:
                        self.progress.emit(min(100, int(downloaded * 100 / expected_size)))
            if self.isInterruptionRequested():
                raise InterruptedError("Update download cancelled.")
            if expected_size and downloaded != expected_size:
                raise RuntimeError("The downloaded update size does not match the GitHub release.")
            if digest.hexdigest().lower() != expected_hash:
                raise RuntimeError("The downloaded update failed SHA-256 verification.")
            os.replace(partial, target)
            if sys.platform.startswith("linux") and target.suffix == ".run":
                target.chmod(target.stat().st_mode | 0o700)
            self.completed.emit(str(target))
        except InterruptedError:
            if partial:
                try:
                    partial.unlink(missing_ok=True)
                except OSError:
                    pass
        except Exception as error:
            if partial:
                try:
                    partial.unlink(missing_ok=True)
                except OSError:
                    pass
            self.failed.emit(str(error) or type(error).__name__)


class Task(QThread):
    progress = Signal(float, str)
    metrics = Signal(object, object)
    preview = Signal(object, float)
    face_found = Signal(object, object)
    segments_found = Signal(object, object)
    media_started = Signal(object)
    media_analyzed = Signal(object)
    media_finished = Signal(object, bool, object)
    cut_preview_started = Signal(object)
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
        self.media_result_ready_at = {}
        # Queue workers wait for the UI to select and prepare each video's
        # preview before a very fast analysis (notably Every Seconds) starts.
        self.preview_ready = threading.Event()
        self.cut_preview_ready = threading.Event()
        self._cleanup_callbacks = []

    def register_cancel(self, callback):
        self.cancel_callback = callback
        if self.isInterruptionRequested() and callback:
            callback()

    def register_cleanup(self, callback):
        """Register idempotent cleanup that runs on every worker exit path."""
        if callback:
            self._cleanup_callbacks.append(callback)

    def run_cleanups(self):
        callbacks, self._cleanup_callbacks = self._cleanup_callbacks, []
        for callback in reversed(callbacks):
            try:
                callback()
            except Exception:
                pass

    def cancel(self):
        self.requestInterruption()
        if self.cancel_callback:
            try:
                self.cancel_callback()
            except Exception:
                pass

    def run(self):
        outcome = None
        payload = None
        try:
            payload = self.work(self)
            if self.isInterruptionRequested():
                outcome = "cancelled"
            else:
                outcome = "result"
        except Exception:
            if self.isInterruptionRequested():
                outcome = "cancelled"
            else:
                outcome = "failed"
                payload = traceback.format_exc()
        finally:
            self.run_cleanups()
        if outcome == "result":
            self.result.emit(payload)
        elif outcome == "failed":
            self.failed.emit(payload)
        else:
            self.cancelled.emit()


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
        self._source_token = None
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

    @staticmethod
    def _file_token(source):
        try:
            stat = os.stat(source)
            return str(source), int(stat.st_mtime_ns), int(stat.st_size)
        except OSError:
            return str(source), 0, 0

    def _open(self, source, token):
        if self._source_token == token and self._container is not None:
            return
        if self._container is not None:
            self._container.close()
        self._container = av.open(source)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._source = source
        self._source_token = token
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
        token = self._file_token(source)
        key = (*token, target)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        self._open(source, token)
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
            self._remember((*token, index), image)
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


class ClipThumbnailDecodeTask(QThread):
    """Decode one small timeline-thumbnail batch away from the Qt UI thread."""
    def __init__(self, path, jobs, parent=None):
        super().__init__(parent)
        self.path = str(path)
        self.jobs = list(jobs)
        self.results = []

    @staticmethod
    def _thumbnail_image(image):
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
        return QImage(
            image.data, target_width, target_height,
            target_width * 3, QImage.Format.Format_RGB888).copy()

    def run(self):
        cap = cv2.VideoCapture(self.path)
        try:
            for key, frame, end_frame in self.jobs:
                if self.isInterruptionRequested():
                    return
                image, source_frame = None, None
                if cap.isOpened():
                    for candidate in clip_preview_candidates(frame, end_frame):
                        if self.isInterruptionRequested():
                            return
                        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
                        ok, decoded = cap.read()
                        if ok and decoded is not None:
                            image, source_frame = decoded, candidate
                            break
                qimage = (self._thumbnail_image(image)
                          if image is not None else QImage())
                self.results.append((key, qimage, source_frame))
        finally:
            cap.release()


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
        else:
            self.timeline.editCancelled.emit()

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
        self.setAcceptHoverEvents(True)
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

    def _update_hover_cursor(self, position):
        cursor = (Qt.CursorShape.ArrowCursor
                  if position.y() <= 13
                  else Qt.CursorShape.SizeHorCursor)
        self.setCursor(cursor)

    def hoverEnterEvent(self, event):
        self._update_hover_cursor(event.pos())
        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event):
        self._update_hover_cursor(event.pos())
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        super().hoverLeaveEvent(event)

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
        super().__init__(
            start_x, 100, TIMELINE_INDICATOR_WIDTH,
            TIMELINE_INDICATOR_HEIGHT)
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
        self.context_selected = False
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        super().paint(painter, option, widget)
        header = QRectF(
            self.rect().left() + .5, self.rect().top() + .5,
            max(0.0, self.rect().width() - 1), 5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.base_color.lighter(118))
        painter.drawRect(header)
        hover_active = self.direct_hover or self.linked_hover
        selected = self.isSelected() or self.context_selected
        if not hover_active:
            if selected:
                self._paint_selected_bottom(painter)
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
        if selected:
            self._paint_selected_bottom(painter)

    def _paint_selected_bottom(self, painter):
        inner = self.rect().adjusted(.5, .5, -.5, -.5)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#f8fafc"), 1))
        painter.drawLine(inner.bottomLeft(), inner.bottomRight())

    def set_context_selected(self, selected):
        selected = bool(selected)
        if selected != self.context_selected:
            self.context_selected = selected
            self.update()

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
        super().__init__(
            x, 101, TIMELINE_INDICATOR_WIDTH,
            TIMELINE_INDICATOR_HEIGHT)
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
            f"Clip {index} · available when analysis is complete"
            if self.live_preview else f"Preview clip {index}")
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
            selectable = self.timeline.clip_selection_available()
            self.selected = selectable
            self.update()
            self.timeline.show_clip_duration(
                self.start_frame, self.end_frame, self.index)
            start_seconds = self.timeline.media.seconds_for_frame(self.start_frame)
            end_seconds = self.timeline.media.seconds_for_frame(self.end_frame)
            if selectable:
                self.timeline.segmentActivated.emit(
                    start_seconds, end_seconds, self.index)
            else:
                self.timeline.clear_context_selection()
                self.timeline.segmentPreviewRequested.emit(
                    start_seconds, end_seconds)
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
        self.timeline._hide_hover_time()
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
        self.timeline._hide_hover_time()
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
        # Recreating a dense graphics scene faster than it can be painted only
        # queues mouse events and makes the grip feel stuck. Sparse timelines
        # retain the 30 Hz path; dense ones leave enough time for input/paint.
        zoom_hz = 12 if len(self.timeline._indicator_items) > 60 else 30
        if now - self._last_zoom_rebuild < 1 / zoom_hz:
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

    def enterEvent(self, event):
        self.timeline._hide_hover_time()
        super().enterEvent(event)


class TimelineView(QGraphicsView):
    seekRequested = Signal(float)
    scrubStarted = Signal()
    scrubFinished = Signal(float)
    editStarted = Signal()
    editCancelled = Signal()
    cutSelected = Signal(object)
    segmentActivated = Signal(float, float, int)
    segmentSelected = Signal(float, float, int)
    segmentPreviewRequested = Signal(float, float)
    contentChanged = Signal()
    faceRemoveRequested = Signal(object)
    segmentRemoveRequested = Signal(int)
    cutAddRequested = Signal(float)
    cutRemoveRequested = Signal(object)
    cutMoved = Signal(object)
    selectionCleared = Signal()

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
        self._selected_segment_index = None
        self._clip_thumbnail_cache = OrderedDict()
        self._clip_thumbnail_source_frames = OrderedDict()
        self._face_thumbnail_cache = OrderedDict()
        self._clip_thumbnail_attempted = set()
        self._clip_thumbnail_decoder = None
        self._clip_thumbnail_pending = set()
        self._face_thumbnail_attempted = set()
        self._clip_thumbnail_limit = CACHE_PROFILE["clip_thumbnail_cache"]
        self._face_thumbnail_limit = CACHE_PROFILE["face_thumbnail_cache"]
        # Clip frames are decoded by ClipThumbnailDecodeTask, so use the
        # machine profile to avoid rebuilding the entire graphics scene after
        # every couple of thumbnails. Face crops still load on the UI thread
        # and therefore retain deliberately small batches.
        self._clip_thumbnail_batch_size = CACHE_PROFILE["thumbnail_batch"]
        self._face_thumbnail_batch_size = min(
            2, CACHE_PROFILE["thumbnail_batch"])
        self._thumbnail_rebuild_pending = False
        self._large_timeline = False
        self._dense_timeline = False
        self._max_rendered_segments = 400
        self._max_thumbnail_segments = 80
        self._viewport_rebuild_timer = QTimer(self)
        self._viewport_rebuild_timer.setSingleShot(True)
        self._viewport_rebuild_timer.setInterval(40)
        self._viewport_rebuild_timer.timeout.connect(
            self._rebuild_visible_timeline)
        self.horizontalScrollBar().valueChanged.connect(
            self._schedule_viewport_rebuild)
        self._wheel_zoom_factor = 1.0
        self._wheel_zoom_x = None
        self._wheel_zoom_timer = QTimer(self)
        self._wheel_zoom_timer.setSingleShot(True)
        self._wheel_zoom_timer.setInterval(50)
        self._wheel_zoom_timer.timeout.connect(self._apply_wheel_zoom)
        self._live_face_media_id = None
        self._live_face_ids = set()
        self._live_face_lane_right_edges = []
        self._live_segment_media_id = None
        self._live_segment_cuts = None
        self._live_segment_removed = None
        self._live_segment_load_thumbnails = False
        self._clip_duration_label = None
        self._hover_guide = None
        self._hover_time_bubble = QLabel(None, Qt.WindowType.ToolTip)
        self._hover_time_bubble.setAttribute(
            Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._hover_time_bubble.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hover_time_bubble.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground)
        self._hover_time_bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hover_time_bubble.setFixedHeight(23)
        self._hover_time_bubble.setStyleSheet(
            "QLabel { background-color:#292c30; color:#e5e7eb; "
            "border:1px solid #50545b; border-radius:3px; "
            "padding:2px 5px; font-size:9px; font-weight:600; }")
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

    def _schedule_viewport_rebuild(self, _value=None):
        if self._large_timeline and self.media:
            self._viewport_rebuild_timer.start()

    def _rebuild_visible_timeline(self):
        if not self.media:
            return
        if (self._live_segment_media_id == self.media.media_id and
                self._live_segment_cuts is not None and
                not self.media.analyzed):
            self.show_live_segments(
                self.media, self._live_segment_cuts,
                self._live_segment_removed or [],
                load_thumbnails=self._live_segment_load_thumbnails)
        else:
            self.rebuild()

    def set_media(self, media):
        changed = media is not self.media
        if changed:
            self.current_seconds = 0.0
            self._selected_segment_index = None
        self.media = media
        if changed and media:
            self.fit_to_view()
        else:
            self.rebuild()

    def display_timecode(self, seconds, duration=None):
        """Format timeline times using the shared View preference."""
        formatter = player_timecode if self.show_milliseconds else transport_timecode
        return formatter(seconds, duration)

    def clip_selection_available(self):
        """A clip has contextual selection only after a real cut exists."""
        return bool(self.media and self.media.cuts)

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

    def _clip_thumbnails(self, frames, frame_ends=None):
        """Decode tiny timeline thumbnails once and reuse them across rebuilds."""
        if not self.media or not frames:
            return {}
        path = os.path.abspath(self.media.path)
        token = self._thumbnail_file_token(path)
        keys = {int(frame): (*token, int(frame)) for frame in frames}
        missing = [int(frame) for frame in frames
                   if (keys[int(frame)] not in self._clip_thumbnail_cache and
                       keys[int(frame)] not in self._clip_thumbnail_attempted and
                       keys[int(frame)] not in self._clip_thumbnail_pending)]
        if missing and self._clip_thumbnail_decoder is None:
            decode_now = missing[:self._clip_thumbnail_batch_size]
            frame_ends = frame_ends or {}
            jobs = [
                (keys[frame], frame, frame_ends.get(frame, frame + 1))
                for frame in decode_now]
            self._clip_thumbnail_pending.update(key for key, _frame, _end in jobs)
            decoder = ClipThumbnailDecodeTask(path, jobs, self)
            self._clip_thumbnail_decoder = decoder
            decoder.finished.connect(
                lambda current=decoder:
                self._clip_thumbnail_decode_finished(current))
            decoder.start()
        thumbnails = {}
        for frame in frames:
            key = keys[int(frame)]
            pixmap = self._clip_thumbnail_cache.get(key, QPixmap())
            if key in self._clip_thumbnail_cache:
                self._clip_thumbnail_cache.move_to_end(key)
                self._clip_thumbnail_source_frames.move_to_end(key)
            thumbnails[frame] = pixmap
        return thumbnails

    def _clip_thumbnail_decode_finished(self, decoder):
        if self._clip_thumbnail_decoder is decoder:
            self._clip_thumbnail_decoder = None
        interrupted = decoder.isInterruptionRequested()
        for key, image, source_frame in decoder.results:
            self._clip_thumbnail_pending.discard(key)
            self._clip_thumbnail_attempted.add(key)
            if not interrupted:
                self._remember_clip_thumbnail(
                    key,
                    QPixmap.fromImage(image) if not image.isNull() else QPixmap(),
                    source_frame)
        for key, _frame, _end in decoder.jobs:
            self._clip_thumbnail_pending.discard(key)
        decoder.deleteLater()
        if not interrupted and self.media:
            self._schedule_thumbnail_rebuild()

    def stop_thumbnail_decoder(self, timeout_ms=3000):
        decoder = self._clip_thumbnail_decoder
        if decoder is None:
            return True
        decoder.requestInterruption()
        stopped = decoder.wait(max(0, int(timeout_ms)))
        if stopped and self._clip_thumbnail_decoder is decoder:
            self._clip_thumbnail_decoder = None
        return stopped

    @staticmethod
    def _thumbnail_file_token(path):
        try:
            stat = os.stat(path)
            return path, int(stat.st_mtime_ns), int(stat.st_size)
        except OSError:
            return path, 0, 0

    def _remember_clip_thumbnail(self, key, pixmap, source_frame):
        self._clip_thumbnail_cache[key] = pixmap
        self._clip_thumbnail_source_frames[key] = source_frame
        self._clip_thumbnail_cache.move_to_end(key)
        self._clip_thumbnail_source_frames.move_to_end(key)
        while len(self._clip_thumbnail_cache) > self._clip_thumbnail_limit:
            old_key, _pixmap = self._clip_thumbnail_cache.popitem(last=False)
            self._clip_thumbnail_source_frames.pop(old_key, None)

    def _schedule_thumbnail_rebuild(self):
        if self._thumbnail_rebuild_pending:
            return
        self._thumbnail_rebuild_pending = True

        def rebuild_next_batch():
            self._thumbnail_rebuild_pending = False
            scrollbar = self.horizontalScrollBar()
            if (isinstance(scrollbar, TimelineZoomScrollBar) and
                    scrollbar._zoom_edge is not None):
                self._schedule_thumbnail_rebuild()
                return
            if self.media:
                if (self._live_segment_media_id == self.media.media_id and
                        self._live_segment_cuts is not None and
                        not self.media.analyzed):
                    self.show_live_segments(
                        self.media, self._live_segment_cuts,
                        self._live_segment_removed or [],
                        load_thumbnails=self._live_segment_load_thumbnails)
                else:
                    self.rebuild()

        QTimer.singleShot(16, rebuild_next_batch)

    def clip_thumbnail_source_frame(self, start_frame, end_frame):
        """Return the cached source frame used by a clip indicator."""
        if not self.media:
            return None
        start = clip_preview_frame(start_frame, end_frame)
        path = os.path.abspath(self.media.path)
        key = (*self._thumbnail_file_token(path), start)
        if key not in self._clip_thumbnail_cache:
            self._clip_thumbnails([start], {start: end_frame})
        pixmap = self._clip_thumbnail_cache.get(key, QPixmap())
        if pixmap.isNull():
            return None
        return self._clip_thumbnail_source_frames.get(key, start)

    def _face_thumbnails(self, events):
        thumbnails = {}
        pending = []
        for event in events:
            path = os.path.abspath(event.crop_file) if event.crop_file else ""
            key = self._thumbnail_file_token(path)
            if (key not in self._face_thumbnail_cache and
                    key not in self._face_thumbnail_attempted):
                pending.append((event, path, key))
        if len(pending) > self._face_thumbnail_batch_size:
            self._schedule_thumbnail_rebuild()
        for _event, path, key in pending[:self._face_thumbnail_batch_size]:
            self._face_thumbnail_attempted.add(key)
            if key not in self._face_thumbnail_cache:
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
                self._face_thumbnail_cache[key] = pixmap
                self._face_thumbnail_cache.move_to_end(key)
                while len(self._face_thumbnail_cache) > self._face_thumbnail_limit:
                    self._face_thumbnail_cache.popitem(last=False)
        for event in events:
            path = os.path.abspath(event.crop_file) if event.crop_file else ""
            key = self._thumbnail_file_token(path)
            pixmap = self._face_thumbnail_cache.get(key, QPixmap())
            if key in self._face_thumbnail_cache:
                self._face_thumbnail_cache.move_to_end(key)
            thumbnails[event.event_id] = pixmap
        return thumbnails

    def invalidate_face_thumbnail(self, path):
        if path:
            absolute = os.path.abspath(path)
            for key in [key for key in self._face_thumbnail_cache
                        if key[0] == absolute]:
                self._face_thumbnail_cache.pop(key, None)
            self._face_thumbnail_attempted = {
                key for key in self._face_thumbnail_attempted
                if key[0] != absolute}

    def discard_media_cache(self, media):
        """Release in-memory thumbnails owned by a removed media item."""
        video_path = os.path.abspath(media.path)
        decoder = self._clip_thumbnail_decoder
        if (decoder is not None and
                os.path.normcase(os.path.abspath(decoder.path)) ==
                os.path.normcase(video_path)):
            decoder.requestInterruption()
        face_paths = {
            os.path.abspath(event.crop_file)
            for event in media.faces if event.crop_file}
        for key in [key for key in self._clip_thumbnail_cache
                    if key[0] == video_path]:
            self._clip_thumbnail_cache.pop(key, None)
            self._clip_thumbnail_source_frames.pop(key, None)
        for key in [key for key in self._face_thumbnail_cache
                    if key[0] in face_paths]:
            self._face_thumbnail_cache.pop(key, None)
        self._clip_thumbnail_attempted = {
            key for key in self._clip_thumbnail_attempted
            if key[0] != video_path}
        self._face_thumbnail_attempted = {
            key for key in self._face_thumbnail_attempted
            if key[0] not in face_paths}

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

    def _clear_scene_items(self):
        # scene.clear() destroys C++ graphics items synchronously.  Invalidate
        # the reference first so re-entrant control updates remain safe.
        self.playhead = None
        self.scene().clear()

    def rebuild(self):
        self._live_face_media_id = None
        self._live_face_ids = set()
        self._live_face_lane_right_edges = []
        self._live_segment_media_id = None
        self._live_segment_cuts = None
        self._live_segment_removed = None
        self._live_segment_load_thumbnails = False
        self._clip_duration_label = None
        self._hover_guide = None
        self._hover_time_bubble.hide()
        self._clear_scene_items()
        self._indicator_items = []
        self._indicator_lefts = []
        self._active_indicator_items = set()
        if not self.media:
            self._large_timeline = False
            self._dense_timeline = False
            text = self.scene().addSimpleText("Import a video to begin")
            text.setBrush(QColor(TEXT_MUTED))
            return
        boundaries = [0, *(event.frame for event in sorted(
            self.media.cuts, key=lambda cut: cut.frame)), self.media.total_frames]
        removed = set(self.media.removed_segments)
        self._visible_layout = []
        self._layout_source_starts = []
        self._layout_source_ends = []
        self._layout_display_starts = []
        self._layout_display_ends = []
        display_start = 0.0
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
            if index in removed:
                continue
            source_start = self.media.seconds_for_frame(start)
            source_end = self.media.seconds_for_frame(end)
            self._visible_layout.append((start, end, index, display_start))
            self._layout_source_starts.append(source_start)
            self._layout_source_ends.append(source_end)
            self._layout_display_starts.append(display_start)
            display_start += max(0.0, source_end - source_start)
            self._layout_display_ends.append(display_start)
        self._display_duration = display_start
        self.content_width = max(self.viewport().width() - 20,
                                 self._display_duration * self.pixels_per_second)
        scene_height = 175
        self.scene().setSceneRect(
            0, 0, self.content_width + TIMELINE_INDICATOR_WIDTH,
            scene_height)
        self._draw_ruler()
        self._large_timeline = (
            len(self._visible_layout) > self._max_rendered_segments)
        rendered_layout = self._visible_layout
        if self._large_timeline:
            visible_rect = self.mapToScene(
                self.viewport().rect()).boundingRect()
            overscan = max(100.0, visible_rect.width() * 0.5)
            visible_left = visible_rect.left() - overscan
            visible_right = visible_rect.right() + overscan
            rendered_layout = [
                item for item in self._visible_layout
                if (self.x_for_frame(item[1]) >= visible_left and
                    self.x_for_frame(item[0]) <= visible_right)]
        self._dense_timeline = (
            self._large_timeline and
            len(rendered_layout) > self._max_rendered_segments)
        if self._dense_timeline:
            if self._clip_thumbnail_decoder is not None:
                self._clip_thumbnail_decoder.requestInterruption()
            rendered_layout = []
            overview = QGraphicsRectItem(
                0, 35, self.content_width, 57)
            overview.setPen(QPen(QColor("#59616c"), 1))
            overview.setBrush(QColor("#383b40"))
            overview.setToolTip(
                f"{len(self._visible_layout)} clips - zoom in to inspect details")
            self.scene().addItem(overview)
            density_label = self.scene().addSimpleText(
                f"{len(self._visible_layout)} clips - zoom in for details")
            density_label.setBrush(QColor(TEXT_MUTED))
            density_label.setPos(12, 48)
            density_label.setZValue(3)
        clip_frames = [clip_preview_frame(start, end)
                       for start, end, _index, _display in rendered_layout]
        clip_ends = {clip_preview_frame(start, end): end
                     for start, end, _index, _display in rendered_layout}
        has_sequences = self.media.analyzed or bool(self.media.cuts)
        clip_thumbnails = (self._clip_thumbnails(clip_frames, clip_ends)
                           if self.media.analyzed and
                           self.media.mode != "faces" and
                           len(rendered_layout) <= self._max_thumbnail_segments
                           else {})
        segment_colors = (self._segment_colors(len(boundaries) - 1)
                          if has_sequences else
                          [QColor("#383b40")] * (len(boundaries) - 1))
        segment_height = 57
        for start, end, index, _display_start in rendered_layout:
            segment_enabled = index not in self.media.excluded_segments
            x = self.x_for_frame(start)
            width = max(1, self.x_for_frame(end) - x)
            tooltip = (f"Scene {index} · "
                       f"{self.display_timecode(self.media.seconds_for_frame(start), self.media.duration)} – "
                       f"{self.display_timecode(self.media.seconds_for_frame(end), self.media.duration)}")
            segment = SceneSegmentItem(
                (x, 35, width, segment_height), segment_colors[index - 1], tooltip)
            segment.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsSelectable,
                self.clip_selection_available())
            segment.start_frame = start
            segment.end_frame = end
            segment.index = index
            segment.set_context_selected(
                index == self._selected_segment_index)
            if not has_sequences:
                # The initial full-length clip keeps the standard timeline
                # treatment without pretending an analysis result exists.
                segment.setToolTip("")
            elif not self.media.analyzed:
                segment.setToolTip(
                    f"Clip {index} · "
                    f"{self.display_timecode(self.media.seconds_for_frame(start), self.media.duration)} – "
                    f"{self.display_timecode(self.media.seconds_for_frame(end), self.media.duration)}")
            segment.setOpacity(1.0 if segment_enabled else .38)
            self.scene().addItem(segment)
            if has_sequences and width >= 64:
                label = self.scene().addSimpleText(str(index))
                label.setBrush(QColor("#cbd5e1"))
                label.setPos(x + 8, 48)
                label.setOpacity(1.0 if segment_enabled else .38)
                label.setZValue(3)
            if (self.media.analyzed and self.media.mode != "faces" and
                    index not in self.media.removed_segments):
                chip = ClipPreviewChip(
                    start, end, index, x, self, segment,
                    clip_thumbnails.get(clip_preview_frame(start, end)),
                    segment_enabled,
                    segment_colors[index - 1])
                self.scene().addItem(chip)
                self._indicator_items.append(
                    (x, x + TIMELINE_INDICATOR_WIDTH, chip))
        name = self.scene().addSimpleText(Path(self.media.path).name)
        name.setFont(QFont("JetBrains Mono", 8))
        name_width = max(1.0, name.boundingRect().width())
        name_fade = QLinearGradient(0, 0, name_width, 0)
        name_color = QColor(TEXT_MUTED)
        name_fade.setColorAt(0.0, name_color)
        name_fade.setColorAt(0.5, name_color)
        transparent_name = QColor(name_color)
        transparent_name.setAlpha(0)
        name_fade.setColorAt(1.0, transparent_name)
        name.setBrush(QBrush(name_fade))
        name.setPos(12, 68)
        name.setZValue(2)
        drawn_handle_positions = set()
        ordered_cuts = sorted(self.media.cuts, key=lambda cut: cut.frame)
        visible_scene_rect = self.mapToScene(
            self.viewport().rect()).boundingRect()
        handle_overscan = max(100.0, visible_scene_rect.width() * 0.5)
        for boundary_index, event in enumerate(ordered_cuts, 1):
            # A removed interval is already a ripple join.  Its source-side
            # boundaries collapse onto the same display point and must not be
            # draggable as if they were an ordinary visible cut.
            if (boundary_index in removed or
                    boundary_index + 1 in removed):
                continue
            handle_x = self.x_for_frame(event.frame)
            if (self._dense_timeline or
                    (self._large_timeline and not (
                        visible_scene_rect.left() - handle_overscan <= handle_x <=
                        visible_scene_rect.right() + handle_overscan))):
                continue
            position_key = round(handle_x, 3)
            if position_key in drawn_handle_positions:
                continue
            drawn_handle_positions.add(position_key)
            handle = CutHandle(event, self)
            handle.setPos(handle_x, 28)
            handle.setToolTip(
                f"Cut · {self.display_timecode(self.media.seconds_for_frame(event.frame), self.media.duration)}")
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
                lane, right = place_timeline_indicator_lane(
                    left, lane_right_edges)
                chip = FaceChip(
                    event, start_x, face_color,
                    face_thumbnails.get(event.event_id))
                chip.setY(
                    face_y - 100 + lane * FACE_INDICATOR_LANE_STEP)
                self.scene().addItem(chip)
                self._indicator_items.append((start_x, right, chip))
            scene_height = max(
                scene_height,
                face_y + len(lane_right_edges) * FACE_INDICATOR_LANE_STEP + 8)
            self.scene().setSceneRect(
                0, 0, self.content_width + TIMELINE_INDICATOR_WIDTH,
                scene_height)
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
        lane, right = place_timeline_indicator_lane(
            start_x, self._live_face_lane_right_edges)
        thumbnail = self._face_thumbnails([event]).get(event.event_id)
        chip = FaceChip(
            event, start_x, self._face_colors([event])[0], thumbnail,
            live_preview=True)
        chip.setY(lane * FACE_INDICATOR_LANE_STEP)
        self.scene().addItem(chip)
        insert_at = bisect_right(self._indicator_lefts, start_x)
        self._indicator_lefts.insert(insert_at, start_x)
        self._indicator_items.insert(insert_at, (start_x, right, chip))
        scene_rect = self.sceneRect()
        required_height = max(
            175, 108 + len(self._live_face_lane_right_edges) *
            FACE_INDICATOR_LANE_STEP)
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
        self._live_segment_cuts = None
        self._live_segment_removed = None
        self._live_segment_load_thumbnails = False

    def show_live_segments(self, media, cuts, removed, load_thumbnails=True):
        """Render the complete transient cut timeline without committing it."""
        if media is not self.media:
            return
        # Reuse the exact normal-mode renderer so segment blocks, cut handles,
        # ripple positions, labels, playhead mapping, and final thumbnails are
        # identical.  Restore the model immediately: live queue results remain
        # visual-only until that item actually completes successfully.
        saved = (
            media.cuts,
            media.removed_segments,
            media.excluded_segments,
            media.analyzed,
        )
        try:
            media.cuts = list(cuts)
            media.removed_segments = list(removed)
            media.excluded_segments = []
            # Progressive layouts draw precise colored segments without doing
            # repeated thumbnail I/O.  The finalized layout uses the same
            # full-quality thumbnails as normal mode.
            media.analyzed = bool(load_thumbnails)
            self.rebuild()
        finally:
            (media.cuts, media.removed_segments,
             media.excluded_segments, media.analyzed) = saved
        self._live_segment_media_id = media.media_id
        self._live_segment_cuts = list(cuts)
        self._live_segment_removed = list(removed)
        self._live_segment_load_thumbnails = bool(load_thumbnails)

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
            self.display_timecode(duration, self.media.duration))
        label.setFont(QFont("JetBrains Mono", 8))
        label.setBrush(QColor(TEXT_MUTED))
        indicator_center = self.x_for_frame(start_frame) + 20
        label_width = label.boundingRect().width()
        label_x = max(
            2.0,
            min(indicator_center - label_width / 2,
                max(2.0, self.scene().sceneRect().right() - label_width - 2.0)),
        )
        label.setPos(label_x, 130)
        label.setToolTip(f"Clip {int(index)} exact duration")
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
        sample_width = QFontMetrics(QFont()).horizontalAdvance(
            self.display_timecode(duration, duration))
        label_spacing = max(72.0, sample_width + 18.0)
        if timecode_has_hours(duration):
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
        minor_step = step / 4
        visible_start = 0.0
        visible_end = duration
        if self.content_width > max(1, self.viewport().width()) * 2:
            visible_rect = self.mapToScene(
                self.viewport().rect()).boundingRect()
            overscan = max(100.0, visible_rect.width() * 0.5)
            visible_start = max(
                0.0, (visible_rect.left() - overscan) /
                max(self.content_width, 1.0) * duration)
            visible_end = min(
                duration, (visible_rect.right() + overscan) /
                max(self.content_width, 1.0) * duration)
        minor_index = max(0, int(visible_start // minor_step))
        minor = minor_index * minor_step
        while minor <= visible_end + .001:
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
        moment = max(0.0, int(visible_start // step) * step)
        while moment <= visible_end + .001:
            x = moment / max(duration, .001) * self.content_width
            tick = QGraphicsLineItem(x, 22, x, 31)
            tick.setPen(QPen(QColor("#7d8794"), 1.25))
            self.scene().addItem(tick)
            label = self.scene().addSimpleText(
                self.display_timecode(moment, duration))
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
        position = bisect_right(
            self._layout_source_starts, source_seconds) - 1
        if position < 0:
            display_seconds = 0.0
        else:
            source_start = self._layout_source_starts[position]
            source_end = self._layout_source_ends[position]
            display_start = self._layout_display_starts[position]
            display_seconds = display_start + max(
                0.0, min(source_seconds, source_end) - source_start)
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
        position = bisect_right(
            self._layout_display_starts, display_seconds) - 1
        if position >= 0:
            return min(
                self._layout_source_ends[position],
                self._layout_source_starts[position] +
                display_seconds - self._layout_display_starts[position],
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
        previous_frame = int(handle.event.frame)
        handle.event.frame = max(
            lower, min(upper, self.frame_for_x(handle.x())))
        handle.setToolTip(
            f"Cut · {self.display_timecode(self.media.seconds_for_frame(handle.event.frame), self.media.duration)}")
        if int(handle.event.frame) != previous_frame:
            self.cutMoved.emit(handle.event)

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
                    anchor.y() - self._hover_time_bubble.height() - 22)
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
        items = [item for item in self.items(position)
                 if item is not self._hover_guide]
        interactive_types = (
            CutHandle, PlayheadItem, FaceChip, ClipPreviewChip, SceneSegmentItem)
        return next((item for item in items
                     if isinstance(item, interactive_types)),
                    items[0] if items else None)

    def _clear_indicator_selection(self):
        for _left, _right, item in self._indicator_items:
            if isinstance(item, ClipPreviewChip) and item.selected:
                item.selected = False
                item.update()

    def clear_segment_context(self):
        self._selected_segment_index = None
        for item in self.scene().items():
            if isinstance(item, SceneSegmentItem):
                item.set_context_selected(False)

    def select_segment_context(self, index):
        index = int(index)
        self._selected_segment_index = index
        for item in self.scene().items():
            if isinstance(item, SceneSegmentItem):
                item.set_context_selected(item.index == index)

    def clear_context_selection(self):
        self._clear_indicator_selection()
        self.clear_segment_context()
        signals_were_blocked = self.scene().blockSignals(True)
        try:
            self.scene().clearSelection()
        finally:
            self.scene().blockSignals(signals_were_blocked)
        self.selectionCleared.emit()

    def mousePressEvent(self, event):
        item = self._timeline_item_at(event.position().toPoint())
        selectable_item = (
            isinstance(item, (CutHandle, FaceChip, ClipPreviewChip)) or
            (isinstance(item, SceneSegmentItem) and
             self.clip_selection_available()))
        if event.button() == Qt.MouseButton.LeftButton and not selectable_item:
            self.clear_context_selection()
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
                self.clip_selection_available()):
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
            self._wheel_zoom_factor *= (
                1.18 if event.angleDelta().y() > 0 else .85)
            self._wheel_zoom_x = float(event.position().x())
            if not self._wheel_zoom_timer.isActive():
                self._wheel_zoom_timer.start()
            event.accept()
            return
        super().wheelEvent(event)

    def _apply_wheel_zoom(self):
        factor = self._wheel_zoom_factor
        viewport_x = self._wheel_zoom_x
        self._wheel_zoom_factor = 1.0
        self._wheel_zoom_x = None
        if self.media and abs(factor - 1.0) > 1e-9:
            self.zoom_by_factor(factor, viewport_x)

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
            "view/show_milliseconds", False, type=bool)
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
        self._queue_saved_bytes = 0
        self._manual_face_draft = None
        self.preview_end_ms = None
        self.preview_start_ms = None
        self.preview_armed = False
        self._segment_preview_status_active = False
        self._segment_preview_last_status_ms = None
        self._pending_interactive_seek_ms = None
        self._interactive_seek_inflight_ms = None
        self._frame_step_target = None
        self._frame_step_media_id = None
        self._playback_proxy_tasks = {}
        # Playback proxies are deliberately serialized and suspended while a
        # user-requested analysis/export owns the machine.  A proxy is only a
        # UI optimization and must never compete with the actual work.
        self._deferred_playback_proxy_ids = set()
        self._shutdown_started = False
        self._pending_proxy_path = None
        self._pending_proxy_restore = None
        self._edit_preview_active = False
        self._edit_preview_frame = None
        self._edit_preview_media_id = None
        self._active_playback_face_key = None
        self._playback_face_index = None
        self._undo_stacks = {}
        self._redo_stacks = {}
        self._pending_timeline_edit = None
        self._play_after_edit_sync = False
        self._update_check_task = None
        self._update_download_task = None
        self._update_prompted_version = None
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
        # QMediaPlayer.positionChanged may be coarser than a face's short
        # visibility window.  Poll only while playing; lookup remains O(1).
        self._face_overlay_timer = QTimer(self)
        self._face_overlay_timer.setInterval(33)
        self._face_overlay_timer.timeout.connect(
            self._poll_playback_face_overlay)
        self._frame_decoder = None
        self._ensure_frame_decoder()
        self._build_actions()
        self._set_milliseconds_visible(self.show_milliseconds)
        QApplication.instance().installEventFilter(self)
        self._apply_theme()
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()
        self._update_transport_access()
        if getattr(sys, "frozen", False):
            QTimer.singleShot(1800, self._check_for_updates)

    def eventFilter(self, watched, event):
        """Provide Premiere-style unmodified arrow-key frame stepping."""
        if (hasattr(self, "video_backdrop") and
                watched in (self.video_backdrop, self.viewer_panel,
                            self.main_splitter, self.menuBar()) and
                event.type() in (
                    QEvent.Type.Move, QEvent.Type.Resize,
                    QEvent.Type.LayoutRequest, QEvent.Type.Show)):
            if (watched is self.video_backdrop and
                    event.type() in (QEvent.Type.Move, QEvent.Type.Resize)):
                self._position_media_info()
            self._schedule_media_info_position()
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
        self._software_video_output = use_software_video_output()
        self.video = (SoftwareVideoWidget() if self._software_video_output
                      else FaceMarkerVideoWidget())
        self.video.rightClicked.connect(self.show_video_context_menu)
        self.video.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.video.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.video.setStyleSheet("background: transparent;")
        self.video.setMinimumSize(500, 280)
        self.video.hide()
        self.player.setVideoOutput(
            self.video.videoSink() if self._software_video_output else self.video)
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
        self.auto_process = QCheckBox("Auto process queue")
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
        self.media_list.setIconSize(STATUS_ICON_SIZE)
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
        # Keep this overlay in the window's coordinate system.  A child of the
        # menu bar inherits its style/content offsets, which makes its apparent
        # left edge drift away from the player as the window is resized.
        self.media_info = QLabel(self)
        self.media_info.setObjectName("mediaInfo")
        self.media_info.setTextFormat(Qt.TextFormat.RichText)
        self.media_info.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.media_info.setContentsMargins(0, 0, 0, 0)
        self.media_info.setIndent(0)
        self.media_info.hide()
        self._media_info_position_timer = QTimer(self)
        self._media_info_position_timer.setSingleShot(True)
        # Wait one short UI frame so nested splitters/layouts have all settled.
        self._media_info_position_timer.setInterval(12)
        self._media_info_position_timer.timeout.connect(
            self._position_media_info)
        self.video_backdrop = VideoBackdrop()
        self.video_backdrop.installEventFilter(self)
        self.viewer_panel.installEventFilter(self)
        self.main_splitter.installEventFilter(self)
        self.menuBar().installEventFilter(self)
        backdrop_layout = QStackedLayout(self.video_backdrop)
        backdrop_layout.setContentsMargins(0, 0, 0, 0)
        backdrop_layout.addWidget(self.video)
        self.video.stack_face_overlay_on(self.video_backdrop)
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
        self.timeline_lock_effect = QGraphicsOpacityEffect(
            self.timeline.viewport())
        self.timeline_lock_effect.setOpacity(1.0)
        self.timeline.viewport().setGraphicsEffect(
            self.timeline_lock_effect)
        self.timeline.show_milliseconds = self.show_milliseconds
        self.seek.show_milliseconds = self.show_milliseconds
        self.timeline.seekRequested.connect(self._request_timeline_seek)
        self.timeline.scrubStarted.connect(self._begin_timeline_scrub)
        self.timeline.scrubFinished.connect(self._finish_timeline_scrub)
        self.timeline.editStarted.connect(self._begin_timeline_edit)
        self.timeline.editCancelled.connect(self._cancel_timeline_edit)
        self.timeline.cutSelected.connect(self.show_cut)
        self.timeline.segmentActivated.connect(self.show_segment)
        self.timeline.segmentSelected.connect(self.show_segment)
        self.timeline.segmentPreviewRequested.connect(self.play_segment)
        self.timeline.contentChanged.connect(self._timeline_content_changed)
        self.timeline.faceRemoveRequested.connect(self.remove_face_permanently)
        self.timeline.segmentRemoveRequested.connect(self.remove_segment_permanently)
        self.timeline.cutAddRequested.connect(self.add_cut_at)
        self.timeline.cutRemoveRequested.connect(self.remove_cut_marker)
        self.timeline.cutMoved.connect(self._cut_marker_moved)
        self.timeline.selectionCleared.connect(self.clear_selection)
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
        self.selection_cut_time_row = QWidget()
        self.selection_cut_time_row.setObjectName("selectionCutTimeRow")
        selection_cut_time_layout = QHBoxLayout(self.selection_cut_time_row)
        selection_cut_time_layout.setContentsMargins(0, 0, 0, 0)
        selection_cut_time_layout.setSpacing(6)
        selection_cut_time_label = QLabel("Time:", objectName="muted")
        selection_cut_time_layout.addWidget(selection_cut_time_label)
        self.selection_cut_time = MillisecondTimeEdit()
        self.selection_cut_time.setObjectName("selectionCutTime")
        self.selection_cut_time.setFont(self.selection_detail.font())
        self.selection_cut_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selection_cut_time.setFixedHeight(24)
        self.selection_cut_time.setToolTip("Edit cut marker time")
        self.selection_cut_time.editingFinished.connect(
            self._apply_selected_cut_time)
        selection_cut_time_layout.addWidget(self.selection_cut_time)
        selection_cut_time_layout.addStretch(1)
        self.selection_cut_time_row.hide()
        self._selection_cut_original_text = ""
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
        self._segment_first_icon = themed_icon("fa6s.backward-step")
        self._segment_last_icon = themed_icon("fa6s.forward-step")
        self._segment_trash_icon = themed_icon("fa6s.trash")
        self._segment_restore_icon = themed_icon("fa6s.rotate-left")
        self.segment_cut_button = QToolButton()
        self.segment_cut_button.setIcon(themed_icon("fa6s.scissors"))
        self.segment_cut_button.setToolTip("Add cut at playhead (Ctrl+K)")
        self.segment_cut_button.clicked.connect(self.add_cut)
        self.segment_cut_button.hide()
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
        for button in (self.segment_cut_button, self.remove_button,
                       self.face_previous_button, self.face_next_button,
                       self.face_reset_button, self.face_add_button,
                       self.face_cancel_button, self.face_save_button):
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
        selection_card_layout.addWidget(self.selection_cut_time_row)
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
        self.saved_data_summary = QLabel(objectName="muted")
        self.saved_data_summary.hide()
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
            button.setObjectName("faceMetricAction")
            button.setFixedSize(20, 20)
            button.setIconSize(QSize(10, 10))
            row.addWidget(button)
            row.addStretch(1)
            face_metrics_layout.addLayout(row)
        self.face_metrics.hide()
        self.progress_label = QLabel("Ready", objectName="muted")
        self.progress_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.progress_percent = QLabel("", objectName="muted")
        self.progress_percent.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.progress_percent.hide()
        self.progress_state_icon = QLabel()
        # Keep transparent breathing room around circular glyphs. A pixmap
        # touching an equally-sized QLabel loses its antialiased top edge on
        # some Qt/Windows scale factors.
        # The 20 px canvas leaves transparent breathing room around the
        # standard-size glyph, independently of the adjacent text margins.
        self.progress_state_icon.setFixedSize(STATUS_ICON_SIZE)
        self.progress_state_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_state_icon.hide()
        progress_status_row = QHBoxLayout()
        progress_status_row.setContentsMargins(0, 0, 0, 0)
        progress_status_row.setSpacing(5)
        progress_status_row.addWidget(
            self.progress_state_icon, 0, Qt.AlignmentFlag.AlignVCenter)
        progress_status_row.addWidget(
            self.progress_label, 1, Qt.AlignmentFlag.AlignVCenter)
        progress_status_row.addWidget(self.progress_percent)
        self.action_button = QPushButton("Start", objectName="primary")
        self.action_button.clicked.connect(self.toggle_primary_action)
        layout.addWidget(self.saved_data_summary)
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
        root.splitterMoved.connect(
            lambda _position, _index: self._schedule_media_info_position())
        self.setStatusBar(QStatusBar())
        self._set_progress_metrics(stage=None)
        self._schedule_media_info_position()

    def _schedule_media_info_position(self):
        timer = getattr(self, "_media_info_position_timer", None)
        if timer is not None:
            # Restarting makes this a true debounce: position from the final
            # layout event, never from an intermediate splitter geometry.
            timer.start()

    def _position_media_info(self):
        if not hasattr(self, "media_info") or not hasattr(self, "video_backdrop"):
            return
        menu_bar = self.menuBar()
        player_top_left = self.mapFromGlobal(
            self.video_backdrop.mapToGlobal(QPoint(0, 0)))
        left = player_top_left.x()
        right = left + self.video_backdrop.width()
        corner = menu_bar.cornerWidget(Qt.Corner.TopRightCorner)
        if corner is not None and corner.isVisible():
            corner_left = self.mapFromGlobal(
                corner.mapToGlobal(QPoint(0, 0))).x()
            right = min(right, corner_left - 6)
        menu_top = self.mapFromGlobal(menu_bar.mapToGlobal(QPoint(0, 0))).y()
        left = max(0, min(left, self.width()))
        right = max(left, min(right, self.width()))
        self.media_info.setGeometry(
            left, menu_top, max(0, right - left), menu_bar.height())
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
        self._schedule_media_info_position()

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
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo_timeline_edit)
        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcuts([
            QKeySequence.StandardKey.Redo,
            primary_shortcut(Qt.Key.Key_Z, shift=True),
        ])
        self.redo_action.triggered.connect(self.redo_timeline_edit)
        self.delete_timeline_action = QAction("Delete Timeline Item", self)
        self.delete_timeline_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        self.delete_timeline_action.triggered.connect(
            self.delete_selected_timeline_item)
        self.addAction(self.delete_timeline_action)
        edit_menu.addAction(playback_action)
        edit_menu.addActions([previous_frame_action, next_frame_action])
        edit_menu.addSeparator()
        edit_menu.addActions([self.undo_action, self.redo_action])
        edit_menu.addSeparator()
        edit_menu.addAction(self.add_cut_action)
        view_menu = ShortcutMenu("&View", self)
        self.menuBar().addMenu(view_menu)
        view_menu.addAction(reset_layout_action)
        view_menu.addActions([zoom_in_action, zoom_out_action])
        view_menu.addSeparator()
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
        self.seek.show_milliseconds = self.show_milliseconds
        self.timeline._hide_hover_time()
        timeline_scene = self.timeline.scene()
        signals_were_blocked = timeline_scene.blockSignals(True)
        try:
            self.timeline.rebuild()
        finally:
            timeline_scene.blockSignals(signals_were_blocked)
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
            self._update_media_info_text()
            self._refresh_selection_timecodes()
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
        spin_up_disabled = (image_dir / "spin-up-disabled.svg").as_posix()
        spin_down_disabled = (image_dir / "spin-down-disabled.svg").as_posix()
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
        QComboBox:disabled, QSpinBox:disabled {{ color:#5d626a; border-color:#34373c; background:#292b2f; }}
        QComboBox::drop-down {{ width:25px; border:0; border-left:1px solid #444950; }}
        QComboBox QAbstractItemView {{ background:#25282c; color:#e5e7eb; border:1px solid #555a62; selection-background-color:#286da9; padding:3px; outline:0; }}
        QSpinBox {{ padding-right:26px; }}
        QSpinBox::up-button {{ subcontrol-origin:border; subcontrol-position:top right; width:24px; background:transparent; border:0; }}
        QSpinBox::down-button {{ subcontrol-origin:border; subcontrol-position:bottom right; width:24px; background:transparent; border:0; }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background:transparent; border:0; }}
        QSpinBox::up-arrow {{ image:url("{spin_up}"); width:8px; height:5px; }}
        QSpinBox::down-arrow {{ image:url("{spin_down}"); width:8px; height:5px; }}
        QSpinBox::up-arrow:disabled {{ image:url("{spin_up_disabled}"); }}
        QSpinBox::down-arrow:disabled {{ image:url("{spin_down_disabled}"); }}
        QSpinBox QLabel#intervalMinutesHint {{ color:#9ca3af; background:transparent; border:0; padding:0; }}
        QLabel#intervalMinutesHint:disabled {{ color:#565b63; }}
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
        QToolButton#faceMetricAction {{ background:#303339; border:1px solid #555a62; border-radius:7px; margin:1px; padding:0; }}
        QToolButton#faceMetricAction:hover {{ background:#464a50; border-color:#737b87; }}
        QToolButton#faceMetricAction:pressed {{ background:#25282c; border-color:#ffffff; }}
        QWidget#selectionCutTimeRow {{ background:transparent; border:0; }}
        QLineEdit#selectionCutTime {{ background:#25282c; color:#f3f4f6; border:1px solid #555a62; border-radius:5px; margin-left:1px; padding:2px 7px; font-weight:400; }}
        QLineEdit#selectionCutTime:hover, QLineEdit#selectionCutTime:focus {{ border-color:#8b949f; background:#2d3136; }}
        QToolButton#selectionClose {{ background:transparent; border:0; border-radius:8px; padding:0; }}
        QToolButton#selectionClose:hover {{ background:#3a3d42; border:0; border-radius:8px; }}
        QToolButton#selectionClose:pressed {{ background:#303339; border:0; border-radius:8px; }}
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
        QFrame#transportGroup QLineEdit#editableTimecode:disabled, QFrame#transportGroup QLineEdit#timecode:disabled {{ background:transparent; color:#5d626a; border:0; }}
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

        if removed_current:
            self.player.stop()
            self.player.setSource(QUrl())
            self._release_frame_decoder()
        self._discard_media_runtime(removed_media)

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
            self.video.hide()
            self.timeline.set_media(None)
            self.show_cut(None)
            self.seek.setRange(0, 0)
            zero = self._player_timecode(0, 0)
            self.clock.setText(zero)
            self.duration_clock.setText(zero)
            self.media_info.clear()
            self.media_info.hide()
            self._set_saved_data_summary()
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
        self._release_frame_decoder()
        self.video.hide()
        for media in list(self.project.media):
            self._discard_media_runtime(media)
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
        self._set_saved_data_summary()
        self._restore_selected_media_progress()
        self._update_batch_availability()
        self._refresh_action_button()
        self.statusBar().showMessage(
            f"Removed {count} videos from the list · original files were not deleted", 5000)

    @staticmethod
    def _media_cache_path(media):
        return os.path.join(USER_DATA_DIR, "cache", str(media.media_id))

    @staticmethod
    def _remove_media_cache(media):
        """Remove only the private cache owned by one media item."""
        cache_root = os.path.abspath(os.path.join(USER_DATA_DIR, "cache"))
        return remove_owned_directory(
            ScenespyApp._media_cache_path(media), cache_root)

    def _discard_media_runtime(self, media):
        """Drop background work, transient state, and cache for removed media."""
        media_id = media.media_id
        self._deferred_playback_proxy_ids.discard(media_id)
        self._undo_stacks.pop(media_id, None)
        self._redo_stacks.pop(media_id, None)
        self.timeline.discard_media_cache(media)
        cache_cleanup_deferred = False
        proxy_task = self._playback_proxy_tasks.pop(media_id, None)
        if proxy_task and proxy_task.isRunning():
            proxy_task.cancel()
            if not proxy_task.wait(3000):
                # Its cancel-aware FFmpeg loop will finish asynchronously; do
                # the disk cleanup only after it can no longer recreate files.
                proxy_task.finished.connect(
                    lambda m=media: self._remove_media_cache(m))
                cache_cleanup_deferred = True
        self._batch_analyzed_media.discard(media_id)
        self._batch_completed_media.discard(media_id)
        self._batch_failed_media.discard(media_id)
        if hasattr(self, "_queue_media_ids"):
            self._queue_media_ids = [
                queued for queued in self._queue_media_ids
                if queued != media_id]
        if self._pending_proxy_path and self._pending_proxy_path[0] == media_id:
            self._pending_proxy_path = None
        if (self._pending_proxy_restore and
                self._pending_proxy_restore[0] == media_id):
            self._pending_proxy_restore = None
        if not cache_cleanup_deferred:
            self._remove_media_cache(media)

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
            "Auto process queue", menu, hover_color=ACCENT)
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
            self._deferred_playback_proxy_ids.discard(media.media_id)
            return
        # Do not decode/transcode a second copy of the source while analysis or
        # cutting is active, and never build multiple proxies concurrently.
        if self.task or self._playback_proxy_tasks:
            self._deferred_playback_proxy_ids.add(media.media_id)
            return
        self._deferred_playback_proxy_ids.discard(media.media_id)

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
        if task.isRunning():
            task.finished.connect(
                lambda m=media, t=task:
                self._playback_proxy_finished(m, t))
            return
        if self._playback_proxy_tasks.get(media.media_id) is task:
            self._playback_proxy_tasks.pop(media.media_id, None)
        task.deleteLater()
        if not self.task and not self._shutdown_started:
            QTimer.singleShot(0, self._resume_deferred_playback_proxy)

    def _pause_playback_proxies(self):
        """Yield CPU, decoder and disk bandwidth to analysis/export work."""
        tasks = list(self._playback_proxy_tasks.items())
        for media_id, proxy_task in tasks:
            self._deferred_playback_proxy_ids.add(media_id)
            if proxy_task.isRunning():
                proxy_task.cancel()
        # Cancellation is cooperative but FFmpeg checks it frequently.  Wait a
        # bounded interval so the real task does not start beside the proxy.
        for _media_id, proxy_task in tasks:
            if proxy_task.isRunning():
                proxy_task.wait(3000)

    def _resume_deferred_playback_proxy(self):
        """Restart one useful proxy after foreground processing becomes idle."""
        if (self.task or self._shutdown_started or
                self._playback_proxy_tasks or
                not self._deferred_playback_proxy_ids):
            return
        media_by_id = {
            media.media_id: media for media in getattr(self.project, "media", [])}
        self._deferred_playback_proxy_ids.intersection_update(media_by_id)
        if not self._deferred_playback_proxy_ids:
            return
        current_id = getattr(self.current_media, "media_id", None)
        media_id = (current_id if current_id in self._deferred_playback_proxy_ids
                    else next(iter(self._deferred_playback_proxy_ids)))
        self._deferred_playback_proxy_ids.discard(media_id)
        self._ensure_playback_proxy(media_by_id[media_id])

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
        self._invalidate_playback_face_index()
        self._active_playback_face_key = None
        self.video.face_overlay.clear_face()
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
        self._stop_segment_preview()
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
        self._set_saved_data_summary(
            self.current_media.exported_bytes
            if self.current_media.exported else None)
        self._restore_selected_media_progress()
        self.show_cut(None)

    def _restore_selected_media_progress(self):
        """Show progress belonging to the selected media, not the previous task."""
        media = self.current_media
        if self._current_media_is_processing():
            value, label = getattr(self.task, "media_progress", {}).get(
                media.media_id, (0, "Processing"))
        elif media and media.exported:
            value, label = 1000, self._completed_media_progress_label(media)
        elif media and media.analyzed:
            value, label = 1000, "Video analyzed"
        else:
            value, label = 0, "Ready"
        self._set_progress_value(value, immediate=True)
        self.progress_label.setText(label)
        self._refresh_progress_state_icon()
        self._update_transport_access()

    @staticmethod
    def _completed_media_progress_label(media):
        if media.mode == "faces":
            available = [event for event in media.faces
                         if face_event_is_included(media, event)]
            total = sum(1 for event in available if event.enabled)
            return ScenespyApp._face_export_progress_label(total, total)
        unavailable = set(media.excluded_segments) | set(media.removed_segments)
        total = sum(
            1 for index, _segment in enumerate(media.segments(), 1)
            if index not in unavailable)
        noun = "Scenes" if media.mode == "scene" else "Segments"
        return f"{noun} cut: {total} / {total}"

    def _update_media_info_text(self):
        if not self.current_media:
            self.media_info.clear()
            return
        fps_text = f"{self.current_media.fps:.2f}".rstrip("0").rstrip(".")
        extension = Path(self.current_media.path).suffix.lstrip(".").upper()
        try:
            size_text = format_file_size(os.path.getsize(self.current_media.path))
        except OSError:
            size_text = ""
        details = (
            f"{self.current_media.width} × {self.current_media.height} · "
            f"{fps_text} fps · "
            f"{self._player_timecode(self.current_media.duration, self.current_media.duration)}")
        if size_text:
            details += f" · {size_text}"
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
            QTimer.singleShot(0, lambda: self.current_media and
                              self._sync_playback_face_overlay(
                                  self.current_media.frame_for_seconds(
                                      self.player.position() / 1000)))
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
            self._face_overlay_timer.start()
        elif self._pending_proxy_path:
            QTimer.singleShot(0, self._activate_pending_playback_proxy)
        if not playing:
            self._face_overlay_timer.stop()
            self._poll_playback_face_overlay()
            if (self._segment_preview_status_active and
                    self.preview_armed and self.preview_end_ms is not None):
                self._stop_segment_preview("Clip preview stopped")
        self.play_button.setIcon(self._pause_icon if playing else self._play_icon)

    def _poll_playback_face_overlay(self):
        media = self.current_media
        if not media:
            return
        self._sync_playback_face_overlay(
            media.frame_for_seconds(self.player.position() / 1000))

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
        self._stop_segment_preview()
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
        self._frame_decoder = FramePreviewDecoder(
            self, cache_size=CACHE_PROFILE["frame_cache"])
        self._frame_decoder.frameReady.connect(self._show_edit_preview_frame)
        self._frame_decoder.start()

    def _release_frame_decoder(self, force=False):
        decoder = self._frame_decoder
        if decoder is None:
            return True
        self._frame_decoder = None
        decoder.stop()
        stopped = decoder.wait(3000)
        if not stopped and force:
            decoder.terminate()
            stopped = decoder.wait(1000)
        if stopped:
            decoder.deleteLater()
        else:
            # Never destroy a running QThread.  It owns a bounded cache and
            # will delete itself as soon as the current decoder call returns.
            decoder.finished.connect(decoder.deleteLater)
        return stopped

    def _show_edit_preview_frame(self, media_id, frame, image):
        if (not self._edit_preview_active or not self.current_media or
                self.current_media.media_id != media_id or
                self._edit_preview_media_id != media_id or
                self._edit_preview_frame != frame or image.isNull()):
            return
        self.processing_preview.setStyleSheet("background:#000000; border:0;")
        self.processing_preview.set_source_pixmap(QPixmap.fromImage(image))
        match = self._playback_face_candidate(frame)
        if match is not None:
            _event_id, _candidate_frame, box, saved = match
            self.processing_preview.set_face_marker(
                box, self.current_media.width or image.width(),
                self.current_media.height or image.height(),
                "#00ff00" if saved else DANGER)
        else:
            self.processing_preview.clear_face_marker()
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
        # Queue results stay transactional until that item has exported, but
        # its preview transport must already follow the exact finalized cut
        # layout visible in the timeline.
        task = getattr(self, "task", None)
        if task and getattr(self, "processing_media", None) is media:
            live = task.live_segments.get(media.media_id)
            if live is not None:
                cuts, live_removed, _finalized = live
                if cuts is not None:
                    boundaries = [
                        0, *(cut.frame for cut in cuts), media.total_frames]
                    removed = set(live_removed or [])
                    return [
                        segment for index, segment in enumerate(
                            zip(boundaries, boundaries[1:]), 1)
                        if index not in removed]
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

    def _set_cut_time_editor(self, event=None):
        if not isinstance(event, CutEvent) or not self.current_media:
            self._selection_cut_original_text = ""
            self.selection_cut_time_row.hide()
            return
        text = self._player_timecode(
            self.current_media.seconds_for_frame(event.frame),
            self.current_media.duration)
        self._selection_cut_original_text = text
        self.selection_cut_time.setText(text)
        width = self.selection_cut_time.fontMetrics().horizontalAdvance(text) + 19
        self.selection_cut_time.setFixedWidth(max(54, width))
        self.selection_cut_time_row.show()

    def _apply_selected_cut_time(self):
        event, media = self.selected_cut, self.current_media
        if (not isinstance(event, CutEvent) or not media or
                self._current_media_is_processing()):
            self._set_cut_time_editor(event)
            return
        value = self.selection_cut_time.text().strip()
        if value == self._selection_cut_original_text:
            return
        seconds = parse_timecode(value)
        if seconds is None:
            self._set_cut_time_editor(event)
            return
        lower, upper = self.timeline.bounds_for(event)
        target_frame = max(
            lower, min(upper, media.frame_for_seconds(seconds)))
        if target_frame == event.frame:
            self._set_cut_time_editor(event)
            return
        self._record_timeline_edit("Change cut time")
        if not event.original_frame:
            event.original_frame = event.frame
        event.frame = target_frame
        media.normalize_cuts()
        self.timeline.rebuild()
        self._timeline_content_changed()
        self.show_cut(event)

    def _cut_marker_moved(self, event):
        """Refresh Selection during a drag without rebuilding the timeline."""
        selected = self.selected_cut
        if (not isinstance(event, CutEvent) or
                not isinstance(selected, CutEvent) or
                selected.event_id != event.event_id):
            return
        self.selection_detail.setText(
            f"Frame {event.frame}\nSource: {event.source}")
        self._set_cut_time_editor(event)

    def _set_segment_selection_detail(self, start_seconds, end_seconds, index,
                                      include_analysis_details=True):
        duration = end_seconds - start_seconds
        excluded = index in self.current_media.excluded_segments
        detail = (
            f"Start: {self._player_timecode(start_seconds, self.current_media.duration)}\n"
            f"End: {self._player_timecode(end_seconds, self.current_media.duration)}\n"
            f"Duration: {self._player_timecode(duration, self.current_media.duration)}")
        if include_analysis_details:
            detail += (
                f"\nStatus: {'Excluded' if excluded else 'Included'}\n"
                "Move the boundary markers to adjust this range.")
        self.selection_detail.setText(detail)

    def _configure_segment_actions(self, index):
        """Show the same frame/toggle controls before and after analysis."""
        editable = not self._current_media_is_processing()
        excluded = index in self.current_media.excluded_segments
        self.face_previous_button.setIcon(self._segment_first_icon)
        self.face_previous_button.setToolTip("Show first frame")
        self.face_previous_button.setEnabled(editable)
        self.face_previous_button.setVisible(True)
        self.face_next_button.setIcon(self._segment_last_icon)
        self.face_next_button.setToolTip("Show last frame")
        self.face_next_button.setEnabled(editable)
        self.face_next_button.setVisible(True)
        self.segment_cut_button.setVisible(True)
        self.face_reset_button.setVisible(False)
        self.face_add_button.setVisible(False)
        self.face_cancel_button.setVisible(False)
        self.face_save_button.setVisible(False)
        self.remove_button.setEnabled(editable)
        self.remove_button.setVisible(True)
        self.remove_button.setIcon(
            self._segment_restore_icon if excluded else self._segment_trash_icon)
        self.remove_button.setToolTip(
            "Include sequence" if excluded else "Exclude sequence")
        self._refresh_cut_action_state()

    def _refresh_selection_timecodes(self):
        """Refresh an open Selection card after changing the View format."""
        if not self.current_media:
            return
        if self.selected_segment:
            index, start_seconds, end_seconds = self.selected_segment
            self.timeline.select_segment_context(index)
            self._set_segment_selection_detail(
                start_seconds, end_seconds, index,
                include_analysis_details=self.current_media.analyzed)
        elif self.selected_cut is not None:
            self.show_cut(self.selected_cut)

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

    @staticmethod
    def _capture_timeline_state(media):
        return {
            "cuts": copy.deepcopy(media.cuts),
            "faces": copy.deepcopy(media.faces),
            "excluded_segments": list(media.excluded_segments),
            "removed_segments": list(media.removed_segments),
            "exported": bool(media.exported),
        }

    def _push_undo_snapshot(self, state, label="Edit timeline"):
        media = self.current_media
        if not media or self._current_media_is_processing():
            return
        stack = self._undo_stacks.setdefault(media.media_id, [])
        stack.append((state, str(label)))
        del stack[:-50]
        self._redo_stacks[media.media_id] = []
        self._refresh_undo_actions()

    def _record_timeline_edit(self, label):
        if not self.current_media or self._current_media_is_processing():
            return False
        self._pending_timeline_edit = None
        self._push_undo_snapshot(
            self._capture_timeline_state(self.current_media), label)
        return True

    def _clear_edit_history(self, media):
        if not media:
            return
        self._undo_stacks.pop(media.media_id, None)
        self._redo_stacks.pop(media.media_id, None)
        if (self._pending_timeline_edit and
                self._pending_timeline_edit[0] == media.media_id):
            self._pending_timeline_edit = None
        self._refresh_undo_actions()

    def _restore_timeline_state(self, state):
        media = self.current_media
        if not media:
            return
        media.cuts = copy.deepcopy(state["cuts"])
        media.faces = copy.deepcopy(state["faces"])
        media.excluded_segments = list(state["excluded_segments"])
        media.removed_segments = list(state["removed_segments"])
        media.exported = bool(state["exported"])
        media.normalize_cuts()
        self._manual_face_draft = None
        self.selected_cut = None
        self.selected_segment = None
        self.timeline.set_media(media)
        self.show_cut(None)
        self._invalidate_playback_face_index()
        self._refresh_transport_timeline()
        self._refresh_media_metrics()
        self._refresh_media_item_states()
        self._refresh_action_button()
        self._refresh_cut_action_state()

    def undo_timeline_edit(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, QLineEdit) and focus.hasFocus():
            focus.undo()
            return
        media = self.current_media
        if not media or self._current_media_is_processing():
            return
        stack = self._undo_stacks.setdefault(media.media_id, [])
        if not stack:
            return
        state, label = stack.pop()
        self._redo_stacks.setdefault(media.media_id, []).append((
            self._capture_timeline_state(media), label))
        self._restore_timeline_state(state)
        self.statusBar().showMessage(f"Undid: {label}", 3000)
        self._refresh_undo_actions()

    def redo_timeline_edit(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, QLineEdit) and focus.hasFocus():
            focus.redo()
            return
        media = self.current_media
        if not media or self._current_media_is_processing():
            return
        stack = self._redo_stacks.setdefault(media.media_id, [])
        if not stack:
            return
        state, label = stack.pop()
        self._undo_stacks.setdefault(media.media_id, []).append((
            self._capture_timeline_state(media), label))
        self._restore_timeline_state(state)
        self.statusBar().showMessage(f"Redid: {label}", 3000)
        self._refresh_undo_actions()

    def _refresh_undo_actions(self):
        if not hasattr(self, "undo_action"):
            return
        media_id = self.current_media.media_id if self.current_media else None
        locked = self._current_media_is_processing() if self.current_media else False
        undo = self._undo_stacks.get(media_id, []) if media_id else []
        redo = self._redo_stacks.get(media_id, []) if media_id else []
        self.undo_action.setEnabled(bool(undo) and not locked)
        self.redo_action.setEnabled(bool(redo) and not locked)
        self.undo_action.setText(f"Undo {undo[-1][1]}" if undo else "Undo")
        self.redo_action.setText(f"Redo {redo[-1][1]}" if redo else "Redo")

    def delete_selected_timeline_item(self):
        """Permanently remove the selected timeline item, like its context menu."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QSpinBox, QComboBox)) and focus.hasFocus():
            return
        if (not self.current_media or self._current_media_is_processing() or
                self._manual_face_draft is not None):
            return
        if self.selected_segment is not None:
            self.remove_segment_permanently(self.selected_segment[0])
            return
        event = self.selected_cut
        if isinstance(event, FaceEvent):
            self.remove_face_permanently(event)
        elif isinstance(event, CutEvent):
            self.remove_cut_marker(event)

    def _timeline_content_changed(self):
        if self._pending_timeline_edit is not None:
            media_id, state, label = self._pending_timeline_edit
            self._pending_timeline_edit = None
            if self.current_media and self.current_media.media_id == media_id:
                self._push_undo_snapshot(state, label)
        self._mark_current_unexported()
        self._refresh_transport_timeline()
        self._refresh_media_metrics()
        self._refresh_action_button()
        self._refresh_cut_action_state()

    def _begin_timeline_edit(self):
        """Stop transient previews before a cut boundary is edited."""
        if (self.current_media and not self._current_media_is_processing() and
                self._pending_timeline_edit is None):
            self._pending_timeline_edit = (
                self.current_media.media_id,
                self._capture_timeline_state(self.current_media),
                "Move cut marker",
            )
        self.player.pause()
        self._interactive_seek_timer.stop()
        self._pending_interactive_seek_ms = None
        self._interactive_seek_inflight_ms = None
        self._reset_frame_step_target()
        self._stop_segment_preview()

    def _cancel_timeline_edit(self):
        self._pending_timeline_edit = None

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
        self._stop_segment_preview()
        self.player.setPosition(max(0, min(self.player.duration(), int(value))))

    def _set_selection_available(self, available):
        """Show contextual actions only while an analyzed event is selected."""
        if not hasattr(self, "selection_card"):
            return
        available = bool(available)
        has_media = self.current_media is not None
        editable = has_media and not self._current_media_is_processing()
        self.selection_card.setEnabled(editable)
        has_selection = bool(
            self.selected_cut is not None or self.selected_segment is not None or
            self._manual_face_draft is not None)
        if self.selected_segment is not None:
            self.timeline.select_segment_context(self.selected_segment[0])
        else:
            self.timeline.clear_segment_context()
        show_result_actions = available and has_selection
        self.selection_result_actions.setVisible(show_result_actions)
        self.preanalysis_actions.setVisible(not show_result_actions)
        self.preanalysis_cut_button.setEnabled(editable)
        self.preanalysis_disable_button.setEnabled(False)
        if not isinstance(self.selected_cut, CutEvent):
            self._set_cut_time_editor(None)
        if available:
            if not has_selection:
                self.selection_title.setText("No event selected")
                self.selection_detail.setText(
                    "Add a cut at the playhead or select a clip to disable it.")
                self.preanalysis_disable_button.setEnabled(False)
            self._sync_progress_metrics_visibility()
            return
        self._manual_face_draft = None
        self.selected_cut = None
        self.selected_segment = None
        self.timeline.clear_segment_context()
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
        self.timeline.clear_segment_context()
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
        self._invalidate_playback_face_index()
        self.timeline.rebuild()
        if self.current_media:
            self._sync_playback_face_overlay(
                self.current_media.frame_for_seconds(
                    self.player.position() / 1000))

    def show_segment(self, start_seconds, end_seconds, index):
        if not self.current_media or end_seconds <= start_seconds:
            return
        self.timeline.select_segment_context(index)
        if not self.current_media.analyzed:
            self.select_preanalysis_segment(start_seconds, end_seconds, index)
            self.play_segment(start_seconds, end_seconds)
            return
        self._manual_face_draft = None
        self.selected_cut = None
        self.selected_segment = (int(index), float(start_seconds), float(end_seconds))
        self._set_selection_available(True)
        self._sync_progress_metrics_visibility()
        self._set_face_navigation_state(None)
        cuts = sorted(self.current_media.cuts, key=lambda cut: cut.frame)
        adjacent = []
        if index > 1 and index - 2 < len(cuts): adjacent.append(cuts[index - 2])
        if index - 1 < len(cuts): adjacent.append(cuts[index - 1])
        for cut in adjacent:
            if not cut.original_frame:
                cut.original_frame = cut.frame
        self._configure_segment_actions(index)
        self.video.face_overlay.clear_face()
        self.face_review_preview.clear_source()
        self.face_review_preview.hide()
        if not self._processing_preview_selected():
            self.viewer_stack.setCurrentWidget(self.video)
            self.video.show()
        self.player.pause()
        self.selection_title.setText(f"Clip {index}")
        self._set_segment_selection_detail(start_seconds, end_seconds, index)
        self._show_segment_thumbnail(start_seconds, end_seconds)
        self.play_segment(start_seconds, end_seconds)

    def select_preanalysis_segment(self, start_seconds, end_seconds, index):
        if (not self.current_media or self.current_media.analyzed or
                end_seconds <= start_seconds):
            return
        self._manual_face_draft = None
        self.selected_cut = None
        self.selected_segment = (int(index), float(start_seconds), float(end_seconds))
        self._set_selection_available(True)
        self._sync_progress_metrics_visibility()
        excluded = index in self.current_media.excluded_segments
        self._configure_segment_actions(index)
        self.selection_title.setText(f"Clip {index}")
        self._set_segment_selection_detail(
            start_seconds, end_seconds, index,
            include_analysis_details=False)
        self.preanalysis_disable_button.setEnabled(
            not self._current_media_is_processing())
        self.preanalysis_disable_button.setIcon(themed_icon(
            "fa6s.rotate-left" if excluded else "fa6s.trash"))
        self.preanalysis_disable_button.setToolTip(
            "Restore selected clip to analysis" if excluded else
            "Hide selected clip from analysis")
        self.selection_preview.clear_source()
        self.selection_preview.hide()

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

    def _show_segment_thumbnail(self, start_seconds, end_seconds):
        """Reuse the exact source frame chosen for the timeline indicator."""
        if not self.current_media:
            return
        start_frame = self.current_media.frame_for_seconds(start_seconds)
        end_frame = self.current_media.frame_for_seconds(end_seconds)
        frame_number = self.timeline.clip_thumbnail_source_frame(
            start_frame, end_frame)
        self.selection_preview.clear_source()
        if frame_number is None:
            self.selection_preview.hide()
            return
        self._show_segment_still(frame_number)

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
        self._segment_preview_status_active = True
        self._segment_preview_last_status_ms = None
        self.player.setPosition(start_ms)
        QTimer.singleShot(80, self._ensure_segment_preview_started)
        self.statusBar().showMessage(
            f"Previewing clip · "
            f"{self._player_timecode(start_seconds, self.current_media.duration)} / "
            f"{self._player_timecode(end_seconds, self.current_media.duration)}")

    def _update_segment_preview_status(self, position_ms, force=False):
        if (not self._segment_preview_status_active or
                self.preview_start_ms is None or self.preview_end_ms is None or
                not self.current_media):
            return
        position_ms = max(self.preview_start_ms,
                          min(int(position_ms), self.preview_end_ms))
        if (not force and self._segment_preview_last_status_ms is not None and
                abs(position_ms - self._segment_preview_last_status_ms) < 100):
            return
        current = self.statusBar().currentMessage()
        if current and not current.startswith("Previewing clip"):
            self._segment_preview_status_active = False
            return
        self._segment_preview_last_status_ms = position_ms
        self.statusBar().showMessage(
            "Previewing clip · "
            f"{self._player_timecode(position_ms / 1000, self.current_media.duration)} "
            f"/ {self._player_timecode(self.preview_end_ms / 1000, self.current_media.duration)}")

    def _stop_segment_preview(self, message=None):
        owned_status = (self._segment_preview_status_active or
                        self.statusBar().currentMessage().startswith(
                            "Previewing clip"))
        self.preview_end_ms = None
        self.preview_start_ms = None
        self.preview_armed = False
        self._segment_preview_status_active = False
        self._segment_preview_last_status_ms = None
        if not owned_status:
            return
        if message:
            self.statusBar().showMessage(message, 3000)
        else:
            self.statusBar().clearMessage()

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
        frames = self._navigation_event_frames()
        target = (next((frame for frame in frames if frame > current), None)
                  if direction > 0 else
                  next((frame for frame in reversed(frames) if frame < current), None))
        if target is not None:
            self.seek_player_ms(round(self.current_media.seconds_for_frame(target) * 1000))

    def _navigation_event_frames(self):
        """Return exact visible clip/face event frames in media coordinates."""
        media = self.current_media
        if not media:
            return []
        ranges = self._timeline_ranges(media)
        frames = [start for start, _end in ranges[1:]]
        if media.mode == "faces":
            frames.extend(
                face_event_source_frame(face)
                for face in media.faces
                if (face_event_is_included(media, face) and
                    ((face.enabled and self.timeline.show_saved_faces) or
                     (not face.enabled and self.timeline.show_detected_faces))))
        return sorted(set(frames))

    def _position_changed(self, value):
        exact_edit_seconds = None
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
                # QMediaPlayer reports integer milliseconds, while project
                # edits use exact frame timestamps (especially important for
                # fractional rates such as 29.97).  Keep this confirmation on
                # the requested frame instead of moving the timeline back to
                # the backend's rounded millisecond.
                exact_edit_seconds = self.current_media.seconds_for_frame(
                    self._edit_preview_frame)
                play_after_sync = self._play_after_edit_sync
                self._play_after_edit_sync = False
                self._end_edit_preview()
                if play_after_sync:
                    QTimer.singleShot(0, self.player.play)
        source_seconds = (exact_edit_seconds if exact_edit_seconds is not None
                          else value / 1000)
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
        if self.current_media:
            self._sync_playback_face_overlay(
                self.current_media.frame_for_seconds(source_seconds))
        if not self.clock.hasFocus():
            self.clock.setText(self._player_timecode(
                timeline_seconds, timeline_duration))
        self.duration_clock.setText(self._player_timecode(
            timeline_duration, timeline_duration))
        if (self.preview_armed and self.preview_end_ms is not None and
                value >= self.preview_end_ms - 10):
            end = self.preview_end_ms
            self._stop_segment_preview("Clip preview complete")
            self.player.pause()
            self.player.setPosition(end)
        elif self.preview_armed and self.preview_end_ms is not None:
            self._update_segment_preview_status(value)
        self._refresh_cut_action_state()

    def _playback_face_candidate(self, source_frame):
        """Look up the cached trustworthy face box for a playback frame."""
        media = self.current_media
        if not media or media.mode != "faces":
            return None
        if self._playback_face_index is None:
            self._playback_face_index = self._build_playback_face_index()
        return self._playback_face_index.get(int(source_frame))

    def _build_playback_face_index(self):
        """Index short face-display windows once instead of scanning per tick."""
        media = self.current_media
        if not media or media.mode != "faces":
            return {}
        window = max(2, min(6, round(media.fps * .10)))
        indexed = {}
        for event in media.faces:
            if (not face_event_is_included(media, event) or
                    (event.enabled and not self.timeline.show_saved_faces) or
                    (not event.enabled and not self.timeline.show_detected_faces)):
                continue
            start = max(0, int(event.start_frame) - 1)
            end = max(start, int(event.end_frame) - 1)
            candidates = [
                candidate for candidate in event.review_frames
                if len(candidate.get("box") or ()) == 4]
            if not candidates and len(event.box) == 4:
                candidates = [{"frame": event.best_frame, "box": event.box}]
            for candidate in candidates:
                candidate_frame = max(0, int(candidate["frame"]) - 1)
                box = [int(value) for value in candidate["box"]]
                first = max(start - window, candidate_frame - window, 0)
                last = min(end + window, candidate_frame + window,
                           media.total_frames - 1)
                for display_frame in range(first, last + 1):
                    score = (abs(candidate_frame - display_frame),
                             -float(event.quality or 0), event.event_id,
                             candidate_frame)
                    existing = indexed.get(display_frame)
                    if existing is None or score < existing[0]:
                        indexed[display_frame] = (
                            score, (event.event_id, candidate_frame, box,
                                    event.enabled))
        return {frame: match for frame, (_score, match) in indexed.items()}

    def _invalidate_playback_face_index(self):
        self._playback_face_index = None

    def _sync_playback_face_overlay(self, source_frame):
        overlay = self.video.face_overlay
        if (source_frame is None or not self.current_media or
                self.current_media.mode != "faces" or
                self.viewer_stack.currentWidget() is not self.video or
                self._processing_preview_selected()):
            if self._active_playback_face_key is not None:
                overlay.clear_face()
                self._active_playback_face_key = None
            return
        match = self._playback_face_candidate(source_frame)
        if match is None:
            if self._active_playback_face_key is not None:
                overlay.clear_face()
                self._active_playback_face_key = None
            return
        event_id, candidate_frame, box, saved = match
        color = "#00ff00" if saved else DANGER
        key = (event_id, candidate_frame, tuple(box), color)
        if (key == self._active_playback_face_key and
                overlay._box is not None):
            overlay.raise_edges()
            return
        video_size = self.video.videoSink().videoSize()
        source_width = self.current_media.width or video_size.width()
        source_height = self.current_media.height or video_size.height()
        overlay.set_face(box, source_width, source_height, color)
        self._active_playback_face_key = key

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
        # Starting a new attempt replaces any previous error state for this
        # media with the white in-progress state.
        self._batch_failed_media.discard(media.media_id)
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
        self._set_timeline_locked_visual(locked)
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
        self._refresh_cut_action_state()
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
        self._refresh_undo_actions()

    def _set_timeline_locked_visual(self, locked):
        effect = getattr(self, "timeline_lock_effect", None)
        if effect is not None:
            effect.setOpacity(.58 if locked else 1.0)

    def _show_processing_frame(self, image, seconds=-1.0):
        if not self._processing_preview_active or image is None:
            return
        try:
            rgb = image.convert("RGB")
            qimage = QImage(
                rgb.tobytes("raw", "RGB"), rgb.width, rgb.height,
                rgb.width * 3, QImage.Format.Format_RGB888).copy()
            self.processing_preview.setText("")
            self.processing_preview.clear_face_marker()
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
        self._add_cut_frame(self._playhead_cut_frame())

    def _playhead_cut_frame(self):
        """Return the source frame drawn under the timeline playhead."""
        if not self.current_media:
            return 0
        playhead = getattr(self.timeline, "playhead", None)
        if playhead is not None:
            return self.timeline.frame_for_x(playhead.x())
        return self.current_media.frame_for_seconds(
            self.player.position() / 1000)

    def _can_add_cut_frame(self, frame):
        if not self.current_media or self._current_media_is_processing():
            return False
        frame = int(frame)
        return (0 < frame < self.current_media.total_frames and
                all(cut.frame != frame for cut in self.current_media.cuts))

    def _can_add_cut_at(self, seconds):
        frame = (self.current_media.frame_for_seconds(seconds)
                 if self.current_media else 0)
        return ScenespyApp._can_add_cut_frame(self, frame)

    def _refresh_cut_action_state(self):
        if not hasattr(self, "preanalysis_cut_button"):
            return
        frame = self._playhead_cut_frame()
        enabled = self._can_add_cut_frame(frame)
        duplicate = bool(self.current_media and any(
            cut.frame == frame for cut in self.current_media.cuts))
        tooltip = ("A cut already exists at the playhead" if duplicate else
                   "Add cut at playhead (Ctrl+K)")
        self.preanalysis_cut_button.setEnabled(enabled)
        self.preanalysis_cut_button.setToolTip(tooltip)
        if hasattr(self, "segment_cut_button"):
            self.segment_cut_button.setEnabled(enabled)
            self.segment_cut_button.setToolTip(tooltip)
        if hasattr(self, "add_cut_action"):
            self.add_cut_action.setEnabled(enabled)

    def add_cut_at(self, seconds):
        frame = (self.current_media.frame_for_seconds(seconds)
                 if self.current_media else 0)
        ScenespyApp._add_cut_frame(self, frame)

    def _add_cut_frame(self, frame):
        frame = int(frame)
        if not ScenespyApp._can_add_cut_frame(self, frame):
            return
        if 0 < frame < self.current_media.total_frames:
            if hasattr(self, "_record_timeline_edit"):
                self._record_timeline_edit("Add cut marker")
            # A playhead may sit between two frame boundaries at high zoom.
            # Cuts are frame-based, so keep the rebuilt playhead on the exact
            # same boundary as the new marker instead of restoring the former
            # fractional timestamp on either side of it.
            cut_seconds = self.current_media.seconds_for_frame(frame)
            self.timeline.current_seconds = cut_seconds
            existing = sorted(cut.frame for cut in self.current_media.cuts)
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
            if self.selected_segment is not None:
                self.clear_selection()
            self.timeline.rebuild()
            self.timeline.set_position(cut_seconds)
            if hasattr(self, "_request_edit_preview"):
                self._request_edit_preview(cut_seconds)
            if hasattr(self, "_flush_interactive_seek"):
                self._flush_interactive_seek(round(cut_seconds * 1000))
            self._refresh_media_metrics()
            self._refresh_action_button()
            self._refresh_cut_action_state()

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
                       self.segment_cut_button,
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
        self.selected_segment = None
        self._set_selection_available(True)
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
        previous_state = self._capture_timeline_state(media)
        media.faces.append(event)
        self._manual_face_draft = None
        self._set_manual_face_controls(False)
        if not self._apply_face_frame(event, candidate, record_undo=False):
            media.faces = [face for face in media.faces if face.event_id != event.event_id]
            self.show_cut(None)
            return
        self._push_undo_snapshot(previous_state, "Add face")
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
            f"{Path(media.path).stem}_face_{event.best_frame}.png")
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

    def _apply_face_frame(self, event, candidate, record_undo=True):
        if not self.current_media or not event.crop_file:
            return False
        crop_box = [int(value) for value in candidate["crop_box"]]
        candidate_box = [int(value) for value in candidate["box"]]
        if (os.path.isfile(event.crop_file) and
                int(candidate["frame"]) == event.best_frame and
                candidate_box == list(event.box) and
                crop_box == list(event.crop_box)):
            return True
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
        previous_path = Path(event.crop_file)
        target = next_face_edit_path(previous_path, event.event_id)
        if not cv2.imwrite(str(target), crop):
            self.statusBar().showMessage("Could not update the selected face crop.", 5000)
            return False
        if record_undo:
            label = ("Change face frame" if int(candidate["frame"]) != event.best_frame
                     else "Adjust face crop")
            self._record_timeline_edit(label)
        event.crop_file = str(target)
        event.best_frame = int(candidate["frame"])
        event.box = candidate_box
        event.crop_box = crop_box
        self.timeline.invalidate_face_thumbnail(str(previous_path))
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
            if any(cut.original_frame and cut.frame != cut.original_frame
                   for cut in adjacent):
                self._record_timeline_edit("Reset clip boundaries")
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
        if [int(value) for value in crop_box] == list(event.crop_box):
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
            QPixmap.fromImage(image), crop_box, width, height,
            "#00ff00" if event.enabled else DANGER)
        self.viewer_stack.setCurrentWidget(self.face_review_preview)
        self.video.hide()
        self.face_review_preview.show()
        return True

    def show_cut(self, event):
        self.timeline.clear_segment_context()
        if not self.current_media:
            self._set_selection_available(False)
            return
        if not self.current_media.analyzed and not isinstance(event, CutEvent):
            self._set_selection_available(False)
            return
        self._manual_face_draft = None
        self._set_manual_face_controls(False)
        self.selected_cut = event
        self.selected_segment = None
        self._set_selection_available(True)
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
            self.segment_cut_button.setVisible(False)
            self.face_previous_button.setVisible(True)
            self.face_next_button.setVisible(True)
            self.face_reset_button.setVisible(True)
            self.remove_button.setVisible(True)
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
            # Face engine frame numbers are one-based.  The timeline chip's
            # left edge uses the corresponding zero-based media frame, so
            # selection/navigation must use that exact same coordinate.
            seconds = self.current_media.seconds_for_frame(
                face_event_source_frame(event))
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
                f"Visible: {self._player_timecode(self.current_media.seconds_for_frame(event.start_frame), self.current_media.duration)} – "
                f"{self._player_timecode(self.current_media.seconds_for_frame(event.end_frame), self.current_media.duration)}\n"
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
            self.segment_cut_button.setVisible(False)
            self.face_previous_button.setVisible(False)
            self.face_next_button.setVisible(False)
            self.face_reset_button.setVisible(False)
            self.face_save_button.setVisible(False)
            self.remove_button.setVisible(True)
            self.video.face_overlay.clear_face()
            self.face_review_preview.clear_source()
            self.face_review_preview.hide()
            if not self._processing_preview_selected():
                self.viewer_stack.setCurrentWidget(self.video)
                self.video.show()
            seconds = self.current_media.seconds_for_frame(event.frame)
            self.selection_title.setText("Cut marker")
            self.selection_detail.setText(
                f"Frame {event.frame}\nSource: {event.source}")
            self._set_cut_time_editor(event)
            self.remove_button.setIcon(themed_icon("fa6s.trash"))
            self.remove_button.setToolTip("Remove marker")
            self.selection_preview.clear_source()
            self.selection_preview.hide()
        # Position the graphics playhead synchronously at the canonical frame.
        # QMediaPlayer reports integer milliseconds asynchronously and may
        # otherwise introduce a visible sub-frame shift at high zoom levels.
        self.timeline.set_position(seconds)
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
            self._record_timeline_edit(
                "Include clip" if index in self.current_media.excluded_segments
                else "Exclude clip")
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
            self._record_timeline_edit(
                "Include face" if not event.enabled else "Exclude face")
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
        self._record_timeline_edit("Remove cut marker")
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
        if not any(face.event_id == event.event_id
                   for face in self.current_media.faces):
            return
        self._record_timeline_edit("Remove face")
        self.current_media.faces = [
            face for face in self.current_media.faces
            if face.event_id != event.event_id]
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
        self._record_timeline_edit("Remove clip")
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

    @staticmethod
    def _media_is_queue_eligible(media):
        """Return whether a new queue may take ownership of this media."""
        # An analyzed item is an independent project awaiting export.  A new
        # queue must leave its results, edits, and settings untouched.
        return bool(media and not media.analyzed and not media.exported)

    @classmethod
    def _queue_candidates(cls, media_items):
        return [media for media in media_items
                if cls._media_is_queue_eligible(media)]

    @staticmethod
    def _queue_result_preview_remaining(
            ready_at, now=None,
            minimum_ms=QUEUE_RESULT_PREVIEW_MIN_MS):
        if ready_at is None:
            return 0.0
        now = time.monotonic() if now is None else float(now)
        elapsed = max(0.0, now - float(ready_at))
        return max(0.0, float(minimum_ms) / 1000.0 - elapsed)

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

    @staticmethod
    def _ai_installer_command():
        """Return the packaged runtime installer without invoking a shell."""
        if getattr(sys, "frozen", False):
            executable_dir = Path(sys.executable).resolve().parent
        else:
            executable_dir = Path(__file__).resolve().parents[1]
        if sys.platform == "win32":
            candidates = (
                executable_dir / "install_runtime_windows.ps1",
                executable_dir / "scripts" / "install_runtime_windows.ps1",
            )
            script = next((path for path in candidates if path.is_file()), None)
            if script is None:
                return None
            return (
                "powershell.exe",
                ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                 str(script), "-TorchMode", "auto", "-SkipFFmpeg"],
                None,
            )
        candidates = [
            executable_dir / "install_runtime.sh",
            executable_dir / "scripts" / "install_runtime_unix.sh",
        ]
        if sys.platform == "darwin":
            candidates.insert(
                0, executable_dir.parent / "Resources" /
                "scenespy-runtime" / "install_runtime.sh")
        script = next((path for path in candidates if path.is_file()), None)
        if script is None:
            return None
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("TORCH_MODE", "auto")
        environment.insert("SKIP_FFMPEG", "1")
        return "/bin/bash", [str(script)], environment

    def _show_ai_install_result(self, success, detail=""):
        message = QMessageBox(self)
        if success:
            icon = qta.icon(
                "fa6s.circle-check", color="#38c982", scale_factor=.92)
            message.setWindowTitle("AI features installed")
            message.setText(
                "AI face detection is ready.\n"
                "The requested analysis will now continue.")
        else:
            icon = qta.icon(
                "fa6s.triangle-exclamation", color="#f3d45f",
                scale_factor=.92)
            message.setWindowTitle("Could not install AI features")
            message.setText(
                "The AI dependencies could not be installed.\n"
                "Check your internet connection and available disk space, "
                "then try again.")
            if detail:
                summary = detail.strip().splitlines()[-1][:500]
                message.setInformativeText(summary)
                message.setDetailedText(detail[-6000:])
        message.setIconPixmap(icon.pixmap(QSize(38, 38)))
        message.setStandardButtons(QMessageBox.StandardButton.Ok)
        message.exec()

    def _install_face_dependencies(self):
        required_gb = (6 if sys.platform in {"win32", "linux"} and
                       shutil.which("nvidia-smi") else 3)
        try:
            storage_root = Path(runtime_shared.AI_PACK_DIR).parent
            storage_root.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(storage_root).free
        except OSError:
            free_bytes = 0
        required_bytes = required_gb * 1024 ** 3
        if free_bytes and free_bytes < required_bytes:
            available_gb = free_bytes / 1024 ** 3
            self._show_ai_install_result(
                False,
                f"At least {required_gb} GB of free disk space is required. "
                f"Only {available_gb:.1f} GB is currently available.")
            return False
        command = self._ai_installer_command()
        if command is None:
            self._show_ai_install_result(
                False,
                "The packaged dependency installer was not found. "
                "Install the current Scenespy release and try again.")
            return False
        program, arguments, environment = command
        process = QProcess(self)
        self._dependency_install_process = process
        if environment is not None:
            process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        output = []
        process.readyReadStandardOutput.connect(
            lambda: output.append(bytes(
                process.readAllStandardOutput()).decode("utf-8", "replace")))
        result = {"finished": False, "success": False}
        progress = QMessageBox(self)
        progress.setIconPixmap(qta.icon(
            "fa6s.download", color="#4295dd", scale_factor=.92).pixmap(
                QSize(38, 38)))
        progress.setWindowTitle("Installing AI features")
        progress.setText(
            "Downloading and installing AI face detection dependencies.\n"
            "This may take several minutes and require up to 6 GB of disk space.")
        progress.setInformativeText(
            "Scenespy will automatically select the best runtime for this computer.")
        progress.setStandardButtons(QMessageBox.StandardButton.Cancel)

        def finished(exit_code, _exit_status):
            output.append(bytes(
                process.readAllStandardOutput()).decode("utf-8", "replace"))
            result["finished"] = True
            result["success"] = exit_code == 0
            progress.accept()

        def failed_to_start(_error):
            result["finished"] = True
            result["success"] = False
            progress.accept()

        process.finished.connect(finished)
        process.errorOccurred.connect(failed_to_start)
        self.statusBar().showMessage("Installing AI face detection features...")
        process.start(program, arguments)
        progress.exec()
        if process.state() != QProcess.ProcessState.NotRunning:
            process.kill()
            process.waitForFinished(3000)
            self.statusBar().showMessage("AI dependency installation cancelled", 5000)
            self._dependency_install_process = None
            return False
        self._dependency_install_process = None
        if not result["finished"] or not result["success"]:
            self.statusBar().showMessage("AI dependency installation failed", 6000)
            self._show_ai_install_result(False, "".join(output))
            return False
        runtime_shared.refresh_ai_runtime()
        missing = runtime_shared.missing_face_detection_modules()
        if missing:
            detail = "Installer completed, but these modules are unavailable: " + ", ".join(missing)
            self._show_ai_install_result(False, detail + "\n\n" + "".join(output))
            return False
        self.available_inference_accel = detect_available_inference_accel()
        self.available_accel = (self.available_encoder_accel |
                                self.available_inference_accel)
        self._sync_acceleration_options(self.mode.currentData())
        self.statusBar().showMessage("AI face detection features installed", 6000)
        self._show_ai_install_result(True)
        return True

    def _ensure_face_dependencies(self):
        missing = runtime_shared.missing_face_detection_modules()
        if not missing:
            return True
        confirmation = QMessageBox(self)
        warning_icon = qta.icon(
            "fa6s.triangle-exclamation", color="#f3d45f",
            scale_factor=.92)
        confirmation.setIconPixmap(warning_icon.pixmap(QSize(38, 38)))
        confirmation.setWindowTitle("AI features are not installed")
        confirmation.setText(
            "Face detection requires additional AI dependencies.\n"
            "They were not installed with Scenespy and may require up to 6 GB "
            "of disk space.")
        confirmation.setInformativeText(
            "Install them now? Scenespy will select the correct CPU, NVIDIA, "
            "or Apple runtime automatically. Allow up to 3 GB for CPU/Apple "
            "or up to 6 GB for NVIDIA/CUDA.")
        confirmation.setStandardButtons(QMessageBox.StandardButton.Ok)
        install_button = confirmation.addButton(
            "Install dependencies", QMessageBox.ButtonRole.ActionRole)
        install_button.setObjectName("primary")
        confirmation.setDefaultButton(QMessageBox.StandardButton.Ok)
        confirmation.exec()
        if confirmation.clickedButton() is not install_button:
            return False
        return self._install_face_dependencies()

    def analyze(self):
        if not self.current_media or self.task: return
        if self.batch_mode:
            self.process_batch()
            return
        mode = self.mode.currentData()
        if mode == "faces" and not self._ensure_face_dependencies():
            return
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
            pipeline_timings = {}
            cache = os.path.join(USER_DATA_DIR, "cache", media.media_id)
            os.makedirs(cache, exist_ok=True)
            analysis_path = media.path
            proxy_path = os.path.join(cache, "preanalysis_proxy.mp4")
            if not frame_map.is_identity:
                analysis_path = create_analysis_proxy(
                    media.path, proxy_path, frame_map, media.fps,
                    stop_cb=task.isInterruptionRequested,
                    timings=pipeline_timings)

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
                        pipeline_timings=pipeline_timings,
                        face_callback=lambda face: task.face_found.emit(
                            media, self._mapped_face_event(
                                face, frame_map, cache, engine._output_dir)))
                    task.register_cancel(engine.stop)
                    face_complete = False
                    try:
                        if not engine.run():
                            raise RuntimeError(
                                "Face analysis stopped before producing results")
                        events = self._mapped_face_events(
                            engine, frame_map, cache)
                        cuts, excluded, removed = final_timeline(
                            frame_map.join_frames)
                        face_complete = True
                        return ("faces", events, cuts, excluded, removed)
                    finally:
                        if not face_complete:
                            remove_owned_directory(engine._output_dir, cache)
                cfg = PROFILES[media.sensitivity].copy()
                cfg.update(ENCODER=accel, INFERENCE="cpu", DEBUG=False,
                           PIPELINE_TIMINGS=pipeline_timings,
                           ANALYSIS_CACHE_FILE=os.path.join(
                               cache, "scene_analysis_v1.json"))
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
                pass
        self.start_task(
            work, lambda result, target=media: self.analysis_finished(target, result),
            "Analyzing video", preview_media=media,
            done_label="Video analyzed",
            minimum_preview_ms=(900 if mode == "interval" else 0))

    def _process_batch_item(self, task, media, edit_plan, output, mode,
                            sensitivity, interval, accel):
        mapping, excluded_ranges, removed_ranges = edit_plan
        if not mapping.ranges:
            raise RuntimeError("No sequence is included for analysis")
        cache = os.path.join(USER_DATA_DIR, "cache", media.media_id)
        os.makedirs(cache, exist_ok=True)
        proxy_path = os.path.join(cache, "preanalysis_proxy.mp4")
        analysis_path = media.path
        pipeline_timings = {}
        if not mapping.is_identity:
            analysis_path = create_analysis_proxy(
                media.path, proxy_path, mapping, media.fps,
                stop_cb=task.isInterruptionRequested,
                timings=pipeline_timings)
        try:
            previewer = PreviewAdapter(task, mapping, media.fps)
            if mode == "faces":
                engine = FaceDetectionEngine(
                    analysis_path, output, LogAdapter(task), ProgressAdapter(task),
                    previewer=previewer, profile=sensitivity, accel=accel,
                    preview_enabled=True, reset_frames=mapping.join_frames,
                    keep_rejected=True,
                    pipeline_timings=pipeline_timings,
                    face_callback=lambda face: task.face_found.emit(
                        media, self._mapped_face_event(
                            face, mapping, output, engine._output_dir)))
                task.register_cancel(engine.stop)
                face_complete = False
                try:
                    if not engine.run():
                        raise RuntimeError("Face processing did not finish")
                    if task.isInterruptionRequested():
                        raise RuntimeError("Face processing was cancelled")
                    events = self._mapped_face_events(engine, mapping, output)
                    cuts, excluded, removed = self._mapped_timeline(
                        mapping, mapping.join_frames, mode,
                        excluded_ranges, removed_ranges)
                    task.media_analyzed.emit(media)
                    face_complete = True
                    return (media, "faces", events, cuts, excluded, removed,
                            exported_path_size(engine._output_dir))
                finally:
                    if not face_complete:
                        remove_owned_directory(engine._output_dir, output)

            cfg = PROFILES[sensitivity].copy()
            cfg.update(ENCODER=accel, INFERENCE="cpu", DEBUG=False,
                       PIPELINE_TIMINGS=pipeline_timings,
                       ANALYSIS_CACHE_FILE=os.path.join(
                           cache, "scene_analysis_v1.json"))
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
                if finalized:
                    task.media_result_ready_at[media.media_id] = time.monotonic()
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
            pipeline_timings.update(engine._stage_timings)
            if live_timeline:
                cuts, excluded, removed = live_timeline
            else:
                analysis_boundaries = [
                    start for start, _end in engine.analysis_scenes if start > 0]
                cuts, excluded, removed = self._mapped_timeline(
                    mapping, analysis_boundaries, mode,
                    excluded_ranges, removed_ranges)
                task.segments_found.emit(
                    media, (cuts, removed, True))
                task.media_result_ready_at[media.media_id] = time.monotonic()
            boundaries = [0, *(cut.frame for cut in cuts), media.total_frames]
            unavailable = set(excluded) | set(removed)
            numbered_export_segments = [
                (segment_index, segment)
                for segment_index, segment in enumerate(
                    zip(boundaries, boundaries[1:]), 1)
                if segment_index not in unavailable]
            export_numbers = [index for index, _segment in numbered_export_segments]
            export_segments = [segment for _index, segment in numbered_export_segments]
            extensive_export = len(export_segments) > 400
            task.media_analyzed.emit(media)
            if not extensive_export:
                task.cut_preview_ready.clear()
                task.cut_preview_started.emit(media)
                while (not task.isInterruptionRequested() and
                       not task.cut_preview_ready.wait(.025)):
                    pass
            if task.isInterruptionRequested():
                raise RuntimeError("Video processing was cancelled")
            if media.frame_times:
                cfg["FRAME_TIMES"] = media.frame_times
            representative_frame = None
            if extensive_export and export_segments:
                middle_start, middle_end = export_segments[
                    len(export_segments) // 2]
                representative_frame = middle_start + max(
                    0, (middle_end - middle_start - 1) // 2)
                cfg["STATIC_PREVIEW"] = True
            exporter = SceneEngine(
                media.path, output, cfg, LogAdapter(task), ProgressAdapter(task),
                previewer=PreviewAdapter(task), preview_enabled=True,
                preview_frames=(
                    [representative_frame]
                    if representative_frame is not None else [
                        frame
                        for start, end in export_segments
                        for frame in (start, max(start, end - 1))]))
            exporter.scene_mode = mode == "scene"
            exporter.detected = len(export_segments)
            exporter._start_time = time.time()
            task.register_cancel(exporter.stop)
            exporter.start_preview()
            export_complete = False
            try:
                exporter._cut_scenes(export_segments, export_numbers)
                if task.isInterruptionRequested():
                    raise RuntimeError("Video processing was cancelled")
                export_complete = True
            finally:
                exporter.finish_preview(
                    show_final=not task.isInterruptionRequested())
                if not export_complete:
                    remove_owned_directory(exporter._cut_output_dir, output)
            return (media, "cuts", cuts, cuts, excluded, removed,
                    exported_path_size(exporter._cut_output_dir))
        finally:
            pass

    def process_batch(self):
        if self.task:
            return
        if len(self.project.media) < 2:
            self.statusBar().showMessage(
                "A queue requires at least two videos in Media.", 6000)
            return
        media_queue = self._queue_candidates(self.project.media)
        if not media_queue:
            self.statusBar().showMessage(
                "Queue has no new videos to process. Pending exports were left unchanged.",
                5000)
            return
        mode = self.mode.currentData()
        if mode == "faces" and not self._ensure_face_dependencies():
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
        empty_media = [media for media in media_queue
                       if not self._media_has_processable_frames(media)]
        if empty_media:
            self.statusBar().showMessage(
                "Restore at least one timeline sequence in every queued video.",
                6000)
            return
        sensitivity = self.sensitivity.currentText()
        interval = self.interval.value()
        accel = self.accel.currentData() or "cpu"
        self.player.pause()
        self._set_progress_metrics(0, 0, mode, stage="detected")
        self._batch_analyzed_media.clear()
        self._batch_completed_media.clear()
        self._batch_failed_media.clear()
        self._queue_saved_bytes = 0
        self._set_saved_data_summary()
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
            prepared_edit_plans = {}
            for index, media in enumerate(media_queue, 1):
                if task.isInterruptionRequested():
                    break
                task.preview_ready.clear()
                task.media_started.emit(media)
                while (not task.isInterruptionRequested() and
                       not task.preview_ready.wait(.025)):
                    pass
                if task.isInterruptionRequested():
                    break
                task.progress.emit(-1, f"Processing {index}/{total} · {Path(media.path).name}")
                try:
                    # Snapshot this item only when its turn begins, so edits
                    # made while an earlier queue item runs are respected.
                    edit_plan = prepared_edit_plans.pop(
                        media.media_id, None)
                    if edit_plan is None:
                        edit_plan = self._analysis_edit_plan(media)
                    item_result = self._process_batch_item(
                        task, media, edit_plan, output, mode,
                        sensitivity, interval, accel)
                    completed.append(item_result)
                    task.media_finished.emit(media, True, item_result)
                    if mode == "faces":
                        face_events = item_result[2]
                        exportable = [
                            event for event in face_events if event.enabled]
                        exported = sum(
                            1 for event in exportable
                            if os.path.isfile(event.crop_file))
                        completion_label = self._face_export_progress_label(
                            exported, len(exportable))
                    elif mode == "scene":
                        completion_label = "Complete · displaying detected scenes"
                    else:
                        completion_label = "Complete · displaying segments"
                    if index == total:
                        # _finish_task_ui uses this value, so the final queue
                        # item keeps its useful result instead of reverting to
                        # a generic Complete label.
                        task.done_label = completion_label
                    task.progress.emit(1.0, completion_label)
                except Exception as error:
                    if task.isInterruptionRequested():
                        break
                    failures.append((Path(media.path).name, str(error) or type(error).__name__))
                    task.media_finished.emit(media, False, None)
                hold_result = index < total or mode == "interval"
                if hold_result and not task.isInterruptionRequested():
                    # Prepare the immutable edit mapping while the completed
                    # timeline remains on screen.  Actual decoding stays
                    # strictly sequential and begins only after the handoff.
                    if index < total:
                        next_media = media_queue[index]
                        prepared_edit_plans[next_media.media_id] = (
                            self._analysis_edit_plan(next_media))
                    remaining = self._queue_result_preview_remaining(
                        task.media_result_ready_at.get(media.media_id))
                    deadline = time.monotonic() + remaining
                    while (not task.isInterruptionRequested() and
                           time.monotonic() < deadline):
                        time.sleep(min(.025, max(
                            0.0, deadline - time.monotonic())))
            return completed, failures

        self.batch_active = True
        self._refresh_queue_metrics()
        self.start_task(
            work, self.batch_finished, f"Processing queue · {len(media_queue)} videos",
            preview_media=media_queue[0])

    def batch_finished(self, result):
        completed, failures = result
        for media, kind, events, cuts, excluded, removed, item_saved_bytes in completed:
            self._clear_edit_history(media)
            media.analyzed = True
            media.exported = True
            media.cuts = cuts
            media.normalize_cuts()
            media.excluded_segments = excluded
            media.removed_segments = removed
            media.exported_bytes = max(0, int(item_saved_bytes))
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
        self._queue_saved_bytes = sum(
            max(0, int(result[-1])) for result in completed)
        self._set_saved_data_summary(
            self.current_media.exported_bytes
            if self.current_media and self.current_media.exported else None)
        attempted = len(completed) + len(failures)
        summary = f"Queue complete · {len(completed)}/{attempted} videos processed"
        self.statusBar().showMessage(summary, 10000)
        if failures:
            details = "\n".join(f"• {name}: {message}" for name, message in failures[:8])
            QMessageBox.warning(self, "Queue completed with errors", f"{summary}\n\n{details}")

    def analysis_finished(self, media, result):
        self._clear_edit_history(media)
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

    @staticmethod
    def _face_export_progress_label(exported, total):
        return f"Faces exported: {int(exported)} / {int(total)}"

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
        numbered_segments = [
            (index, segment)
            for index, segment in enumerate(media.segments(), 1)
            if index not in unavailable]
        segment_numbers = [index for index, _segment in numbered_segments]
        segments = [segment for _index, segment in numbered_segments]
        if not segments:
            self.statusBar().showMessage(
                "Include at least one sequence before exporting.", 5000)
            return
        accel = self.accel.currentData() or "cpu"
        if media.mode == "faces":
            initial_total = sum(
                1 for event in media.faces
                if (face_event_is_included(media, event) and event.enabled and
                    os.path.isfile(event.crop_file)))
            export_progress_label = self._face_export_progress_label(
                0, initial_total)
        else:
            noun = "Scenes" if media.mode == "scene" else "Segments"
            export_progress_label = f"{noun} cut: 0 / {len(segments)}"
        self._refresh_media_metrics()
        self._set_progress_metrics(
            getattr(self, "_metric_detected", None), 0, media.mode,
            stage="cut")
        def work(task):
            if media.mode == "faces":
                destination = next_output_path(
                    folder, "faces", media.sensitivity, accel)
                staging = os.path.join(
                    folder,
                    f".{os.path.basename(destination)}.{uuid.uuid4().hex}.part")
                os.makedirs(staging, exist_ok=False)
                task.register_cleanup(
                    lambda path=staging, root=folder:
                    remove_owned_directory(path, root))
                saved = 0
                exported_face_files = {}
                available_faces = [event for event in media.faces
                                   if face_event_is_included(media, event)]
                export_faces = [event for event in available_faces
                                if event.enabled and os.path.isfile(event.crop_file)]
                task.metrics.emit(len(available_faces), saved)
                total_faces = len(export_faces)
                task.progress.emit(
                    0.0, self._face_export_progress_label(0, total_faces))
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
                                source_frame = face_event_source_frame(event)
                                preview_cap.set(
                                    cv2.CAP_PROP_POS_FRAMES, source_frame)
                                ok, frame = preview_cap.read()
                                if ok and frame is not None:
                                    if len(event.box or []) == 4:
                                        x1, y1, x2, y2 = map(int, event.box)
                                        draw_face_marker(
                                            frame, (x1, y1, x2, y2))
                                    preview = resize_for_preview(Image.fromarray(
                                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                                if preview is None:
                                    with Image.open(event.crop_file) as source:
                                        preview = resize_for_preview(
                                            source.convert("RGB"))
                                if preview:
                                    task.preview.emit(
                                        preview,
                                        media.seconds_for_frame(source_frame))
                            except Exception:
                                pass
                        face_number = saved + 1
                        filename = f"face_{face_number}.png"
                        target = os.path.join(staging, filename)
                        shutil.copy2(event.crop_file, target)
                        exported_face_files[event.event_id] = (
                            face_number, filename)
                        saved += 1
                        task.metrics.emit(len(available_faces), saved)
                        task.progress.emit(
                            saved / max(1, total_faces),
                            self._face_export_progress_label(saved, total_faces))
                finally:
                    if preview_cap is not None:
                        preview_cap.release()
                if not export_faces:
                    task.progress.emit(
                        1.0, self._face_export_progress_label(0, 0))
                try:
                    if task.isInterruptionRequested():
                        return None
                    face_records = []
                    for event in available_faces:
                        record = vars(event).copy()
                        exported = exported_face_files.get(event.event_id)
                        record["face_number"] = exported[0] if exported else None
                        record["exported_file"] = exported[1] if exported else ""
                        if exported:
                            exported_path = os.path.join(staging, exported[1])
                            image = cv2.imread(exported_path)
                            record["file_size_bytes"] = os.path.getsize(exported_path)
                            record["image_width"] = (
                                int(image.shape[1]) if image is not None else None)
                            record["image_height"] = (
                                int(image.shape[0]) if image is not None else None)
                        else:
                            record["file_size_bytes"] = None
                            record["image_width"] = None
                            record["image_height"] = None
                        face_records.append(record)
                    exported_records = [
                        record for record in face_records
                        if record["exported_file"]]
                    files_present = all(
                        os.path.isfile(os.path.join(
                            staging, record["exported_file"]))
                        for record in exported_records)
                    numbers_contiguous = [
                        record["face_number"] for record in exported_records
                    ] == list(range(1, saved + 1))
                    filenames_unique = len({
                        record["exported_file"] for record in exported_records
                    }) == len(exported_records)
                    valid_frames = all(
                        1 <= int(record["start_frame"]) <=
                        int(record["best_frame"]) <=
                        int(record["end_frame"])
                        for record in face_records)
                    folder_name = os.path.basename(destination)
                    folder_name_valid = valid_faces_output_name(folder_name)
                    valid_crop_coordinates = all(
                        len(record.get("crop_box") or ()) == 4 and
                        0 <= int(record["crop_box"][0]) < int(record["crop_box"][2]) <= media.width and
                        0 <= int(record["crop_box"][1]) < int(record["crop_box"][3]) <= media.height
                        for record in face_records)
                    dimensions_match = all(
                        record["image_width"] ==
                        int(record["crop_box"][2]) - int(record["crop_box"][0]) and
                        record["image_height"] ==
                        int(record["crop_box"][3]) - int(record["crop_box"][1])
                        for record in exported_records)
                    issues = []
                    if not files_present:
                        issues.append("One or more exported face files are missing")
                    if not numbers_contiguous:
                        issues.append("Exported face numbering is not continuous from 1")
                    if not filenames_unique:
                        issues.append("Duplicate face filenames were generated")
                    if not valid_frames:
                        issues.append("One or more face frame ranges are invalid")
                    if not folder_name_valid:
                        issues.append(
                            "Output folder name does not follow the FD timestamp/profile/acceleration standard")
                    if not valid_crop_coordinates:
                        issues.append("One or more face crop coordinates are invalid")
                    if not dimensions_match:
                        issues.append("One or more exported images do not match their crop dimensions")
                    payload = {
                        "summary": {
                            "status": "completed" if not issues else "completed_with_errors",
                            "mode": "faces_reviewed",
                            "faces_available": len(available_faces),
                            "faces_exported": saved,
                            "faces_not_exported": len(available_faces) - saved,
                            "source_duration": round(float(media.duration), 6),
                            "audit_passed": not issues,
                            "issues": issues,
                        },
                        "faces": face_records,
                        "audit": {
                            "exported_files_present": files_present,
                            "face_numbers_start_at_one_and_are_contiguous": numbers_contiguous,
                            "filenames_unique": filenames_unique,
                            "frame_ranges_valid": valid_frames,
                            "output_folder_name_valid": folder_name_valid,
                            "expected_output_folder_pattern": (
                                "FD_YYYYMMDD_HHMMSS_Sensitivity_acceleration"
                            ),
                            "crop_coordinates_valid": valid_crop_coordinates,
                            "exported_dimensions_match_crop": dimensions_match,
                            "frame_number_convention": (
                                "Face frame fields are one-based; subtract 1 for zero-based media coordinates."
                            ),
                        },
                        "source": {
                            "file": Path(media.path).name,
                            "path": media.path,
                            "duration": round(float(media.duration), 6),
                            "fps": round(float(media.fps), 6),
                            "total_frames": int(media.total_frames),
                        },
                        "output": {"folder": destination},
                    }
                    with open(os.path.join(staging, "faces.json"), "w", encoding="utf-8") as handle:
                        json.dump(payload, handle, indent=2)
                    backup = None
                    if os.path.exists(destination):
                        backup = f"{destination}.{uuid.uuid4().hex}.backup"
                        os.replace(destination, backup)
                    try:
                        os.replace(staging, destination)
                    except Exception:
                        if backup and os.path.exists(backup):
                            os.replace(backup, destination)
                        raise
                    if backup:
                        remove_owned_directory(backup, folder)
                    return destination, exported_path_size(destination)
                finally:
                    remove_owned_directory(staging, folder)
            cfg = PROFILES[media.sensitivity].copy()
            cfg.update(ENCODER=accel, INFERENCE="cpu", DEBUG=False)
            if media.frame_times:
                cfg["FRAME_TIMES"] = media.frame_times
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
            export_complete = False
            try:
                engine._cut_scenes(segments, segment_numbers)
                if task.isInterruptionRequested():
                    return None
                export_complete = True
                return (engine._cut_output_dir,
                        exported_path_size(engine._cut_output_dir))
            finally:
                engine.finish_preview(
                    show_final=not task.isInterruptionRequested())
                if not export_complete:
                    remove_owned_directory(engine._cut_output_dir, folder)
        self.start_task(
            work, lambda output, target=media: self.export_finished(target, output),
            export_progress_label, preview_media=media, minimum_preview_ms=800,
            done_label=(
                self._face_export_progress_label(initial_total, initial_total)
                if media.mode == "faces" else "Complete"))

    def export_finished(self, media, output):
        saved_bytes = None
        if isinstance(output, tuple):
            output, saved_bytes = output
        media.exported = True
        media.exported_bytes = max(0, int(saved_bytes or 0))
        self._set_saved_data_summary(saved_bytes)
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
        self._pause_playback_proxies()
        self._end_edit_preview()
        self._release_frame_decoder()
        self._set_working_ui(True)
        self._set_saved_data_summary()
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
        task.cut_preview_started.connect(self._prepare_queue_cut_preview)
        task.result.connect(lambda result: self.task_done(task, callback, result))
        task.failed.connect(lambda message: self.task_failed(task, message))
        task.cancelled.connect(lambda: self.task_cancelled(task))
        if preview_media:
            self._begin_processing_preview(preview_media)
        task.start()

    def toggle_primary_action(self):
        if self.task:
            self.stop_current_task()
        elif (self.current_media and self.current_media.analyzed and
              not self.current_media.exported):
            self.export_result()
        elif self.batch_mode:
            self.process_batch()
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
        if self.current_media and self.current_media.exported:
            label = "Finished"
            enabled = False
            button_style = "success"
        elif (self.current_media and self.current_media.analyzed and
              not self.current_media.exported):
            # Pending results stay independent from automatic queue mode.
            label = "Export"
            enabled = self._media_has_processable_frames(self.current_media)
            if not enabled:
                disabled_reason = (
                    "Restore at least one timeline sequence before exporting.")
            button_style = "warning"
        elif (self.batch_mode and
              self._media_is_queue_eligible(self.current_media)):
            label = "Process queue"
            queue_media = self._queue_candidates(self.project.media)
            has_content = bool(queue_media) and all(
                self._media_has_processable_frames(media)
                for media in queue_media)
            enabled = has_content
            if not queue_media:
                disabled_reason = (
                    "No new videos to process. Pending exports are not part of the queue.")
            elif not has_content:
                disabled_reason = (
                    "Restore at least one timeline sequence in every queued video.")
            elif not (self.project.output_folder and
                      os.path.isdir(self.project.output_folder)):
                disabled_reason = "Choose the queue output folder when starting."
            button_style = "primary"
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
        QTimer.singleShot(0, self._resume_deferred_playback_proxy)

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

    def _set_saved_data_summary(self, byte_count=None):
        if not hasattr(self, "saved_data_summary"):
            return
        if byte_count is None:
            self.saved_data_summary.clear()
            self.saved_data_summary.hide()
            return
        self.saved_data_summary.setText(
            f"Saved: {format_file_size(max(0, int(byte_count)))}")
        self.saved_data_summary.show()

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
        self._invalidate_playback_face_index()
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
                # Progressive Scene Detection updates remain lightweight.
                # Final layouts use the background thumbnail decoder shared
                # with normal mode, so the complete indicators remain cheap.
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
            if self.task:
                self.task.preview_ready.set()
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
            try:
                row = next(
                    index for index, item in enumerate(self.project.media)
                    if item is media)
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
        finally:
            if self.task:
                self.task.preview_ready.set()

    def batch_media_finished(self, media, succeeded, result=None):
        if succeeded and result:
            self._clear_edit_history(media)
            _media, kind, events, cuts, excluded, removed, saved_bytes = result
            media.analyzed = True
            media.exported = True
            media.cuts = list(cuts)
            media.normalize_cuts()
            media.excluded_segments = list(excluded)
            media.removed_segments = list(removed)
            if kind == "faces":
                media.faces = list(events)
            media.exported_bytes = max(0, int(saved_bytes))
        if succeeded:
            self._batch_analyzed_media.discard(media.media_id)
            self._batch_failed_media.discard(media.media_id)
            self._batch_completed_media.add(media.media_id)
            self._queue_saved_bytes += max(0, int(saved_bytes))
            self._set_saved_data_summary(self._queue_saved_bytes)
        else:
            # A failed queue item always owns the red error state, regardless
            # of how far it progressed before the failure.
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

    def _prepare_queue_cut_preview(self, media):
        """Rewind the finalized queue timeline before its export preview."""
        task = self.task
        if not task:
            return
        try:
            if (self.batch_active and media is self.processing_media and
                    self._processing_preview_selected()):
                self.processing_position_animation.stop()
                self._set_progress_value(0, immediate=True)
                unavailable = (set(media.excluded_segments) |
                               set(media.removed_segments))
                total = sum(
                    1 for index, _segment in enumerate(media.segments(), 1)
                    if index not in unavailable)
                noun = "Scenes" if media.mode == "scene" else "Segments"
                label = f"{noun} cut: 0 / {total}"
                task.media_progress[media.media_id] = (0, label)
                self.progress_label.setText(label)
                self._sync_processing_position(0.0)
        finally:
            task.cut_preview_ready.set()

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
            if item_media.media_id in self._batch_failed_media:
                item.setIcon(status_indicator_icon(
                    "fa6s.circle-xmark", STATUS_ERROR_COLOR))
                item.setToolTip(f"Process failed\n{item_media.path}")
                item.setForeground(QColor("#f2a4aa"))
            elif item_media.exported or item_media.media_id in self._batch_completed_media:
                item.setIcon(status_indicator_icon(
                    "fa6s.circle-check", STATUS_SUCCESS_COLOR))
                item.setToolTip(f"Exported\n{item_media.path}")
                item.setForeground(QColor("#b8e8cf"))
            elif item_media is self.processing_media:
                # Queue analysis and export are one continuous process.  Keep
                # the active item white even after its analysis checkpoint.
                item.setIcon(status_indicator_icon(
                    "fa6s.circle-arrow-right", STATUS_PROCESS_COLOR))
                item.setToolTip(f"Processing now\n{item_media.path}")
                item.setForeground(QColor("#ffffff"))
            elif (item_media.analyzed or
                  item_media.media_id in self._batch_analyzed_media):
                item.setIcon(status_indicator_icon(
                    "fa6s.circle-info", STATUS_PENDING_COLOR))
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
        spec = None
        if status == "Failed":
            spec = ("fa6s.circle-xmark", STATUS_ERROR_COLOR, "Process failed")
        elif status == "Stopping":
            spec = ("fa6s.circle-stop", STATUS_PROCESS_COLOR, "Stopping process")
        elif media and self._current_media_is_processing():
            spec = ("fa6s.circle-arrow-right", STATUS_PROCESS_COLOR, "Processing now")
        elif media and media.exported and status == "Complete":
            spec = ("fa6s.circle-check", STATUS_SUCCESS_COLOR, "Exported")
        elif (media and media.analyzed and not media.exported and
              status == "Video analyzed"):
            spec = ("fa6s.circle-info", STATUS_PENDING_COLOR,
                    "Analyzed · ready to export")
        if spec is None:
            self.progress_state_icon.clear()
            self.progress_state_icon.setToolTip("")
            self.progress_state_icon.hide()
            return
        name, color, tooltip = spec
        icon = status_indicator_icon(name, color)
        self.progress_state_icon.setPixmap(icon.pixmap(STATUS_ICON_SIZE))
        self.progress_state_icon.setToolTip(
            f"{tooltip}\n{media.path}" if media else tooltip)
        self.progress_state_icon.show()

    def task_done(self, task, callback, result):
        if task.isRunning():
            task.finished.connect(
                lambda t=task, cb=callback, value=result:
                self.task_done(t, cb, value))
            return

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
        if task.isRunning():
            task.finished.connect(
                lambda t=task, detail=message:
                self.task_failed(t, detail))
            return
        self._batch_analyzed_media.clear()
        self._batch_completed_media.clear()
        failed_media = self.processing_media
        if failed_media:
            # Keep normal-mode failures visible in Media just like failures
            # belonging to an automatic queue.
            self._batch_failed_media.add(failed_media.media_id)
        self._finish_task_ui(task, "Failed")
        if self.current_media:
            self.timeline.rebuild()
        self._refresh_media_metrics()
        if self.current_media is not failed_media:
            self._restore_selected_media_progress()
        QMessageBox.critical(self, "Scenespy error", message)
        task.deleteLater()

    def task_cancelled(self, task):
        if task.isRunning():
            task.finished.connect(
                lambda t=task: self.task_cancelled(t))
            return
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
                queued_ids = set(getattr(self, "_queue_media_ids", ()))
                for media in saved_media:
                    if media.media_id in queued_ids and not media.exported:
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

    def _check_for_updates(self):
        if self._update_check_task is not None:
            return
        checker = UpdateCheckTask(self)
        self._update_check_task = checker
        checker.updateAvailable.connect(self._offer_update)

        def cleanup():
            self._update_check_task = None
            checker.deleteLater()

        checker.finished.connect(cleanup)
        checker.start()

    def _offer_update(self, release):
        version = str(release.get("version") or "")
        if not version or self._update_prompted_version == version:
            return
        self._update_prompted_version = version
        message = QMessageBox(self)
        icon = qta.icon(
            "fa6s.arrows-rotate", color="#4295dd", scale_factor=.92)
        message.setIconPixmap(icon.pixmap(QSize(38, 38)))
        message.setWindowTitle("Scenespy update available")
        message.setText(
            f"Scenespy {version} is available.\n"
            f"You are currently using version {__version__}.")
        size = max(0, int(release.get("size") or 0))
        size_text = f" ({size / 1024 ** 2:.1f} MB)" if size else ""
        message.setInformativeText(
            "Download the verified installer from GitHub Releases" +
            size_text + "?")
        notes = str(release.get("notes") or "").strip()
        if notes:
            message.setDetailedText(notes[:12000])
        message.setStandardButtons(QMessageBox.StandardButton.Cancel)
        later_button = message.button(QMessageBox.StandardButton.Cancel)
        later_button.setText("Later")
        download_button = message.addButton(
            "Download update", QMessageBox.ButtonRole.ActionRole)
        download_button.setObjectName("primary")
        message.setDefaultButton(later_button)
        message.exec()
        if message.clickedButton() is download_button:
            self._download_update(release)

    def _download_update(self, release):
        if self._update_download_task is not None:
            return
        downloader = UpdateDownloadTask(release, self)
        self._update_download_task = downloader
        result = {"path": "", "error": ""}
        progress = QMessageBox(self)
        progress.setIconPixmap(qta.icon(
            "fa6s.cloud-arrow-down", color="#4295dd",
            scale_factor=.92).pixmap(QSize(38, 38)))
        progress.setWindowTitle("Downloading Scenespy update")
        progress.setText(f"Downloading Scenespy {release['version']} from GitHub...")
        progress.setInformativeText("Progress: 0% · SHA-256 verification will follow.")
        progress.setStandardButtons(QMessageBox.StandardButton.Cancel)

        def update_progress(value):
            progress.setInformativeText(
                f"Progress: {int(value)}% · SHA-256 verification will follow.")

        def completed(path):
            result["path"] = path
            progress.accept()

        def failed(message):
            result["error"] = message
            progress.accept()

        downloader.progress.connect(update_progress)
        downloader.completed.connect(completed)
        downloader.failed.connect(failed)
        downloader.start()
        progress.exec()
        if downloader.isRunning() and not result["path"] and not result["error"]:
            downloader.requestInterruption()
            downloader.wait(5000)
            self.statusBar().showMessage("Update download cancelled", 5000)
        else:
            downloader.wait(1000)
        self._update_download_task = None
        downloader.deleteLater()
        if result["error"]:
            QMessageBox.warning(
                self, "Could not download update",
                "Scenespy could not download or verify the update.\n\n" +
                result["error"])
            return
        if not result["path"]:
            return
        self._launch_update_installer(result["path"])

    def _launch_update_installer(self, path):
        path = str(Path(path).resolve())
        if sys.platform == "win32":
            launched = QProcess.startDetached(path, [])
        elif sys.platform == "darwin":
            launched = QProcess.startDetached("/usr/bin/open", [path])
        else:
            # The Linux release is a Debian package; hand it to the desktop's
            # package/software application instead of treating it as a script.
            launched = QProcess.startDetached("xdg-open", [path])
        if isinstance(launched, tuple):
            launched = launched[0]
        if not launched:
            QMessageBox.warning(
                self, "Could not start update",
                "The verified installer was downloaded, but it could not be opened.\n\n" +
                path)
            return
        self.statusBar().showMessage("Update installer started")
        QApplication.instance().quit()

    def closeEvent(self, event):
        if self._shutdown_started:
            event.accept()
            return
        if self.task and self.task.isRunning():
            answer = QMessageBox.question(
                self, "Close Scenespy",
                "A process is still running. Stop it and close Scenespy?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._shutdown_started = True
            task = self.task
            task.cancel()
            terminate_all_child_processes(grace_seconds=.5)
            if not task.wait(7000):
                terminate_all_child_processes(grace_seconds=.1)
                if not task.wait(2000):
                    # Final exit-only safeguard for a stuck third-party decoder.
                    task.terminate()
                    task.wait(1000)
            task.run_cleanups()
        else:
            self._shutdown_started = True
        for background_task in (
                self._update_check_task, self._update_download_task):
            if background_task is not None and background_task.isRunning():
                background_task.requestInterruption()
                background_task.wait(11000)
        proxy_tasks = list(self._playback_proxy_tasks.values())
        for proxy_task in proxy_tasks:
            if proxy_task.isRunning():
                proxy_task.cancel()
        if proxy_tasks:
            terminate_all_child_processes(grace_seconds=.2)
            for proxy_task in proxy_tasks:
                if not proxy_task.wait(3000):
                    proxy_task.terminate()
                    proxy_task.wait(1000)
                proxy_task.run_cleanups()
        self._playback_proxy_tasks.clear()
        self._deferred_playback_proxy_ids.clear()
        self._interactive_seek_timer.stop()
        self.processing_position_animation.stop()
        self._end_edit_preview()
        self._end_processing_preview()
        self._release_frame_decoder(force=True)
        if not self.timeline.stop_thumbnail_decoder(timeout_ms=3000):
            thumbnail_decoder = self.timeline._clip_thumbnail_decoder
            if thumbnail_decoder is not None:
                thumbnail_decoder.terminate()
                thumbnail_decoder.wait(1000)
                self.timeline._clip_thumbnail_decoder = None
        self.player.stop()
        self.player.setSource(QUrl())
        self.processing_preview.clear_source()
        self.selection_preview.clear_source()
        self.face_review_preview.clear_source()
        self.timeline.set_media(None)
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
