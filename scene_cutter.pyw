# ============================================================
# Imports: stdlib
# ============================================================
import sys
import os
import socket
import subprocess
import threading
import time
import datetime
import json
import gc
import bisect
import textwrap
import traceback
import faulthandler
import tkinter.filedialog as fd
import tkinter.messagebox as mb
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Imports: third-party (lightweight / always needed)
# ============================================================
import customtkinter as ctk
from PIL import Image
import numpy as np

# ============================================================
# Imports: third-party (heavy / always used by engines)
# ============================================================
import cv2

# ============================================================
# Lazy import: torch (deferred until face engine starts)
# ============================================================
torch = None
TORCH_AVAILABLE = False

# ============================================================
# Config: user settings persistence
# ============================================================
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
CRASH_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_cutter_crash.log")
_CRASH_LOG_HANDLE = None


def log_crash(message):
    try:
        with open(CRASH_LOG_FILE, "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            f.write(f"\n[{ts}] {message}\n")
    except Exception:
        pass


def install_crash_logging():
    global _CRASH_LOG_HANDLE
    try:
        _CRASH_LOG_HANDLE = open(CRASH_LOG_FILE, "a", encoding="utf-8")
        faulthandler.enable(file=_CRASH_LOG_HANDLE, all_threads=True)
    except Exception:
        pass

    def _sys_hook(exc_type, exc, tb):
        log_crash("Unhandled exception:\n" + "".join(traceback.format_exception(exc_type, exc, tb)))

    def _thread_hook(args):
        log_crash("Unhandled thread exception:\n" + "".join(
            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook


def load_settings():
    """Load user settings from JSON file."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"last_video": "", "last_output": ""}


def save_settings(video="", output=""):
    """Save user settings to JSON file."""
    try:
        settings = {"last_video": video, "last_output": output}
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


# ============================================================
# Config: profiles
# ============================================================
# Hybrid detection: FIXED threshold + ADAPTIVE min_dur.
#
# Thresholds are fixed per profile (PySceneDetect content_val scale),
# validated against real-world trailer/film/documentary tests:
#   Low    (42) -> conservative, ASL ~5-10s (documentaries/films)
#   Normal (27) -> balanced, ASL ~2-6s (movies/series/TV/web)
#   High   (18) -> sensitive, ASL ~0.7-3s (trailers/shorts/action)
#
# min_dur: profile sets the FLOOR (5s/2s/0.7s), adaptive scan can only
# raise it within a bounded range.
# ============================================================
PROFILES = {
    "Low": {"label": "Low", "base_threshold": 42, "min_dur": 5.0, "dur_max_boost": 5.0},
    "Normal": {"label": "Normal", "base_threshold": 27, "min_dur": 2.0, "dur_max_boost": 4.0},
    "High": {"label": "High", "base_threshold": 18, "min_dur": 0.7, "dur_max_boost": 2.5},
    "Auto": {"label": "Auto", "ADAPTIVE": True},
}

# Profile display names (for output folder tagging)
PROFILE_LABELS = {k: v["label"] for k, v in PROFILES.items()}

ACCEL_OPTIONS = ["cpu", "nvidia", "amd", "intel"]
# Cut workers: 2 for speed with many scenes, balanced I/O usage.
# Using stream copy (-c copy) where possible ensures lossless output.
MAX_CUT_WORKERS = 2
ENABLE_PREVIEW_DEFAULT = True
PREVIEW_INTERVAL = 0.2
PREVIEW_FPS = 2
INSTANCE_SOCKET = None
INSTANCE_PORT = 54321
PREVIEW_MAX_WIDTH = 420
PREVIEW_MAX_HEIGHT = 240
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
PROCESS_COOLDOWN_SECONDS = 1.25

# ============================================================
# Config: compatibility / constants
# ============================================================
MODE_ACCEL_COMPAT = {
    "scene": {"encoder": {"cpu", "nvidia", "amd", "intel"}, "inference": {"cpu"}},
    "interval": {"encoder": {"cpu", "nvidia", "amd", "intel"}, "inference": {"cpu"}},
    "faces": {"encoder": {"cpu"}, "inference": {"cpu", "nvidia"}},
}

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

MODE_ABBREV = {"faces": "FD", "scene": "SD", "interval": "ES"}

# ============================================================
# Debug configuration
# ============================================================
DEBUG = False  # Set to True to enable debug logging to console

# ============================================================
# Config: UI theme palette
# ============================================================
BG_MAIN = "#1a1a1a"
BG_PANEL = "#313131"
BG_CARD = "#404040"
BG_INPUT = "#1a1a1a"
BORDER_SOFT = "#787474"
BORDER_SOFT2 = "#4C4848"
TEXT_MAIN = "#e5e7eb"
TEXT_MUTED = "#9ca3af"
ACCENT = "#1f538d"
SUCCESS = "#22c55e"
DANGER = "#ef4444"


# ============================================================
# Utility: helpers used across the app
# ============================================================
def test_ffmpeg_encoder(encoder: str) -> bool:
    try:
        result = run_hidden(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             "color=c=black:s=160x120:d=0.1",
             "-c:v", encoder, "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def detect_available_accel():
    available = {"cpu"}
    try:
        if _ensure_torch() and torch.cuda.is_available():
            available.add("nvidia")
    except Exception:
        pass
    if test_ffmpeg_encoder("h264_amf"):
        available.add("amd")
    if test_ffmpeg_encoder("h264_qsv"):
        available.add("intel")
    return available


def build_output_dir(base_output, mode, profile, accel):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    mode_tag = MODE_ABBREV.get(mode, mode.upper())
    path = os.path.join(base_output, f"{mode_tag}_{ts}_{profile}_{accel}")
    os.makedirs(path, exist_ok=True)
    return path


def is_valid_video_file(path: str) -> bool:
    try:
        out = check_output_hidden(
            ["ffprobe", "-v", "error", "-show_streams",
             "-select_streams", "v", "-of", "json", path],
            stderr=subprocess.DEVNULL
        )
        data = json.loads(out)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and s.get("codec_name", "").lower() != "gif":
                return True
    except Exception:
        pass
    return False


def remove_temp_file(path):
    if not path:
        return
    for _ in range(20):
        try:
            if not os.path.exists(path):
                return
            os.remove(path)
            return
        except Exception:
            gc.collect()
            time.sleep(0.15)


def remux_if_needed(path, temp_files=None):
    ext = os.path.splitext(path)[1].lower()
    if ext != ".mkv":
        return path

    fixed = path[:-4] + "_fixed.mkv"
    remove_temp_file(fixed)

    run_hidden(
        ["ffmpeg", "-y", "-fflags", "+genpts+igndts", "-err_detect", "ignore_err",
         "-i", path, "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
         "-max_interleave_delta", "0", "-avoid_negative_ts", "make_zero", fixed],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if os.path.exists(fixed) and is_valid_video_file(fixed):
        if temp_files is not None:
            temp_files.append(fixed)
        return fixed
    remove_temp_file(fixed)
    return path


def video_decode_error_summary(path, timeout=120):
    try:
        result = run_hidden(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "NUL"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=timeout
        )
        if result.returncode == 0 and not result.stderr.strip():
            return ""
        return (result.stderr or "FFmpeg decode check failed").strip()
    except Exception as e:
        return str(e)


def reencode_fixed_video(path, temp_files=None, timeout=300):
    fixed = path.rsplit(".", 1)[0] + "_fixed.mp4"
    remove_temp_file(fixed)
    result = run_hidden(
        ["ffmpeg", "-y", "-err_detect", "ignore_err",
         "-i", path, "-c:v", "libx264", "-crf", "22",
         "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "copy", fixed],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        timeout=timeout
    )
    if result.returncode == 0 and os.path.exists(fixed) and is_valid_video_file(fixed):
        if temp_files is not None:
            temp_files.append(fixed)
        return fixed
    remove_temp_file(fixed)
    err = result.stderr.decode(errors="ignore") if isinstance(result.stderr, bytes) else str(result.stderr or "")
    raise RuntimeError(err[:300] or "FFmpeg could not repair the video")


def prepare_video_for_processing(path, temp_files=None):
    prepared = remux_if_needed(path, temp_files=temp_files)
    decode_error = video_decode_error_summary(prepared)
    if not decode_error:
        return prepared
    return reencode_fixed_video(prepared, temp_files=temp_files)


def resize_for_preview(img, max_w=PREVIEW_MAX_WIDTH, max_h=PREVIEW_MAX_HEIGHT):
    w, h = img.size
    if w <= 0 or h <= 0:
        return None
    scale = min(max_w / w, max_h / h)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


def _safe_frame_index(v):
    if isinstance(v, (int, float)):
        return int(v)
    if hasattr(v, "get_frames"):
        return v.get_frames()
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return int(v.flat[0])
    except Exception:
        pass
    return 0


def _ensure_torch():
    """Lazy-load torch / YOLO on first call."""
    global torch, TORCH_AVAILABLE
    if torch is not None:
        return TORCH_AVAILABLE
    try:
        import torch as _t
        torch = _t
        TORCH_AVAILABLE = True
    except Exception:
        torch = None
        TORCH_AVAILABLE = False
    return TORCH_AVAILABLE


def _ensure_yolo():
    """Lazy-load ultralytics.YOLO on first call."""
    try:
        from ultralytics import YOLO
        return YOLO
    except Exception:
        return None


def _ensure_mediapipe():
    """Lazy-load mediapipe on first call."""
    try:
        import mediapipe as mp
        return mp
    except Exception:
        return None


def _ensure_scenedetect():
    """Lazy-load scenedetect modules on first call."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        from scenedetect.stats_manager import StatsManager
        return open_video, SceneManager, ContentDetector, StatsManager, ContentDetector.Components
    except Exception:
        return None


def single_instance():
    global INSTANCE_SOCKET
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", INSTANCE_PORT))
        s.listen(1)
        INSTANCE_SOCKET = s
    except OSError:
        return False
    return True


def run_hidden(cmd, **kwargs):
    return subprocess.run(
        cmd,
        creationflags=CREATE_NO_WINDOW,
        **kwargs
    )


def check_output_hidden(cmd, **kwargs):
    return subprocess.check_output(
        cmd,
        creationflags=CREATE_NO_WINDOW,
        **kwargs
    )


# ============================================================
# Widgets
# ============================================================
class Section(ctk.CTkFrame):
    def __init__(self, master, title, **kwargs):
        super().__init__(master, fg_color=BG_CARD, border_width=1,
                         border_color=BORDER_SOFT2, corner_radius=0, **kwargs)
        ctk.CTkLabel(self, text=title, font=("Consolas", 14, "bold")
                     ).pack(anchor="w", padx=12, pady=(8, 4))


class LabeledEntry(ctk.CTkFrame):
    def __init__(self, master, label, placeholder="", width=160):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, font=("Consolas", 12)).pack(anchor="w")
        self.entry = ctk.CTkEntry(
            width=width, corner_radius=15, fg_color=BG_MAIN,
            border_width=1, border_color=BORDER_SOFT,
            text_color=TEXT_MAIN, placeholder_text_color=TEXT_MUTED,
            placeholder_text=placeholder
        )
        self.entry.pack(pady=(2, 8), fill="x")

    def get(self):
        return self.entry.get()


class LogBox(ctk.CTkTextbox):
    def __init__(self, master, height=140):
        super().__init__(master, height=height, fg_color=BG_MAIN,
                         corner_radius=15, border_color=BORDER_SOFT2, border_width=1)
        self.configure(state="disabled", font=("Consolas", 12), wrap="char")
        self.pack_propagate(False)
        self.status_lines = []
        self.initialized = False

    def _terminal_wrap_width(self):
        width_px = self.winfo_width()
        if width_px <= 1:
            width_px = 640
        return max(42, min(96, int(width_px / 8)))

    def _format_terminal_message(self, text, max_chars=900):
        cleaned = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars].rstrip() + "..."
        return "\n".join(textwrap.wrap(
            cleaned,
            width=self._terminal_wrap_width(),
            break_long_words=True,
            break_on_hyphens=False,
        )) or "Unknown error"

    def _ensure_initialized(self):
        if self.initialized:
            return
        self.delete("1.0", "end")
        self.status_lines = [
            "Scenes detected: -",
            "Scenes cut:      -",
            "Estimated time:  --:--"
        ]
        for line in self.status_lines:
            self.insert("end", line + "\n")
        self.initialized = True

    def write_status(self, detected=None, cut=None, eta=None):
        lines = [
            f"Scenes detected: {detected if detected is not None else '-'}",
            f"Scenes cut:      {cut if cut is not None else '-'}",
            f"Estimated time:  {eta if eta is not None else '--:--'}"
        ]
        self.configure(state="normal")
        if not self.initialized:
            self.delete("1.0", "end")
            self.status_lines = lines
            for line in lines:
                self.insert("end", line + "\n")
            self.initialized = True
        else:
            for i, line in enumerate(lines):
                self.delete(f"{i + 1}.0", f"{i + 1}.end")
                self.insert(f"{i + 1}.0", line)
        self.configure(state="disabled")

    def clear_status(self):
        self.status_lines = ["Processing..."]
        self.initialized = False
        self._render()

    def append_message(self, text, kind="info"):
        self.configure(state="normal")
        self._ensure_initialized()
        tag = f"msg_{kind}"
        message = self._format_terminal_message(text)
        if self.index("end-1c") != "1.0":
            self.insert("end", "\n")
        start = self.index("end")
        self.insert("end", message)
        end = self.index("end")
        self.tag_add(tag, start, end)
        self.tag_config(tag, foreground=TEXT_MAIN)
        self.configure(state="disabled")

    def write_finished(self, text):
        self.configure(state="normal")
        self._ensure_initialized()

        current = self.get("3.0", "3.end").strip()
        if not current:
            current = "Estimated time: --:--"
        self.delete("3.0", "3.end")
        self.insert("3.0", current + " ")
        start = self.index("3.end")
        self.insert("3.end", f"({text})")
        end = self.index("3.end")
        self.tag_add("finished", start, end)
        self.tag_config("finished", foreground="#22c55e")
        self.configure(state="disabled")

    def _render(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        for line in self.status_lines:
            self.insert("end", line + "\n")
        self.configure(state="disabled")


class ProgressBar(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.bar = ctk.CTkProgressBar(self)
        self.bar.pack(fill="x", pady=4)
        self.bar.set(0)
        self._after_id = None
        self._enabled = True
        self.label = ctk.CTkLabel(self, text="0%", font=("Consolas", 11))
        self.label.pack(anchor="e")
        self._normal_color = self.bar.cget("progress_color")
        self._logical_value = 0.0
        self._visual_value = 0.0
        self._animating = False
        self._speed = 0.03  # Faster animation to keep up with progress

    def update(self, value):
        if not self._enabled:
            return
        value = max(0.0, min(1.0, value))
        if abs(value - self._logical_value) < 0.005 or value < self._logical_value:
            return
        if not self.winfo_exists():
            return
        self._logical_value = value
        if not self._animating:
            self._animating = True
            self._after_id = self.after(10, self._animate_step)

    def _animate_step(self):
        if self._logical_value - self._visual_value < 0.01:
            self._visual_value = self._logical_value
            self.bar.set(self._visual_value)
            self.label.configure(text=f"{int(self._visual_value * 100)}%")
            self._animating = False
            return
        delta = max((self._logical_value - self._visual_value) * self._speed, 0.004)
        self._visual_value += delta
        self.bar.set(self._visual_value)
        self.label.configure(text=f"{int(self._visual_value * 100)}%")
        self._after_id = self.after(16, self._animate_step)

    def mark_finished(self):
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self._animating = False
        self._logical_value = 1.0
        self._visual_value = 1.0
        self.bar.configure(progress_color="#22c55e")
        self.bar.set(1.0)
        self.label.configure(text="100%")

    def reset(self):
        self._enabled = False
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self._animating = False
        self._logical_value = 0.0
        self._visual_value = 0.0
        self.bar.configure(progress_color=self._normal_color)
        self.bar.set(0)
        self.label.configure(text="0%")
        self._enabled = True


class PreviewFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=BG_MAIN, border_width=1,
                         border_color=BORDER_SOFT2, corner_radius=15)
        self.info_label = ctk.CTkLabel(self, text="", font=("Consolas", 10))
        self.info_label.pack(anchor="n", pady=4)
        self.label = ctk.CTkLabel(self, text="")
        self.label.pack(expand=True, anchor="center")
        self._img_ref = None

    def update_image(self, image):
        if not image or not self.winfo_exists():
            return
        self._img_ref = ctk.CTkImage(light_image=image, size=image.size)
        self.label.configure(image=self._img_ref)

    def update_info(self, text):
        self.info_label.configure(text=text)

    def clear_image(self):
        self.label.configure(image=None)
        self._img_ref = None

    def clear_all(self):
        self.clear_image()
        self.info_label.configure(text="")


class FileSelector(ctk.CTkFrame):
    def __init__(self, master, label="File", width=400):
        super().__init__(master, fg_color="transparent")
        self.paths = []
        ctk.CTkLabel(self, text=label, font=("Consolas", 12)).pack(anchor="w")
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(4, 8))
        self.entry = ctk.CTkEntry(
            row, width=width, corner_radius=15, fg_color=BG_MAIN,
            border_width=1, border_color=BORDER_SOFT,
            text_color="#ededed", font=("Consolas", 11),
            placeholder_text_color=TEXT_MUTED
        )
        self.entry.pack(side="left")
        self.button = ctk.CTkButton(
            row, text="…", width=10, height=10, corner_radius=15,
            fg_color=BG_CARD, hover_color="#615f5f", border_width=1,
            border_color=BORDER_SOFT, text_color=TEXT_MUTED,
            command=self.select
        )
        self.button.pack(side="right", padx=(6, 0))
        self.entry.pack_propagate(False)
        self.button.pack_propagate(False)

    def select(self):
        paths = list(fd.askopenfilenames(
            filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"),
                       ("All files", "*.*")]
        ))
        if paths:
            self.paths = paths
            self.entry.delete(0, "end")
            if len(paths) == 1:
                self.entry.insert(0, paths[0])
            else:
                self.entry.insert(0, f"{len(paths)} videos selected")

    def get(self):
        return self.entry.get()

    def get_paths(self):
        value = self.get().strip()
        if self.paths:
            if len(self.paths) > 1 and value == f"{len(self.paths)} videos selected":
                return list(self.paths)
            if len(self.paths) == 1 and value == self.paths[0]:
                return list(self.paths)
        return [value] if value else []


class DirectorySelector(FileSelector):
    def select(self):
        path = fd.askdirectory()
        if path:
            self.entry.delete(0, "end")
            self.entry.insert(0, path)


class RadioGroup(ctk.CTkFrame):
    def __init__(self, master, variable, options, columns=4, radio_width=120, height=32):
        super().__init__(master, fg_color="transparent", height=height)
        self.grid_propagate(False)
        self.radios = []
        for i, (label, value) in enumerate(options):
            rb = ctk.CTkRadioButton(
                self, text=label, variable=variable, value=value,
                width=radio_width, radiobutton_width=10, radiobutton_height=10,
                fg_color=ACCENT, border_color="#4b5563", hover_color="#6366f1",
                text_color=TEXT_MAIN, text_color_disabled=TEXT_MUTED,
                bg_color="transparent", font=("Consolas", 12)
            )
            rb.grid(row=0, column=i, padx=(0, 12), pady=0, sticky="w")
            self.radios.append(rb)


# ============================================================
# Scene analysis helpers (heavy, only used via Auto profile)
# ============================================================
def _otsu_1d(sorted_vals):
    """Otsu thresholding 1D. Retorna threshold ou None."""
    n = len(sorted_vals)
    if n < 4:
        return None
    total_sum = sum(sorted_vals)
    if total_sum < 1e-15:
        return None
    weight_bg = 0
    sum_bg = 0.0
    max_var = 0.0
    best_t = sorted_vals[0]
    for i in range(n):
        weight_bg += 1
        sum_bg += sorted_vals[i]
        weight_fg = n - weight_bg
        if weight_fg == 0:
            break
        sum_fg = total_sum - sum_bg
        mean_bg = sum_bg / weight_bg
        mean_fg = sum_fg / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            best_t = sorted_vals[i]
    if max_var < 1e-15:
        return None
    return best_t


def _compute_ssim_simplified(img1, img2, window_size=11):
    if img1.shape != img2.shape:
        return 0.0
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    mu_a = cv2.GaussianBlur(a, (window_size, window_size), 1.5)
    mu_b = cv2.GaussianBlur(b, (window_size, window_size), 1.5)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a = cv2.GaussianBlur(a * a, (window_size, window_size), 1.5) - mu_a_sq
    sigma_b = cv2.GaussianBlur(b * b, (window_size, window_size), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (window_size, window_size), 1.5) - mu_ab
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
            (mu_a_sq + mu_b_sq + c1) * (sigma_a + sigma_b + c2) + 1e-8)
    return float(np.clip(ssim.mean(), 0.0, 1.0))


def _compute_ecr(gray1, gray2):
    edges1 = cv2.Canny(gray1, 30, 70)
    edges2 = cv2.Canny(gray2, 30, 70)
    total = cv2.countNonZero(cv2.bitwise_or(edges1, edges2))
    if total <= 0:
        return 0.0
    changed = cv2.countNonZero(cv2.bitwise_xor(edges1, edges2))
    return float(changed / total)


def _adaptive_threshold(video_path):
    """Pipeline avancado: multi-metrica + Otsu + pesos dinamicos."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 32.0, 4.0

    fps_cap = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = max(1, fps_cap if fps_cap > 0 else 30)
    if total < 10:
        cap.release()
        return 32.0, 4.0

    frames_per_clip = 6
    sample_interval_s = 2.5
    frame_interval = max(3, int(fps * sample_interval_s))
    num_clips = max(15, min(50, int(total / frame_interval)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, probe = cap.read()
    if probe is not None:
        h, w = probe.shape[:2]
        target_area = 120 * 90
        cur_area = w * h
        if cur_area > target_area * 2:
            scale = (target_area / cur_area) ** 0.5
            rw, rh = int(w * scale), int(h * scale)
        else:
            rw, rh = w, h
        dim = (max(rw, 64), max(rh, 48))
    else:
        dim = (120, 90)

    BGR2GRAY = cv2.COLOR_BGR2GRAY
    BGR2YCrCb = cv2.COLOR_BGR2YCrCb
    BGR2HSV = cv2.COLOR_BGR2HSV
    HIST_NORM = cv2.NORM_MINMAX

    diff_y_list, diff_c_list, diff_hy_list, diff_hs_list = [], [], [], []
    diff_edge_list, diff_struct_list, clip_motion = [], [], []

    for _i in range(num_clips):
        pos = min(_i * frame_interval, max(0, int(total) - frames_per_clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        prev_y, prev_cbcr, prev_hist_y, prev_hist_hs = None, None, None, None
        clip_y_diffs = []

        for _j in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize FIRST, then convert — avoids 3× redundant resizes
            small = cv2.resize(frame, dim)
            gray = cv2.cvtColor(small, BGR2GRAY)
            ycrcb = cv2.cvtColor(small, BGR2YCrCb)
            hsv = cv2.cvtColor(small, BGR2HSV)
            cbcr = ycrcb[:, :, 1:]

            hist_y = cv2.calcHist([gray], [0], None, [32], [0, 256])
            cv2.normalize(hist_y, hist_y, 0, 1, HIST_NORM)
            hist_hs = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist_hs, hist_hs, 0, 1, HIST_NORM)

            if prev_y is not None:
                d_y = cv2.absdiff(gray, prev_y).mean()
                d_c = cv2.absdiff(cbcr, prev_cbcr).mean()
                d_hy = cv2.compareHist(prev_hist_y, hist_y, cv2.HISTCMP_BHATTACHARYYA)
                d_hs = cv2.compareHist(prev_hist_hs, hist_hs, cv2.HISTCMP_BHATTACHARYYA)
                d_edge = _compute_ecr(prev_y, gray)
                d_struct = 1.0 - _compute_ssim_simplified(prev_y, gray)
                diff_y_list.append(d_y)
                diff_c_list.append(d_c)
                diff_hy_list.append(d_hy)
                diff_hs_list.append(d_hs)
                diff_edge_list.append(d_edge)
                diff_struct_list.append(d_struct)
                clip_y_diffs.append(d_y)

            prev_y, prev_cbcr, prev_hist_y, prev_hist_hs = gray, cbcr, hist_y, hist_hs

        if clip_y_diffs:
            clip_motion.append(sum(clip_y_diffs) / len(clip_y_diffs))

    cap.release()
    n = len(diff_y_list)
    if n < 8:
        return 32.0, 4.0

    def _robust_rank(vals):
        s = sorted(vals)
        rank = {}
        for idx, v in enumerate(s):
            if v not in rank:
                rank[v] = idx
        return [rank[v] / max(1, len(s) - 1) for v in vals]

    def _var(vals):
        m = sum(vals) / len(vals)
        return sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)

    ry = _robust_rank(diff_y_list)
    rc = _robust_rank(diff_c_list)
    rhy = _robust_rank(diff_hy_list)
    rhs = _robust_rank(diff_hs_list)
    re = _robust_rank(diff_edge_list)
    rs = _robust_rank(diff_struct_list)

    vars_list = [_var(diff_y_list), _var(diff_c_list),
                 _var(diff_hy_list), _var(diff_hs_list),
                 _var(diff_edge_list), _var(diff_struct_list)]
    v_total = sum(vars_list)
    if v_total < 1e-15:
        weights = [0.28, 0.14, 0.10, 0.12, 0.20, 0.16]
    else:
        weights = [v / v_total for v in vars_list]
        weights = [0.08 + 0.38 * w for w in weights]
        weights[4] *= 1.25
        weights[5] *= 1.15
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

    composite = [
        weights[0] * a + weights[1] * b + weights[2] * c +
        weights[3] * d + weights[4] * e + weights[5] * st
        for a, b, c, d, e, st in zip(ry, rc, rhy, rhs, re, rs)
    ]

    mf = list(composite)
    for i in range(1, n - 1):
        w = sorted([composite[i - 1], composite[i], composite[i + 1]])
        mf[i] = w[1]

    s = sorted(mf)
    q1 = s[n // 4]
    q3 = s[3 * n // 4]
    iqr_v = q3 - q1
    spike_cap = q3 + 4.0 * iqr_v
    spike_hard_level = q3 + 2.0 * iqr_v

    clean = list(mf)
    for i in range(1, n - 1):
        if mf[i] > spike_cap:
            prev_ok = mf[i - 1] < spike_hard_level
            next_ok = mf[i + 1] < spike_hard_level
            if prev_ok and next_ok:
                clean[i] = spike_cap
            else:
                clean[i] = spike_cap + (mf[i] - spike_cap) * 0.3

    motion_index = sum(clip_motion) / max(1, len(clip_motion)) if clip_motion else 0.0
    otsu_t = _otsu_1d(sorted(clean))

    if otsu_t is not None:
        grp0 = [v for v in clean if v <= otsu_t]
        grp1 = [v for v in clean if v > otsu_t]
        if grp0 and grp1:
            noise_m = sum(grp0) / len(grp0)
            trans_m = sum(grp1) / len(grp1)
            sep = trans_m - noise_m
            raw_t = noise_m + sep * 0.35
        else:
            raw_t = otsu_t
    else:
        raw_t = sorted(clean)[n // 2]

    threshold = 15.0 + raw_t * 40.0
    if motion_index > 12:
        motion_boost = min(5.0, (motion_index - 12) * 0.2)
        threshold = min(55.0, threshold + motion_boost)
    threshold = max(15.0, min(55.0, threshold))
    threshold = round(threshold, 1)

    # min_dur: INVERSE relationship with threshold.
    # High threshold (lots of motion)  -> LOW min_dur (accept short scenes)
    # Low threshold (calm video)        -> HIGH min_dur (only long scenes)
    t_norm = 1.0 - (threshold - 15.0) / 40.0  # Inverted: 1.0=calm, 0.0=agitated
    min_dur = 1.5 + 8.5 * (t_norm ** 1.5)
    min_dur = max(1.0, min(10.0, min_dur))
    min_dur = round(min_dur, 1)

    return threshold, min_dur


def _percentile(vals, pct, default=0.0):
    if not vals:
        return default
    return float(np.percentile(np.asarray(vals, dtype=np.float32), pct))


def _median(vals, default=0.0):
    if not vals:
        return default
    return float(np.median(np.asarray(vals, dtype=np.float32)))


def _profile_name(cfg):
    label = cfg.get("label", "Normal")
    return label if label in ("Low", "Normal", "High") else "Auto"


def _make_candidate(frame, score, kind, confidence=None, source=None):
    score = float(score)
    if confidence is None:
        confidence = score / (score + 0.25)
    return {
        "frame": int(frame),
        "score": score,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "type": kind,
        "source": source or kind,
    }


def _classifier_profile_params(profile):
    return {
        "Low": {
            "accept": 0.88, "semantic": 0.95, "gradual": 0.82,
            "min_scene_s": 3.5, "refine_floor": 0.12,
        },
        "Normal": {
            "accept": 0.72, "semantic": 0.82, "gradual": 0.68,
            "min_scene_s": 0.9, "refine_floor": 0.08,
        },
        "High": {
            "accept": 0.58, "semantic": 0.68, "gradual": 0.55,
            "min_scene_s": 0.35, "refine_floor": 0.05,
        },
        "Auto": {
            "accept": 0.66, "semantic": 0.76, "gradual": 0.62,
            "min_scene_s": 0.6, "refine_floor": 0.06,
        },
    }.get(profile, {
        "accept": 0.72, "semantic": 0.82, "gradual": 0.68,
        "min_scene_s": 0.9, "refine_floor": 0.08,
    })


def _gradual_profile_params(profile):
    return {
        "Low": {"step_s": 0.35, "window": 4, "score_q": 94, "score_floor": 0.33, "min_gap_s": 4.0},
        "Normal": {"step_s": 0.25, "window": 4, "score_q": 90, "score_floor": 0.26, "min_gap_s": 1.8},
        "High": {"step_s": 0.20, "window": 3, "score_q": 84, "score_floor": 0.20, "min_gap_s": 0.7},
        "Auto": {"step_s": 0.25, "window": 4, "score_q": 88, "score_floor": 0.23, "min_gap_s": 1.2},
    }.get(profile, {"step_s": 0.25, "window": 4, "score_q": 90, "score_floor": 0.26, "min_gap_s": 1.8})


def _hist_distance(a, b):
    try:
        return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
    except Exception:
        return 0.0


def _cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(1.0 - np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _scene_embedding_from_gray_hsv(gray, hsv):
    h, w = gray.shape

    grid = []
    rows, cols = 4, 6
    for gy in range(rows):
        y1, y2 = int(gy * h / rows), int((gy + 1) * h / rows)
        for gx in range(cols):
            x1, x2 = int(gx * w / cols), int((gx + 1) * w / cols)
            cell_gray = gray[y1:y2, x1:x2]
            cell_hsv = hsv[y1:y2, x1:x2]
            grid.extend([
                float(cell_gray.mean()) / 255.0,
                float(cell_gray.std()) / 255.0,
                float(cell_hsv[:, :, 0].mean()) / 180.0,
                float(cell_hsv[:, :, 1].mean()) / 255.0,
                float(cell_hsv[:, :, 2].mean()) / 255.0,
            ])

    edges = cv2.Canny(gray, 40, 90)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=False)
    edge_grid = []
    orient_grid = []
    for gy in range(rows):
        y1, y2 = int(gy * h / rows), int((gy + 1) * h / rows)
        for gx in range(cols):
            x1, x2 = int(gx * w / cols), int((gx + 1) * w / cols)
            edge_grid.append(float(edges[y1:y2, x1:x2].mean()) / 255.0)
            cell_mag = mag[y1:y2, x1:x2]
            cell_ang = ang[y1:y2, x1:x2]
            hist, _ = np.histogram(
                cell_ang, bins=4, range=(0.0, np.pi * 2),
                weights=cell_mag)
            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-6)
            orient_grid.extend(hist.tolist())

    cy1, cy2 = int(h * 0.20), int(h * 0.80)
    cx1, cx2 = int(w * 0.18), int(w * 0.82)
    center_gray = gray[cy1:cy2, cx1:cx2]
    center_hsv = hsv[cy1:cy2, cx1:cx2]
    center = np.asarray([
        float(center_gray.mean()) / 255.0,
        float(center_gray.std()) / 255.0,
        float(center_hsv[:, :, 0].mean()) / 180.0,
        float(center_hsv[:, :, 1].mean()) / 255.0,
        float(center_hsv[:, :, 2].mean()) / 255.0,
        float(edges[cy1:cy2, cx1:cx2].mean()) / 255.0,
    ], dtype=np.float32)

    dct_src = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    dct = cv2.dct(dct_src)
    dct_low = dct[:6, :8].reshape(-1)
    dct_low = dct_low / (np.linalg.norm(dct_low) + 1e-8)

    emb = np.concatenate([
        np.asarray(grid, dtype=np.float32),
        np.asarray(edge_grid, dtype=np.float32) * 1.5,
        np.asarray(orient_grid, dtype=np.float32) * 0.55,
        center * 1.2,
        dct_low.astype(np.float32) * 0.75,
    ])
    return emb / (np.linalg.norm(emb) + 1e-8)


def _scene_embedding(frame, size=(160, 90)):
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return _scene_embedding_from_gray_hsv(gray, hsv)


def _build_transition_feature_cache(video_path, fps, total_frames, profile="Normal", stop_cb=None):
    gradual_params = _gradual_profile_params(profile)
    semantic_params = _semantic_profile_params(profile)
    step_s = min(gradual_params["step_s"], semantic_params["step_s"])
    frame_step = max(1, int(round(fps * step_s)))
    duration_s = total_frames / max(fps, 1.0) if total_frames else 0.0
    profile_caps = {"Low": 1400, "Normal": 1800, "High": 2200, "Auto": 1900}
    max_samples = profile_caps.get(profile, 1800)
    if duration_s <= 20 * 60:
        max_samples = int(max_samples * 1.25)
    elif duration_s >= 90 * 60:
        max_samples = int(max_samples * 0.85)
    if total_frames and total_frames / frame_step > max_samples:
        frame_step = max(frame_step, int(total_frames / max_samples))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    features = []
    target = (160, 90)
    try:
        for frame_idx in range(0, int(total_frames or 0), frame_step):
            if stop_cb and stop_cb():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            small = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist_hs = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
            cv2.normalize(hist_hs, hist_hs, 0, 1, cv2.NORM_L1)

            features.append({
                "frame": int(frame_idx),
                "time": frame_idx / max(fps, 1.0),
                "hist": hist_hs,
                "luma": float(gray.mean()) / 255.0,
                "contrast": float(gray.std()) / 255.0,
                "black": float((gray < 18).mean()),
                "white": float((gray > 238).mean()),
                "embedding": _scene_embedding_from_gray_hsv(gray, hsv),
            })
    finally:
        cap.release()

    return features


def _frame_signature(frame, size=(160, 90)):
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_L1)
    return gray, hist


def _signature_change_score(prev_sig, cur_sig):
    prev_gray, prev_hist = prev_sig
    cur_gray, cur_hist = cur_sig
    hist_gap = _hist_distance(prev_hist, cur_hist)
    luma_gap = float(cv2.absdiff(prev_gray, cur_gray).mean()) / 255.0
    edge_gap = _compute_ecr(prev_gray, cur_gray)
    struct_gap = 1.0 - _compute_ssim_simplified(prev_gray, cur_gray)
    return hist_gap * 0.45 + luma_gap * 0.20 + edge_gap * 0.20 + struct_gap * 0.15


def _detect_gradual_transitions(video_path, fps, total_frames, profile="Normal", stop_cb=None, features=None):
    """Find fade/dissolve candidates using windowed color/luma statistics.

    PySceneDetect is strong on hard cuts. This pass looks for slower transitions:
    a window before/after the candidate must be visually different, while the
    local sequence shows a sustained ramp or fade pattern.
    """
    params = _gradual_profile_params(profile)

    duration = total_frames / max(fps, 1.0) if total_frames else 0.0
    if duration <= 1.0:
        return []

    if features is None:
        features = _build_transition_feature_cache(
            video_path, fps, total_frames, profile=profile, stop_cb=stop_cb)
    n = len(features)
    win = params["window"]
    if n < win * 2 + 3:
        return []

    scores = []
    candidates = []
    for i in range(win, n - win):
        before = features[i - win:i]
        after = features[i + 1:i + 1 + win]
        cur = features[i]

        hist_before = np.mean([f["hist"] for f in before], axis=0).astype(np.float32)
        hist_after = np.mean([f["hist"] for f in after], axis=0).astype(np.float32)
        hist_gap = _hist_distance(hist_before, hist_after)

        luma_before = _median([f["luma"] for f in before])
        luma_after = _median([f["luma"] for f in after])
        contrast_before = _median([f["contrast"] for f in before])
        contrast_after = _median([f["contrast"] for f in after])
        luma_gap = abs(luma_after - luma_before)
        contrast_gap = abs(contrast_after - contrast_before)

        local = features[i - win:i + win + 1]
        lumas = [f["luma"] for f in local]
        blacks = [f["black"] for f in local]
        whites = [f["white"] for f in local]
        dark_peak = max(blacks)
        white_peak = max(whites)
        luma_slope = abs(lumas[-1] - lumas[0])
        fade_score = 0.0
        if dark_peak > 0.28 or white_peak > 0.25:
            fade_score = max(dark_peak, white_peak) * 0.45 + luma_slope * 0.35

        score = hist_gap * 0.70 + luma_gap * 0.20 + contrast_gap * 0.10 + fade_score
        scores.append(score)
        candidates.append((cur["frame"], cur["time"], score, hist_gap, fade_score))

    if not scores:
        return []

    threshold = max(params["score_floor"], _percentile(scores, params["score_q"]))
    min_gap_frames = max(1, int(params["min_gap_s"] * fps))
    selected = []
    for frame, _time_s, score, hist_gap, fade_score in sorted(candidates, key=lambda x: x[2], reverse=True):
        if score < threshold:
            break
        if hist_gap < 0.18 and fade_score < 0.18:
            continue
        if frame < min_gap_frames or (total_frames and frame > total_frames - min_gap_frames):
            continue
        if any(abs(frame - chosen) < min_gap_frames for chosen in selected):
            continue
        selected.append(int(frame))

    selected_set = set(selected)
    return [
        _make_candidate(frame, score, "gradual", confidence=min(1.0, score / max(threshold, 1e-6)),
                        source="fade" if fade_score >= hist_gap else "dissolve")
        for frame, _time_s, score, hist_gap, fade_score in candidates
        if int(frame) in selected_set
    ]


def _semantic_profile_params(profile):
    return {
        "Low": {"step_s": 0.55, "window": 3, "score_q": 95, "score_floor": 0.18, "min_gap_s": 4.0},
        "Normal": {"step_s": 0.40, "window": 3, "score_q": 90, "score_floor": 0.12, "min_gap_s": 1.5},
        "High": {"step_s": 0.28, "window": 2, "score_q": 84, "score_floor": 0.08, "min_gap_s": 0.6},
        "Auto": {"step_s": 0.35, "window": 3, "score_q": 88, "score_floor": 0.10, "min_gap_s": 1.0},
    }.get(profile, {"step_s": 0.40, "window": 3, "score_q": 90, "score_floor": 0.12, "min_gap_s": 1.5})


def _detect_semantic_transitions(video_path, fps, total_frames, profile="Normal", stop_cb=None, features=None):
    """Find composition/context changes missed by global frame-diff detectors."""
    params = _semantic_profile_params(profile)
    if features is None:
        features = _build_transition_feature_cache(
            video_path, fps, total_frames, profile=profile, stop_cb=stop_cb)
    samples = [(f["frame"], f["embedding"]) for f in features if "embedding" in f]

    win = params["window"]
    if len(samples) < win * 2 + 3:
        return []

    scores = []
    candidates = []
    for i in range(win, len(samples) - win):
        before = np.mean([emb for _, emb in samples[i - win:i]], axis=0)
        after = np.mean([emb for _, emb in samples[i:i + win]], axis=0)
        score = _cosine_distance(before, after)
        scores.append(score)
        candidates.append((samples[i][0], score))

    if not scores:
        return []

    threshold = max(params["score_floor"], _percentile(scores, params["score_q"]))
    min_gap_frames = max(1, int(params["min_gap_s"] * fps))
    selected = []
    for frame, score in sorted(candidates, key=lambda x: x[1], reverse=True):
        if score < threshold:
            break
        if frame < min_gap_frames or frame > total_frames - min_gap_frames:
            continue
        if any(abs(frame - chosen) < min_gap_frames for chosen in selected):
            continue
        selected.append(int(frame))

    selected_set = set(selected)
    return [
        _make_candidate(frame, score, "semantic", confidence=min(1.0, score / max(threshold, 1e-6)))
        for frame, score in candidates
        if int(frame) in selected_set
    ]


def _refine_scene_candidates(video_path, candidates, fps, total_frames, profile="Normal", stop_cb=None):
    if not candidates:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return candidates

    profile_window = {
        "Low": 0.65,
        "Normal": 0.50,
        "High": 0.35,
        "Auto": 0.60,
    }.get(profile, 0.35)
    window_frames = max(3, int(round(fps * profile_window)))
    step = max(1, int(round(fps / 18.0)))
    refined = []

    try:
        for candidate in candidates:
            if stop_cb and stop_cb():
                break
            boundary = int(candidate["frame"])
            left = max(step, int(boundary) - window_frames)
            right = min(int(total_frames) - 1, int(boundary) + window_frames)
            best_frame = int(boundary)
            best_score = -1.0
            scores = []
            prev_sig = None
            prev_idx = None

            for idx in range(left, right + 1, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                sig = _frame_signature(frame)
                if prev_sig is not None and prev_idx is not None:
                    score = _signature_change_score(prev_sig, sig)
                    scores.append(score)
                    if score > best_score:
                        best_score = score
                        best_frame = idx
                prev_sig = sig
                prev_idx = idx

            if scores:
                med = _median(scores)
                spread = _percentile(scores, 90) - _percentile(scores, 50)
                if best_score < med + max(0.08, spread * 0.65):
                    best_frame = int(boundary)
            refined_candidate = dict(candidate)
            refined_candidate["frame"] = int(best_frame)
            refined_candidate["refine_score"] = float(max(0.0, best_score))
            refined.append(refined_candidate)
    finally:
        cap.release()

    return sorted(
        (c for c in refined if 0 < int(c["frame"]) < total_frames),
        key=lambda c: int(c["frame"]))


def _read_frame_at(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ret, frame = cap.read()
    return frame if ret and frame is not None else None


def _camera_motion_compensation_score(frame_a, frame_b, orb=None):
    try:
        gray_a = cv2.cvtColor(cv2.resize(frame_a, (240, 135), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(cv2.resize(frame_b, (240, 135), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        raw_diff = float(cv2.absdiff(gray_a, gray_b).mean()) / 255.0
        if raw_diff <= 1e-5:
            return {"raw": raw_diff, "aligned": raw_diff, "explained": 0.0, "matches": 0}

        orb = orb or cv2.ORB_create(nfeatures=450, fastThreshold=12)
        kp_a, des_a = orb.detectAndCompute(gray_a, None)
        kp_b, des_b = orb.detectAndCompute(gray_b, None)
        if des_a is None or des_b is None or len(kp_a) < 10 or len(kp_b) < 10:
            return {"raw": raw_diff, "aligned": raw_diff, "explained": 0.0, "matches": 0}

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_a, des_b)
        if len(matches) < 10:
            return {"raw": raw_diff, "aligned": raw_diff, "explained": 0.0, "matches": len(matches)}

        matches = sorted(matches, key=lambda m: m.distance)[:80]
        src = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=800, confidence=0.98)
        if matrix is None or inliers is None:
            return {"raw": raw_diff, "aligned": raw_diff, "explained": 0.0, "matches": len(matches)}

        inlier_count = int(inliers.sum())
        if inlier_count < 8:
            return {"raw": raw_diff, "aligned": raw_diff, "explained": 0.0, "matches": len(matches)}

        aligned = cv2.warpAffine(
            gray_a, matrix, (gray_b.shape[1], gray_b.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        aligned_diff = float(cv2.absdiff(aligned, gray_b).mean()) / 255.0
        explained = max(0.0, min(1.0, (raw_diff - aligned_diff) / max(raw_diff, 1e-6)))
        inlier_ratio = inlier_count / max(1, len(matches))
        explained *= max(0.25, min(1.0, inlier_ratio * 1.4))
        return {
            "raw": raw_diff,
            "aligned": aligned_diff,
            "explained": float(explained),
            "matches": int(len(matches)),
            "inliers": inlier_count,
        }
    except Exception:
        return {"raw": 0.0, "aligned": 0.0, "explained": 0.0, "matches": 0}


def _add_candidate_context_scores(video_path, candidates, fps, total_frames, stop_cb=None):
    if not candidates:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return candidates

    out = []
    far = max(2, int(round(fps * 0.75)))
    near = max(1, int(round(fps * 0.25)))
    orb = cv2.ORB_create(nfeatures=450, fastThreshold=12)
    try:
        for cand in candidates:
            if stop_cb and stop_cb():
                break
            frame = int(cand["frame"])
            idxs = [
                max(0, frame - far),
                max(0, frame - near),
                min(total_frames - 1, frame + near),
                min(total_frames - 1, frame + far),
            ]
            frames = [_read_frame_at(cap, idx) for idx in idxs]
            if any(f is None for f in frames):
                out.append(cand)
                continue
            embs = [_scene_embedding(f) for f in frames]
            pre_motion = _cosine_distance(embs[0], embs[1])
            post_motion = _cosine_distance(embs[2], embs[3])
            near_cross = _cosine_distance(embs[1], embs[2])
            far_cross = _cosine_distance(embs[0], embs[3])
            local_motion = max(pre_motion, post_motion, 1e-4)
            context_ratio = far_cross / (local_motion + 0.03)
            camera_motion = _camera_motion_compensation_score(frames[1], frames[2], orb=orb)

            enriched = dict(cand)
            enriched["context_cross"] = round(float(far_cross), 4)
            enriched["context_near_cross"] = round(float(near_cross), 4)
            enriched["context_motion"] = round(float(local_motion), 4)
            enriched["context_ratio"] = round(float(context_ratio), 4)
            enriched["camera_motion_explained"] = round(float(camera_motion.get("explained", 0.0)), 4)
            enriched["camera_motion_raw"] = round(float(camera_motion.get("raw", 0.0)), 4)
            enriched["camera_motion_aligned"] = round(float(camera_motion.get("aligned", 0.0)), 4)
            enriched["camera_motion_matches"] = int(camera_motion.get("matches", 0))
            out.append(enriched)
    finally:
        cap.release()

    return out


# ============================================================
# SceneEngine
# ============================================================
class SceneEngine:
    def __init__(self, video, output, cfg, logbox=None, progressbar=None,
                 previewer=None, preview_enabled=True):
        self.video = video
        self.output = output
        self.cfg = cfg
        self.log = logbox
        self.progress = progressbar
        self.previewer = previewer
        self.preview_enabled = preview_enabled
        self._stop = False
        self.detected = 0
        self.total = 0
        self.done = 0
        self.failed = 0
        self.completed_attempts = 0
        self._start_time = None
        self._end_time = None
        self._video_info_shown = False
        self._fps = None
        self._thumb_container = None
        self._total_frames = None
        self._ffmpeg_proc = None
        self._ffmpeg_procs = set()
        self._ffmpeg_proc_lock = threading.Lock()
        self._ui_alive = True
        self._duration = None
        self._last_preview_ratio = -1
        self._keyframes_cache = None
        self._has_audio_cache = None
        self._preview_cap = None
        self._preview_lock = threading.Lock()
        self._preview_thread = None
        self._preview_stop = False
        self._video_obj = None  # scenedetect video object for cleanup
        self._scene_candidates = []
        self._rejected_scene_candidates = []
        self._cut_failures = []
        self._cut_output_dir = None
        self._temp_files = []

    def add_temp_file(self, path):
        if path and path not in self._temp_files:
            self._temp_files.append(path)

    def cleanup_temp_files(self):
        for path in list(self._temp_files):
            remove_temp_file(path)
        self._temp_files = []

    def stop(self):
        self._stop = True
        self._ui_alive = False
        self._preview_stop = True
        try:
            procs = []
            if self._ffmpeg_proc:
                procs.append(self._ffmpeg_proc)
            try:
                with self._ffmpeg_proc_lock:
                    procs.extend(list(self._ffmpeg_procs))
            except Exception:
                pass
            for proc in procs:
                try:
                    proc.terminate()
                    time.sleep(0.3)
                    if proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if self._video_obj:
                self._video_obj.close()
        except Exception:
            pass
        self._video_obj = None
        try:
            if self._thumb_container:
                self._thumb_container.close()
        except Exception:
            pass
        self._thumb_container = None
        try:
            if self._preview_cap:
                with self._preview_lock:
                    if self._preview_cap:
                        self._preview_cap.release()
        except Exception:
            pass
        self._preview_cap = None
        # Clear cache and heavy vars
        self._keyframes_cache = None
        self._duration = None
        self._total_frames = None
        self._has_audio_cache = None
        self._ffmpeg_proc = None
        try:
            with self._ffmpeg_proc_lock:
                self._ffmpeg_procs.clear()
        except Exception:
            pass
        self._scene_candidates = []
        self._rejected_scene_candidates = []
        self._cut_failures = []
        self._cut_output_dir = None
        self.cleanup_temp_files()

    def total_time(self):
        if not self._start_time:
            return "--:--"
        end = self._end_time or time.time()
        elapsed = max(1, int(end - self._start_time))
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def run(self, scene_mode=True):
        self.scene_mode = scene_mode
        self._stop = False
        self._preview_stop = False
        self._ui_alive = True
        self._analysis_ratio = 0.0
        self._start_time = time.time()
        self._end_time = None
        self.done = 0
        self.failed = 0
        self.completed_attempts = 0
        self.detected = 0
        self.last_preview = 0
        self._last_thumb_time = 0
        self._scene_candidates = []
        self._rejected_scene_candidates = []
        self._cut_failures = []
        self._cut_output_dir = None

        if self.previewer and self.preview_enabled:
            try:
                img = self._get_preview_frame_at(0)
                if img and self._ui_alive:
                    self.previewer.after(0, lambda img=img: self.previewer.update_image(img))
            except Exception:
                pass

        if self.preview_enabled:
            try:
                with self._preview_lock:
                    self._preview_cap = cv2.VideoCapture(self.video)
            except Exception:
                self._preview_cap = None

        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=False)
        self._preview_thread.start()

        if self.previewer and not self._video_info_shown:
            info_text = self._get_video_info_text()
            self.previewer.after(0, lambda t=info_text: self.previewer.update_info(t))
            self._video_info_shown = True

        try:  # FIX: ensure preview_cap is released on exception
            scenes = self._detect_scenes_progressive() if scene_mode else self._fixed_interval()
            if not scenes or self._stop:
                return False

            # Update UI with final detected scene count before cutting
            if self.log:
                self.log.after(0, lambda: self.log.write_status(detected=self.detected, cut=0, eta="--:--"))

            # Keep preview active during cuts - release only after all cuts are done
            self._cut_scenes(scenes)
        finally:
            self._ui_alive = False
            self._preview_stop = True
            if self._preview_thread and self._preview_thread.is_alive():
                try:
                    self._preview_thread.join(timeout=1.0)
                except Exception:
                    pass
            self._preview_thread = None
            # FIX: always release preview cap
            try:
                with self._preview_lock:
                    if self._preview_cap:
                        self._preview_cap.release()
                    self._preview_cap = None
            except Exception:
                pass
            self.cleanup_temp_files()

        self._end_time = time.time()

        return True

    def _get_video_info_text(self):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,r_frame_rate:format=bit_rate",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        try:
            out = check_output_hidden(cmd).decode().splitlines()
            width, height, fps, bitrate = out
            num, den = fps.split("/")
            fps_float = round(int(num) / int(den), 2)
            return f"{width}x{height} | FPS: {fps_float} | Bitrate: {int(bitrate) / 1000:.0f} kbps"
        except Exception:
            return "Video info unavailable"

    def _detect_scenes_progressive(self):
        fps = self._get_video_fps()

        # Show initial status before adaptive scan
        if self.log:
            self.log.after(0, lambda: self.log.write_status(
                detected="analyzing...", cut=0, eta="--:--"))

        threshold, min_dur = self._map_threshold()

        result = _ensure_scenedetect()
        if result is None:
            raise RuntimeError("scenedetect not installed")
        open_video, SceneManager, ContentDetector, StatsManager, CompWeights = result

        backend = "pyav"
        if "_fixed" in os.path.splitext(os.path.basename(self.video))[0].lower():
            backend = "opencv"

        try_backends = ["pyav", "opencv"] if backend == "pyav" else [backend]
        video = None
        last_error = None
        for be in try_backends:
            try:
                if be == "pyav":
                    video = open_video(self.video, backend=be, suppress_output=True)
                else:
                    video = open_video(self.video, backend=be)
                self._video_obj = video  # Store for cleanup
                break
            except Exception as e:
                last_error = e
                if be == "pyav":
                    continue
        if video is None:
            raise RuntimeError(f"Failed to open video: {last_error}")

        if backend == "opencv":
            try:
                _ = video.frame_rate
            except Exception:
                video.close()
                self._video_obj = None  # Clear ref after close
                raise RuntimeError("Failed to read video stream")

        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.auto_downscale = False
        scene_manager.downscale = 2
        scene_manager.add_detector(
            ContentDetector(threshold=threshold,
                            min_scene_len=int(min_dur * fps),
                            luma_only=False,
                            weights=CompWeights(
                                delta_hue=0.85, delta_sat=0.85, delta_lum=1.0, delta_edges=1.15))
        )
        video_duration = None
        try:
            video_duration = self._get_video_duration()
        except Exception:
            pass

        if video_duration and video_duration > 0:
            self._total_frames = int(video_duration * fps)

        if not video_duration or video_duration <= 0:
            try:
                cmd = ["ffprobe", "-v", "error",
                       "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", self.video]
                video_duration = float(check_output_hidden(cmd).decode().strip())
                self._total_frames = int(video_duration * fps)
            except Exception:
                if self._total_frames is None:
                    self._total_frames = int(video.frame_rate * 1)
                video_duration = max((self._total_frames or 1) / fps, 1.0)

        def _progress_cb(frame_num, _):
            if self._stop:
                return False

            try:
                frame_idx = _safe_frame_index(frame_num)
                if frame_idx <= 0:
                    return True
            except Exception:
                return True

            current_time = frame_idx / fps
            if time.time() - getattr(self, "_last_ui_update", 0) < 0.3:
                return True
            self._last_ui_update = time.time()

            if video_duration and video_duration > 0:
                ratio = min(current_time / video_duration, 1.0)
            else:
                ratio = self.done / max(self.total, 1)
            ratio = max(self._analysis_ratio, ratio)
            self._analysis_ratio = ratio

            if self.progress:
                self.progress.after(0, lambda v=ratio * 0.4: self.progress.update(v))

            # Update log - ETA will show "--:--" during detection, real time during cuts
            if self.log:
                eta = self._calculate_eta()
                self.log.after(0, lambda d=self.detected, c=self.done, e=eta:
                self.log.write_status(detected=d, cut=c, eta=e))
            return True

        scene_list = []
        try:
            detect_exception = None

            def _run_detect():
                nonlocal detect_exception
                try:
                    scene_manager.detect_scenes(video=video, callback=_progress_cb)
                except Exception as e:
                    detect_exception = e

            detect_thread = threading.Thread(target=_run_detect, daemon=True)
            detect_thread.start()

            while detect_thread.is_alive():
                if self._stop:
                    try:
                        video.close()
                        self._video_obj = None  # Clear ref after close
                    except Exception:
                        pass
                    break
                time.sleep(0.05)

            detect_thread.join(timeout=1.0)

            if self._stop:
                return []
            if detect_exception:
                raise detect_exception

            scene_list = scene_manager.get_scene_list()
            self.detected = len(scene_list)
        except Exception as e:
            err_str = str(e).lower()
            needs_retry = (
                    "avcodec_send_packet" in err_str or  # 1094995529 / AVERROR_INVALIDDATA
                    "avcodec" in err_str or
                    "decode" in err_str or
                    "invalid data" in err_str or
                    "no start code" in err_str or
                    "avcodec_receive_frame" in err_str or
                    "packet" in err_str or
                    "av.error" in err_str or
                    "fatal" in err_str or
                    "thread" in err_str
            )
            if needs_retry and backend == "pyav":
                # Skip OpenCV — uses same h264 decoder, will fail the same way.
                # Go straight to ffmpeg re-encode fallback.
                fixed = self.video.rsplit(".", 1)[0] + "_fixed.mp4"
                remove_temp_file(fixed)
                self.add_temp_file(fixed)
                if True:
                    print(f"Video has corrupted frames - re-encoding with ffmpeg...")
                    if self.log:
                        self.log.after(0, lambda: self.log.write_status(
                            detected="Fixing corrupted video...", cut=0, eta="--:--"))
                    self._run_ffmpeg_tracked(
                        ["ffmpeg", "-y", "-err_detect", "ignore_err",
                         "-i", self.video, "-c:v", "libx264", "-crf", "22",
                         "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                         "-c:a", "copy", fixed],
                        timeout=300  # FIX: add timeout to prevent hanging
                    )
                    if self.log:
                        self.log.after(0, lambda: self.log.write_status(
                            detected="Re-encoding complete", cut=0, eta="--:--"))

                if os.path.exists(fixed):
                    print(f"Using fixed video: {fixed}")
                    # Verify the fixed file is valid
                    if not is_valid_video_file(fixed):
                        print(f"Fixed file is corrupted, re-encoding again...")
                        remove_temp_file(fixed)
                        self._run_ffmpeg_tracked(
                            ["ffmpeg", "-y", "-err_detect", "ignore_err",
                             "-i", self.video, "-c:v", "libx264", "-crf", "22",
                             "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                             "-c:a", "copy", fixed],
                            timeout=300  # FIX: add timeout to prevent hanging
                        )
                    if not os.path.exists(fixed) or not is_valid_video_file(fixed):
                        print("ffmpeg re-encode failed, no valid fixed file")
                        if self.log:
                            self.log.after(0, lambda: self.log.write_status(
                                detected="Failed to fix corrupted video", cut=0, eta="--:--"))
                        return []
                    self.video = fixed
                    # FIX: invalidate caches when video changes
                    self._fps = None
                    self._duration = None
                    self._keyframes_cache = None
                    # Always use OpenCV for fixed files — most reliable
                    try:
                        video = open_video(self.video, backend="opencv")
                        self._video_obj = video  # Store for cleanup
                    except Exception:
                        try:
                            video = open_video(self.video, backend="pyav", suppress_output=True)
                            self._video_obj = video  # Store for cleanup
                        except Exception:
                            print("Both backends failed on fixed video!")
                            if self.log:
                                self.log.after(0, lambda: self.log.write_status(
                                    detected="Failed to open fixed video", cut=0, eta="--:--"))
                            return []
                    print(f"Video opened successfully, starting detection...")
                    scene_manager = SceneManager(StatsManager())
                    scene_manager.auto_downscale = False
                    scene_manager.downscale = 2
                    scene_manager.add_detector(
                        ContentDetector(threshold=threshold,
                                        min_scene_len=int(min_dur * fps),
                                        luma_only=False,
                                        weights=CompWeights(
                                            delta_hue=0.85, delta_sat=0.85, delta_lum=1.0, delta_edges=1.15))
                    )
                    scene_manager.detect_scenes(video=video, callback=_progress_cb)
                    scene_list = scene_manager.get_scene_list()
                    self.detected = len(scene_list)
                    print(f"Detection complete: {self.detected} scenes found")
                    if self.log:
                        self.log.after(0, lambda d=self.detected: self.log.write_status(
                            detected=f"{d} scenes detected", cut=0, eta="--:--"))
                    try:
                        video.close()
                        self._video_obj = None  # Clear ref after close
                    except Exception:
                        pass
                    scenes = self._compose_scene_list(scene_list, fps, min_dur)
                    return scenes
                else:
                    print("ffmpeg re-encode failed, no fixed file created")
                    if self.log:
                        self.log.after(0, lambda: self.log.write_status(
                            detected="Failed to create fixed video", cut=0, eta="--:--"))
                    return []
            else:
                return []
        finally:
            try:
                if video:
                    video.close()
                    self._video_obj = None  # Clear ref after close
            except Exception:
                pass

        scenes = self._compose_scene_list(scene_list, fps, min_dur)
        self.detected = len(scenes)
        return scenes

    def _compose_scene_list(self, scene_list, fps, min_dur):
        total_frames = self._total_frames
        if not total_frames:
            try:
                total_frames = int(self._get_video_duration() * fps)
            except Exception:
                total_frames = int(fps)

        hard_candidates = []
        for start, end in scene_list or []:
            s = start.get_frames()
            e = end.get_frames()
            if 0 < s < total_frames:
                hard_candidates.append(_make_candidate(int(s), 1.0, "hard_cut", confidence=1.0,
                                                       source="pyscenedetect"))
            if 0 < e < total_frames:
                hard_candidates.append(_make_candidate(int(e), 1.0, "hard_cut", confidence=1.0,
                                                       source="pyscenedetect"))

        profile = _profile_name(self.cfg)
        feature_cache = _build_transition_feature_cache(
            self.video, fps, total_frames, profile=profile, stop_cb=lambda: self._stop)
        gradual_candidates = _detect_gradual_transitions(
            self.video, fps, total_frames, profile=profile,
            stop_cb=lambda: self._stop, features=feature_cache)
        semantic_candidates = _detect_semantic_transitions(
            self.video, fps, total_frames, profile=profile,
            stop_cb=lambda: self._stop, features=feature_cache)

        min_gap = self._boundary_min_gap_frames(fps, min_dur, profile)
        candidates = self._merge_candidates(
            hard_candidates + gradual_candidates + semantic_candidates,
            min_gap, total_frames)
        candidates = _refine_scene_candidates(
            self.video, candidates, fps, total_frames, profile=profile,
            stop_cb=lambda: self._stop)
        candidates = self._merge_candidates(candidates, min_gap, total_frames)
        candidates = _add_candidate_context_scores(
            self.video, candidates, fps, total_frames, stop_cb=lambda: self._stop)
        candidates, rejected = self._classify_candidates(candidates, fps, total_frames, profile)
        self._scene_candidates = candidates
        self._rejected_scene_candidates = rejected
        boundaries = [int(c["frame"]) for c in candidates]
        if not boundaries:
            self.detected = 1
            return [(0, total_frames)]
        scenes = self._scenes_from_boundaries(boundaries, total_frames)
        self.detected = len(scenes)
        return scenes

    def _boundary_min_gap_frames(self, fps, min_dur, profile):
        profile_floor = {
            "Low": 3.5,
            "Normal": 1.2,
            "High": 0.45,
            "Auto": 0.8,
        }.get(profile, 1.2)
        seconds = min(max(profile_floor, min_dur * 0.35), max(min_dur, profile_floor))
        return max(1, int(seconds * fps))

    def _classify_candidates(self, candidates, fps, total_frames, profile):
        if not candidates:
            return [], []

        params = _classifier_profile_params(profile)
        accepted = []
        rejected = []
        sorted_candidates = sorted(candidates, key=lambda c: int(c["frame"]))
        min_scene_frames = max(1, int(params["min_scene_s"] * fps))

        for idx, cand in enumerate(sorted_candidates):
            frame = int(cand["frame"])
            cand_type = cand.get("type", "unknown")
            merged_types = set(cand.get("merged_types", [cand_type]))
            confidence = float(cand.get("confidence", 0.0))
            refine_score = float(cand.get("refine_score", 0.0))
            context_ratio = float(cand.get("context_ratio", 1.0))
            context_cross = float(cand.get("context_cross", 0.0))
            camera_explained = float(cand.get("camera_motion_explained", 0.0))
            camera_aligned = float(cand.get("camera_motion_aligned", 0.0))
            score = confidence

            if cand_type == "hard_cut" or "hard_cut" in merged_types:
                score += 0.30
            if len(merged_types) >= 2:
                score += 0.16
            if refine_score >= params["refine_floor"]:
                score += min(0.18, refine_score * 0.35)
            context_veto = "hard_cut" not in merged_types and context_ratio < 0.75 and context_cross < 0.08
            camera_veto = (
                    "hard_cut" not in merged_types and
                    camera_explained > 0.42 and
                    camera_aligned < 0.075
            )

            if context_ratio >= 1.8 and context_cross >= 0.05:
                score += min(0.16, (context_ratio - 1.0) * 0.06)
            elif "hard_cut" not in merged_types and context_ratio < 1.20:
                score -= 0.45
            if "hard_cut" not in merged_types and camera_explained > 0.25:
                score -= min(0.38, camera_explained * 0.45)
            if cand.get("source") == "fade":
                score += 0.08

            prev_frame = int(sorted_candidates[idx - 1]["frame"]) if idx > 0 else 0
            next_frame = int(sorted_candidates[idx + 1]["frame"]) if idx + 1 < len(sorted_candidates) else total_frames
            left_len = frame - prev_frame
            right_len = next_frame - frame
            if left_len < min_scene_frames or right_len < min_scene_frames:
                score -= 0.18

            threshold = params["accept"]
            if cand_type == "semantic" and len(merged_types) == 1:
                threshold = params["semantic"]
            elif cand_type == "gradual" and "hard_cut" not in merged_types:
                threshold = params["gradual"]
            if context_veto or camera_veto:
                threshold = max(threshold, score + 0.01)

            classified = dict(cand)
            classified["classifier_score"] = round(float(score), 4)
            classified["classifier_threshold"] = round(float(threshold), 4)
            classified["decision"] = "accepted" if score >= threshold else "rejected"

            if score >= threshold:
                accepted.append(classified)
            else:
                rejected.append(classified)

        return accepted, rejected

    def _merge_boundaries(self, hard_boundaries, gradual_boundaries, min_gap, total_frames):
        hard = sorted(set(int(b) for b in hard_boundaries if 0 < b < total_frames))
        gradual = sorted(set(int(b) for b in gradual_boundaries if 0 < b < total_frames))
        merged = list(hard)

        for cand in gradual:
            if any(abs(cand - h) <= min_gap for h in hard):
                continue
            near_idx = None
            near_dist = None
            for idx, existing in enumerate(merged):
                dist = abs(cand - existing)
                if dist <= min_gap and (near_dist is None or dist < near_dist):
                    near_idx = idx
                    near_dist = dist
            if near_idx is None:
                merged.append(cand)
            else:
                merged[near_idx] = int((merged[near_idx] + cand) / 2)

        return sorted(set(merged))

    def _merge_candidates(self, candidates, min_gap, total_frames):
        valid = [
            dict(c) for c in candidates
            if 0 < int(c.get("frame", 0)) < total_frames
        ]
        if not valid:
            return []

        priority = {"hard_cut": 3, "gradual": 2, "semantic": 1}
        valid.sort(key=lambda c: (
            -priority.get(c.get("type"), 0),
            -float(c.get("confidence", 0.0)),
            int(c["frame"])))

        merged = []
        for cand in valid:
            frame = int(cand["frame"])
            match_idx = None
            match_dist = None
            for idx, existing in enumerate(merged):
                dist = abs(frame - int(existing["frame"]))
                if dist <= min_gap and (match_dist is None or dist < match_dist):
                    match_idx = idx
                    match_dist = dist
            if match_idx is None:
                merged.append(cand)
                continue

            existing = merged[match_idx]
            cand_priority = priority.get(cand.get("type"), 0)
            existing_priority = priority.get(existing.get("type"), 0)
            if cand_priority > existing_priority:
                keeper, other = cand, existing
            elif cand_priority < existing_priority:
                keeper, other = existing, cand
            else:
                if cand.get("confidence", 0.0) > existing.get("confidence", 0.0):
                    keeper, other = cand, existing
                else:
                    keeper, other = existing, cand

            if keeper.get("type") == other.get("type") and abs(int(keeper["frame"]) - int(other["frame"])) <= min_gap:
                total_conf = max(1e-6, keeper.get("confidence", 0.0) + other.get("confidence", 0.0))
                keeper = dict(keeper)
                keeper["frame"] = int(round(
                    (int(keeper["frame"]) * keeper.get("confidence", 0.0) +
                     int(other["frame"]) * other.get("confidence", 0.0)) / total_conf))
            keeper = dict(keeper)
            keeper["score"] = max(float(keeper.get("score", 0.0)), float(other.get("score", 0.0)))
            keeper["confidence"] = max(float(keeper.get("confidence", 0.0)),
                                       float(other.get("confidence", 0.0)))
            keeper["merged_types"] = sorted(set(
                keeper.get("merged_types", [keeper.get("type")]) +
                other.get("merged_types", [other.get("type")])))
            merged[match_idx] = keeper

        return sorted(merged, key=lambda c: int(c["frame"]))

    def _scenes_from_boundaries(self, boundaries, total_frames):
        points = [0] + [b for b in boundaries if 0 < b < total_frames] + [total_frames]
        scenes = []
        for i in range(len(points) - 1):
            if points[i + 1] > points[i]:
                scenes.append((points[i], points[i + 1]))
        return scenes or [(0, total_frames)]

    def _fixed_interval(self):
        interval = self.cfg.get("FIXED_INTERVAL", 10)
        cmd = ["ffprobe", "-v", "error",
               "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        duration = float(check_output_hidden(cmd).decode().strip())
        fps = self._get_video_fps()
        self._total_frames = int(duration * fps)

        scenes, t = [], 0.0
        while t < duration:
            scenes.append((int(t * fps), int(min(t + interval, duration) * fps)))
            t += interval
        self.detected = len(scenes)
        return scenes

    def _record_cut_failure(self, task, error):
        idx, output_file, start_time, end_time, _aligned_start, _aligned_end, start_frame, end_frame, _precise = task
        message = str(error) or "Unknown error"
        failure = {
            "scene_number": int(idx),
            "file": os.path.basename(output_file),
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start_time": round(float(start_time), 4),
            "end_time": round(float(end_time), 4),
            "duration": round(max(0.0, float(end_time) - float(start_time)), 4),
            "error_type": type(error).__name__,
            "error": message[:1000],
        }
        self._cut_failures.append(failure)
        self.failed = len(self._cut_failures)
        return failure

    def _write_cut_failure_report(self, outdir):
        if not self._cut_failures:
            return None
        path = os.path.join(outdir, "cut_errors.json")
        report = {
            "video": self.video,
            "failed_scene_count": len(self._cut_failures),
            "failed_duration": round(sum(float(f.get("duration", 0.0)) for f in self._cut_failures), 4),
            "failed_scenes": self._cut_failures,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return path

    def _validate_cut_output(self, output, expected_duration):
        if not os.path.exists(output):
            raise RuntimeError("Output file was not created")
        if not is_valid_video_file(output):
            raise RuntimeError("Output file has no valid video stream")
        cmd = ["ffprobe", "-v", "error",
               "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", output]
        try:
            actual_duration = float(check_output_hidden(cmd).decode().strip())
        except Exception as e:
            raise RuntimeError(f"Could not validate output duration: {e}") from e
        if expected_duration > 0.2 and actual_duration < expected_duration * 0.5:
            raise RuntimeError(
                f"Output duration is too short ({actual_duration:.2f}s of {expected_duration:.2f}s)")
        return actual_duration

    def _cut_scenes(self, scenes):
        outdir = build_output_dir(
            self.output,
            mode="scene" if self.scene_mode else "interval",
            profile=self.cfg.get("label", "NA"),
            accel=self.cfg.get("ENCODER", "cpu")
        )
        self._cut_output_dir = outdir
        self._cut_failures = []
        self.failed = 0
        fps = self._get_video_fps()
        self.total = len(scenes)
        self.done = 0
        self.completed_attempts = 0
        self._cut_lock = threading.Lock()

        # Pre-compute all tasks (decoding is sequential: keyframes/fps first)
        keyframes = self._get_keyframes() or [0.0]
        tasks = []
        for i, (start_frame, end_frame) in enumerate(scenes):
            start_time = start_frame / fps
            end_time = end_frame / fps
            output_file = os.path.join(outdir, f"scene_{i:03d}.mp4")
            aligned_start = self._nearest_keyframe_before(start_time, keyframes)
            aligned_end = self._nearest_keyframe_after(end_time, keyframes)
            start_delta = abs(start_time - aligned_start)
            end_delta = abs(end_time - aligned_end)
            needs_precise_cut = (
                    self.scene_mode or
                    start_delta > (1 / fps) or
                    end_delta > (1 / fps)
            )
            tasks.append((
                i, output_file, start_time, end_time,
                aligned_start, aligned_end, int(start_frame), int(end_frame),
                needs_precise_cut
            ))

        def _cut_one(task):
            idx, output_file, start_time, end_time, aligned_start, aligned_end, start_frame, end_frame, precise = task
            if precise:
                self._run_ffmpeg_precise_cut(
                    self.video, start_frame, end_frame, output_file, aligned_start)
            else:
                self._run_ffmpeg_copy(start_time, end_time, output_file)
            self._validate_cut_output(output_file, end_time - start_time)

        if self.total <= 1:
            try:
                _cut_one(tasks[0])
                self.done = 1
            except Exception as cut_error:
                failure = self._record_cut_failure(tasks[0], cut_error)
                if self.log:
                    err_msg = str(cut_error)[:80] or "Unknown error"
                    err_type = failure["error_type"]
                    self.log.after(0, lambda e=err_msg, t=err_type: self.log.append_message(
                        f"Error: scene_000 could not be exported [{t}] ({e})", kind="error"))
            self.completed_attempts = 1
            if self.log:
                self.log.after(0, lambda: self.log.write_status(detected=self.detected, cut=self.done, eta="00:00"))
            if self.progress:
                self.progress.after(0, lambda v=1.0: self.progress.update(v))
        else:
            # Use ThreadPoolExecutor for concurrent cuts
            with ThreadPoolExecutor(max_workers=min(MAX_CUT_WORKERS, self.total)) as pool:
                futures = {pool.submit(_cut_one, task): task for task in tasks}
                for future in as_completed(futures):
                    if self._stop:
                        pool.shutdown(wait=False)
                        return
                    try:
                        future.result()  # Catch any exceptions from workers
                        with self._cut_lock:
                            self.done += 1
                    except Exception as cut_error:
                        task = futures[future]
                        failure = self._record_cut_failure(task, cut_error)
                        print(f"[CUT ERROR] {cut_error}")
                        if self.log:
                            err_msg = str(cut_error)[:60]
                            scene_name = failure["file"]
                            err_type = failure["error_type"]
                            self.log.after(0, lambda n=scene_name, e=err_msg, t=err_type: self.log.append_message(
                                f"Warning: {n} could not be exported [{t}] ({e})", kind="warning"))
                        # Continue with other cuts

                    with self._cut_lock:
                        self.completed_attempts += 1
                        if self.progress:
                            ratio = self.completed_attempts / self.total
                            self.progress.after(0, lambda v=ratio: self.progress.update(v))
                        if self.log:
                            eta = self._calculate_eta()
                            self.log.after(0, lambda d=self.detected, c=self.done, e=eta:
                            self.log.write_status(d, c, e))

        self._write_scene_metadata(outdir, scenes, fps)
        self._write_cut_failure_report(outdir)

        if self.done <= 0 and self._cut_failures and not self._stop:
            raise RuntimeError("No scenes could be exported. See cut_errors.json for details.")

    def _write_scene_metadata(self, outdir, scenes, fps):
        try:
            candidates_by_frame = {
                int(c.get("frame", -1)): c for c in getattr(self, "_scene_candidates", [])
            }
            metadata = {
                "video": self.video,
                "profile": self.cfg.get("label", "NA"),
                "fps": fps,
                "scene_count": len(scenes),
                "successful_cuts": int(self.done),
                "failed_cuts": getattr(self, "_cut_failures", []),
                "accepted_cuts": getattr(self, "_scene_candidates", []),
                "rejected_cuts": getattr(self, "_rejected_scene_candidates", []),
                "scenes": [],
            }
            failures_by_index = {
                int(f.get("scene_number", -1)): f for f in getattr(self, "_cut_failures", [])
            }
            for idx, (start_frame, end_frame) in enumerate(scenes):
                cut_candidate = candidates_by_frame.get(int(start_frame)) if idx > 0 else None
                failure = failures_by_index.get(idx)
                metadata["scenes"].append({
                    "index": idx,
                    "file": f"scene_{idx:03d}.mp4",
                    "status": "failed" if failure else "exported",
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "start_time": round(start_frame / fps, 4),
                    "end_time": round(end_frame / fps, 4),
                    "duration": round((end_frame - start_frame) / fps, 4),
                    "cut": cut_candidate,
                    "error": failure,
                })
            with open(os.path.join(outdir, "scenes.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Failed to write scene metadata: {e}")

    def _get_progress_ratio(self):
        if self.total > 0:
            return self.done / self.total
        return self._analysis_ratio

    def _calculate_eta(self):
        """Calculate ETA - only during cut phase, not during detection.

        Returns "--:--" during detection phase to avoid showing stale/misleading estimates.
        Shows real ETA based on actual cut rate during cutting phase.
        """
        elapsed = time.time() - self._start_time
        if elapsed < 1:
            return "--:--"

        # Only show ETA during cut phase (when we have actual cut progress)
        completed = getattr(self, "completed_attempts", self.done)
        if self.total > 0 and completed > 0:
            rate = completed / elapsed  # scene attempts per second
            remaining = self.total - completed
            eta_seconds = int(remaining / rate) if rate > 0 else 0
            if 0 < eta_seconds < 86400:
                m, s = divmod(eta_seconds, 60)
                h, m = divmod(m, 60)
                return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        # Detection phase: don't show ETA (it will be reset anyway)
        return "--:--"

    def _get_video_fps(self):
        if self._fps is not None:
            return self._fps
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=avg_frame_rate",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        num, den = check_output_hidden(cmd).decode().strip().split("/")
        self._fps = float(num) / float(den)
        return self._fps

    def _map_threshold(self):
        """Returns (threshold, min_scene_len) for the ContentDetector.

        Hybrid approach: FIXED threshold per profile + ADAPTIVE min_dur.

        Pipeline:
          1. _adaptive_threshold() scans the video -> returns (raw_t, raw_d)
             - raw_t: pacing score (15-55, higher = more motion) — USED ONLY for Auto
             - raw_d: adaptive minimum scene duration (1-10s)

          2. Profile selects a FIXED threshold (PySceneDetect content_val scale):
             - Low    (42) -> conservative, clear scene/location changes
             - Normal (27) -> balanced, standard scene breaks
             - High   (18) -> sensitive, trailers/shorts/action

          3. Profile scales the adaptive min_dur:
             - Low    -> longer minimum scenes -> fewer false positives
             - Normal -> baseline
             - High   -> shorter minimum scenes -> catches fast cuts

        PySceneDetect ContentDetector constraints:
          - threshold: content_val scale (0-255), default=27.0
          - min_scene_len: in frames, default=15, minimum=1
        """
        raw_t, raw_d = _adaptive_threshold(self.video)

        if self.cfg.get("ADAPTIVE"):
            # Auto: map adaptive threshold to PySceneDetect scale.
            # raw_t is on 15-55 scale, PySceneDetect uses content_val (~5-60 typical).
            # Map linearly: 15->10, 35->27, 55->45
            threshold = 10.0 + (raw_t - 15.0) * (35.0 / 40.0)
            threshold = max(10.0, min(50.0, threshold))
            threshold = round(threshold, 1)
            return threshold, raw_d

        # FIXED thresholds per profile — calibrated against PySceneDetect defaults
        # and validated with real-world trailer/film/documentary tests:
        #   Low    (42) -> conservative, only clear scene/location changes
        #   Normal (27) -> balanced, standard scene breaks
        #   High   (18) -> sensitive, catches fast cuts
        #
        # min_dur: profile sets the FLOOR, adaptive scan can only raise it.
        profiles = {
            "Low": {"base_threshold": 42, "min_dur": 5.0, "dur_max_boost": 5.0, "label": "Low"},
            "Normal": {"base_threshold": 27, "min_dur": 2.0, "dur_max_boost": 4.0, "label": "Normal"},
            "High": {"base_threshold": 18, "min_dur": 0.7, "dur_max_boost": 2.5, "label": "High"},
        }

        profile = "Normal"
        for key, val in profiles.items():
            if self.cfg.get("label") == key:
                profile = key
                break

        cfg = profiles[profile]
        threshold = float(cfg["base_threshold"])

        # Adaptive min_dur: profile sets the FLOOR, scan raises it.
        # raw_d range: 1.0 (most agitated) to 10.0 (most calm).
        # boost_ratio = 0.0 for agitated → 1.0 for calm.
        # min_dur = floor + boost_ratio * max_boost
        base_dur = cfg["min_dur"]
        max_boost = cfg["dur_max_boost"]

        boost_ratio = (raw_d - 1.0) / 9.0  # Normalize to [0.0, 1.0]
        min_dur = base_dur + boost_ratio * max_boost

        # Clamp to safe bounds
        min_dur = max(0.5, min(15.0, min_dur))
        min_dur = round(min_dur, 1)

        # DEBUG
        if cfg.get("DEBUG", False):
            try:
                fps_est = self._get_video_fps()
            except Exception:
                fps_est = 30.0
            print(f"[DEBUG] Adaptive scan: raw_t={raw_t}, raw_d={raw_d}")
            print(
                f"[DEBUG] Profile={profile}, threshold={threshold}, min_dur_floor={cfg['min_dur']}, max_boost={cfg['dur_max_boost']}")
            print(f"[DEBUG] min_dur={min_dur}, min_scene_len={int(min_dur * fps_est)} frames (at ~{fps_est}fps)")

        return float(threshold), min_dur

    def _get_video_duration(self):
        if self._duration is not None:
            return self._duration
        cmd = ["ffprobe", "-v", "error",
               "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        self._duration = float(check_output_hidden(cmd).decode().strip())
        return self._duration

    def _get_keyframes(self):
        if self._keyframes_cache is not None:
            return self._keyframes_cache
        cmd = ["ffprobe", "-select_streams", "v", "-skip_frame", "nokey",
               "-show_entries", "frame=pkt_pts_time,best_effort_timestamp_time",
               "-loglevel", "error", "-of", "csv=p=0", self.video]
        result = run_hidden(cmd, capture_output=True, text=True)
        keyframes = []
        for line in result.stdout.splitlines():
            for part in line.replace(",", " ").split():
                try:
                    keyframes.append(float(part.strip()))
                    break
                except Exception:
                    continue
        keyframes = sorted(set(keyframes))
        self._keyframes_cache = keyframes
        return keyframes

    def _run_ffmpeg_tracked(self, cmd, timeout=300):
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW
        )
        self._ffmpeg_proc = proc
        with self._ffmpeg_proc_lock:
            self._ffmpeg_procs.add(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            stdout, stderr = proc.communicate()
            return subprocess.CompletedProcess(cmd, -9, stdout, stderr)
        finally:
            with self._ffmpeg_proc_lock:
                self._ffmpeg_procs.discard(proc)
            if self._ffmpeg_proc is proc:
                self._ffmpeg_proc = None

    def _run_ffmpeg_copy(self, start, end, output):
        duration = end - start
        cmd = ["ffmpeg", "-y", "-ss", f"{start:.6f}",
               "-i", self.video, "-t", f"{duration:.6f}",
               "-c", "copy", "-avoid_negative_ts", "make_zero",
               "-muxpreload", "0", "-muxdelay", "0", output]
        result = self._run_ffmpeg_tracked(cmd, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="ignore")[:300] or "FFmpeg copy failed")

    def _has_audio_stream(self):
        if self._has_audio_cache is not None:
            return self._has_audio_cache
        cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
               "-show_entries", "stream=index",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        try:
            self._has_audio_cache = bool(check_output_hidden(cmd).decode().strip())
        except Exception:
            self._has_audio_cache = False
        return self._has_audio_cache

    def _run_ffmpeg_precise_cut(self, input_file, start_frame, end_frame, output, aligned_start):
        encoder = self.cfg.get("ENCODER", "cpu")
        bitrate = self._get_video_bitrate()
        fps = self._get_video_fps()
        start_time = start_frame / fps
        end_time = end_frame / fps
        local_start_frame = max(0, int(round((start_time - aligned_start) * fps)))
        local_end_frame = max(local_start_frame + 1, int(round((end_time - aligned_start) * fps)))
        local_audio_start = max(0.0, start_time - aligned_start)
        local_audio_end = max(local_audio_start + (1.0 / fps), end_time - aligned_start)

        if encoder == "cpu":
            codec = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-tune", "film"]
        elif encoder == "nvidia":
            codec = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19",
                     "-rc", "vbr", "-b:v", "0", "-spatial_aq", "1",
                     "-temporal_aq", "1", "-rc-lookahead", "20"]
        elif encoder == "amd":
            codec = ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "vbr_peak"]
        elif encoder == "intel":
            codec = ["-c:v", "h264_qsv", "-preset", "medium", "-global_quality", "20"]
        else:
            codec = ["-c:v", "libx264", "-crf", "18"]

        rate_opts = []
        if bitrate and encoder == "cpu":
            rate_opts = ["-maxrate", str(bitrate), "-bufsize", str(bitrate * 2)]

        vf = f"trim=start_frame={local_start_frame}:end_frame={local_end_frame},setpts=PTS-STARTPTS"
        if self._has_audio_stream():
            filter_complex = (
                f"[0:v]{vf}[v];"
                f"[0:a]atrim=start={local_audio_start:.6f}:end={local_audio_end:.6f},"
                f"asetpts=PTS-STARTPTS[a]"
            )
            maps = ["-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]"]
            audio_opts = ["-c:a", "aac", "-b:a", "192k"]
        else:
            maps = ["-vf", vf]
            audio_opts = ["-an"]

        cmd = ["ffmpeg", "-y", "-ss", f"{aligned_start:.6f}", "-i", input_file,
               *maps, *codec, *rate_opts,
               "-pix_fmt", "yuv420p", *audio_opts,
               "-movflags", "+faststart", output]

        result = self._run_ffmpeg_tracked(cmd, timeout=300)

        if result.returncode != 0 and encoder != "cpu":
            codec_fallback = ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]
            cmd = ["ffmpeg", "-y", "-ss", f"{aligned_start:.6f}", "-i", input_file,
                   *maps, *codec_fallback,
                   "-pix_fmt", "yuv420p", *audio_opts,
                   "-movflags", "+faststart", output]
            result2 = self._run_ffmpeg_tracked(cmd, timeout=300)
            if result2.returncode == 0:
                return
            print("FFMPEG ERROR:\n", result.stderr.decode(errors="ignore"))
            if self.log:
                err_msg = result.stderr.decode(errors="ignore")[:80]
                self.log.after(0, lambda e=err_msg: self.log.append_message(
                    f"FFmpeg error: {e}", kind="error"))
            raise RuntimeError(result2.stderr.decode(errors="ignore")[:300] or "FFmpeg fallback failed")
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="ignore")[:300] or "FFmpeg precise cut failed")

    def _nearest_keyframe_before(self, t, keyframes):
        idx = bisect.bisect_right(keyframes, t)
        return keyframes[idx - 1] if idx > 0 else 0.0

    def _nearest_keyframe_after(self, t, keyframes):
        if not keyframes:
            return t
        idx = bisect.bisect_left(keyframes, t)
        if idx < len(keyframes):
            return keyframes[idx]
        return min(keyframes[-1], t)

    def _get_video_bitrate(self):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=bit_rate",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        try:
            return int(check_output_hidden(cmd).decode().strip())
        except Exception:
            return None

    def _preview_loop(self):
        while not self._preview_stop:
            if self.previewer and self.preview_enabled:
                now = time.time()
                if now - self.last_preview >= PREVIEW_INTERVAL:
                    try:
                        ratio = self._get_progress_ratio()
                        ratio = max(0.0, min(1.0, ratio))
                        if abs(ratio - self._last_preview_ratio) < 0.02:
                            time.sleep(0.02)
                            continue
                        self._last_preview_ratio = ratio
                        duration = self._get_video_duration()
                        t = ratio * duration
                        img = self._get_preview_frame_at(t)
                        if img and self._ui_alive:
                            self.previewer.after(
                                0, lambda img=img: (
                                    self.previewer.update_image(img)
                                    if self._ui_alive else None))
                    except Exception:
                        pass
                    self.last_preview = now
            time.sleep(0.05)

    def _get_preview_frame_at(self, t):
        try:
            with self._preview_lock:
                if not self._preview_cap:
                    return None
                self._preview_cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = self._preview_cap.read()
            if not ret or frame is None:
                return None
            return resize_for_preview(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        except Exception:
            return None


# ============================================================
# FaceDetectionEngine (heavy deps loaded lazily)
# ============================================================
class FaceDetectionEngine:
    def __init__(self, video, output, logbox=None, progressbar=None,
                 previewer=None, profile="Normal", accel="cpu", preview_enabled=True):
        self.video = video
        self.output = output
        self.log = logbox
        self.progress = progressbar
        self.previewer = previewer
        self.preview_enabled = preview_enabled
        self._ui_alive = True
        self._last_eta_update = 0  # FIX: Initialize for ETA throttling

        self.profile = profile
        self.accel = accel
        self._stop = False
        self._start_time = None
        self._end_time = None
        self.detected = 0
        self.done = 0
        self._face_ratio = 0.0

        # Lazy-loaded (initialized in run() on worker thread)
        self.device = None
        self.model = None
        self.mp_face = None
        self.torch = None
        self.YOLO = None
        self.mp = None

        self.profile_cfg = {
            "Low": {"conf": 0.45, "min_size": 64, "ttl": 0.6,
                    "min_frames": 0.8, "min_valid_ratio": 0.75, "min_sharpness": 60},
            "Normal": {"conf": 0.35, "min_size": 40, "ttl": 1.2,
                       "min_frames": 0.5, "min_valid_ratio": 0.6, "min_sharpness": 40},
            "High": {"conf": 0.22, "min_size": 24, "ttl": 2.5,
                     "min_frames": 0.25, "min_valid_ratio": 0.35, "min_sharpness": 20},
        }[profile]

        self.last_preview = 0
        self._cap = None  # cv2.VideoCapture for cleanup

    def _load_deps(self):
        """Load heavy dependencies on worker thread to avoid blocking UI."""
        if not _ensure_torch():
            raise RuntimeError("Face detection requires PyTorch, but it is not installed.")

        YOLO = _ensure_yolo()
        if YOLO is None:
            raise RuntimeError("ultralytics package not found.")

        mp = _ensure_mediapipe()
        if mp is None:
            raise RuntimeError("mediapipe package not found.")

        self.torch = torch
        self.YOLO = YOLO
        self.mp = mp

        self.device = "cuda:0" if self.accel == "nvidia" and torch.cuda.is_available() else "cpu"

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", "yolov8n-face.pt")
        self.model = YOLO(model_path)

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=2, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def stop(self):
        self._stop = True
        self._ui_alive = False
        # Release cv2.VideoCapture if open
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        # Delete heavy objects to free VRAM/RAM
        self.model = None
        if self.mp_face:
            try:
                self.mp_face.close()
            except Exception:
                pass
        self.mp_face = None
        # Clear GPU memory if torch was loaded
        if self.torch is not None:
            try:
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
            except Exception:
                pass
        self.torch = None
        self.device = None

    def total_time(self):
        if not self._start_time:
            return "--:--"
        end = self._end_time or time.time()
        elapsed = max(1, int(end - self._start_time))
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _skin_ratio(self, face):
        ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        return skin.mean() / 255

    def _valid_landmarks(self, face):
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        result = self.mp_face.process(rgb)
        if not result.multi_face_landmarks:
            return False
        lm = result.multi_face_landmarks[0].landmark
        left_eye, right_eye, nose, mouth = lm[33], lm[263], lm[1], lm[13]
        if not (left_eye.y < nose.y < mouth.y):
            return False
        if abs(left_eye.x - right_eye.x) < (0.020 if self.profile == "High" else 0.030):
            return False
        return True

    def run(self):
        # Load heavy dependencies on worker thread (non-blocking for UI)
        self._load_deps()

        self._start_time = time.time()
        cap = cv2.VideoCapture(self.video)
        self._cap = cap  # Store for cleanup
        try:  # FIX: ensure cap is always released
            fps_raw = cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps_raw) if fps_raw and fps_raw > 0 else 30.0
            total_frames_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            try:
                total_frames = int(float(total_frames_raw))
            except Exception:
                total_frames = 1

            ttl_frames = int(fps * self.profile_cfg["ttl"])
            tracks = []
            outdir = build_output_dir(self.output, mode="faces",
                                      profile=self.profile, accel=self.accel)
            frame_idx = 0
            track_id = 0
            min_lm_frames = 1 if self.profile == "High" else 2
            iou_thresh = 0.45

            while cap.isOpened():
                if self._stop:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                use_cuda = self.device.startswith("cuda")
                results = self.model.predict(
                    frame, conf=self.profile_cfg["conf"], iou=iou_thresh,
                    imgsz=640 if frame.shape[1] <= 1280 else 800,
                    device=self.device, half=use_cuda, verbose=False)[0]

                new_tracks = []
                for box in results.boxes.xyxy:
                    box = box.squeeze()
                    x1, y1, x2, y2 = map(int, box.tolist())
                    face_raw = frame[y1:y2, x1:x2]
                    w, h = x2 - x1, y2 - y1
                    if w < self.profile_cfg["min_size"] or h < self.profile_cfg["min_size"]:
                        continue
                    aspect = w / h
                    if not (0.65 <= aspect <= 1.35):
                        continue

                    h_frame, w_frame, _ = frame.shape
                    expand_x = int(w * 0.15)
                    expand_y = int(h * 0.15)  # FIX: simplify duplicate variables
                    cx1 = max(0, x1 - expand_x)
                    cy1 = max(0, y1 - expand_y)
                    cx2 = min(w_frame, x2 + expand_x)
                    cy2 = min(h_frame, y2 + expand_y)

                    skin_min = 0.12 if self.profile != "High" else 0.10
                    face_crop = frame[cy1:cy2, cx1:cx2]
                    if face_raw.size == 0 or self._skin_ratio(face_raw) < skin_min:
                        continue

                    matched = False
                    for t in tracks:
                        if self._iou(t["box"], (x1, y1, x2, y2)) > iou_thresh:
                            t["box"] = (x1, y1, x2, y2)
                            t["ttl"] = ttl_frames
                            t["frames"] += 1

                            fh, fw = face_raw.shape[:2]
                            center_face = face_raw[int(fh * 0.25):int(fh * 0.75),
                            int(fw * 0.25):int(fw * 0.75)]
                            if center_face.size == 0:
                                center_face = face_raw
                            sharp = cv2.Laplacian(center_face, cv2.CV_64F).var()
                            if sharp > t["score"]:
                                t["score"] = sharp
                                t["face"] = face_crop.copy()
                            if t["frames"] >= min_lm_frames and self._valid_landmarks(face_raw):
                                t["valid"] = min(t["valid"] + 1, t["frames"])
                            matched = True
                            break

                    if not matched:
                        track_id += 1
                        fh, fw = face_raw.shape[:2]
                        center_face = face_raw[int(fh * 0.25):int(fh * 0.75),
                        int(fw * 0.25):int(fw * 0.75)]
                        if center_face.size == 0:
                            center_face = face_raw
                        sharp = cv2.Laplacian(center_face, cv2.CV_64F).var()
                        new_tracks.append({
                            "id": track_id,
                            "box": (x1, y1, x2, y2),
                            "ttl": ttl_frames,
                            "frames": 1,
                            "valid": 0,
                            "score": sharp,  # FIX: reuse calculated sharp value
                            "face": face_crop.copy()
                        })

                for t in tracks:
                    t["ttl"] -= 1
                    if t["ttl"] <= 0:
                        cfg = self.profile_cfg
                        min_required = max(3, int(fps * cfg["min_frames"]))
                        if (t["frames"] >= min_required and
                                (t["valid"] / max(t["frames"], 1)) >= cfg["min_valid_ratio"] and
                                t["score"] >= cfg["min_sharpness"]):
                            fname = f"face_{self.done + 1:04d}.png"
                            path = os.path.join(outdir, fname)
                            if cv2.imwrite(path, t["face"]):
                                self.done += 1
                                self.detected += 1

                tracks = [t for t in tracks if t["ttl"] > 0] + new_tracks

                if self.previewer and self.preview_enabled:
                    now = time.time()
                    if now - self.last_preview >= PREVIEW_INTERVAL:
                        draw = frame.copy()
                        for t in tracks:
                            x1, y1, x2, y2 = t["box"]
                            cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        img = resize_for_preview(
                            Image.fromarray(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)))
                        if img and self._ui_alive:
                            self.previewer.after(
                                0, lambda img=img: (
                                    self.previewer.update_image(img)
                                    if self._ui_alive else None))
                        self.last_preview = now

                # FIX: Use after() for thread-safe UI updates
                # Only update ETA every ~0.5 seconds to avoid wasteful calculations
                if self.log:
                    now = time.time()
                    if now - getattr(self, "_last_eta_update", 0) >= 0.5:
                        self._last_eta_update = now
                        detected_val = self.detected
                        done_val = self.done
                        eta_val = self._calculate_eta(frame_idx, total_frames)
                        self.log.after(0, lambda d=detected_val, c=done_val, e=eta_val:
                        self.log.write_status(detected=d, cut=c, eta=e))

                # FIX: Use after() for thread-safe UI updates
                if self.progress:
                    ratio = max(self._face_ratio, frame_idx / total_frames)
                    self._face_ratio = ratio
                    progress_ratio = ratio
                    if self._ui_alive:
                        self.progress.after(0, lambda v=progress_ratio: self.progress.update(v))

            # Process remaining tracks
            cfg = self.profile_cfg
            for t in tracks:
                min_required = max(3, int(fps * cfg["min_frames"]))
                if (t["frames"] >= min_required and
                        (t["valid"] / max(t["frames"], 1)) >= cfg["min_valid_ratio"] and
                        t["score"] >= cfg["min_sharpness"]):
                    fname = f"face_{self.done + 1:04d}.png"
                    path = os.path.join(outdir, fname)
                    if cv2.imwrite(path, t["face"]):
                        self.done += 1
                        self.detected += 1
        finally:
            # FIX: always release cap
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            self._cap = None
            # Clear GPU memory after face detection
            if self.torch is not None:
                try:
                    if self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
                except Exception:
                    pass

        self._end_time = time.time()
        return not self._stop

    def _calculate_eta(self, frame_idx, total_frames):
        """Calculate ETA for face detection - optimized and realistic.

        Based on frame processing rate with sanity checks.
        """
        if frame_idx == 0 or total_frames == 0:
            return "--:--"
        elapsed = time.time() - self._start_time
        if elapsed < 1:
            return "--:--"

        # Frames per second
        rate = frame_idx / elapsed
        remaining = total_frames - frame_idx
        eta_seconds = int(remaining / rate) if rate > 0 else 0

        # Sanity check
        if eta_seconds < 0 or eta_seconds > 86400:
            return "--:--"

        m, s = divmod(eta_seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ============================================================
# SceneCutterApp (main UI)
# ============================================================
class SceneCutterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Scenespy - Scene Cutter")
        self.geometry("1000x650")
        self.engine = None
        self.running = False
        self.stop_pending = False
        self.batch_stop = False
        self.resizable(False, False)
        self.preview_enabled = ENABLE_PREVIEW_DEFAULT
        self.available_accel = detect_available_accel()

        # Load saved settings
        self.saved_settings = load_settings()

        self._build_ui()

    def _build_ui(self):
        # Left panel
        self.left = ctk.CTkFrame(
            self, width=420, fg_color=BG_PANEL, border_width=1,
            border_color=BORDER_SOFT2, corner_radius=0)
        self.left.pack(side="left", fill="y", padx=10, pady=10)

        files = Section(self.left, "Files")
        files.pack(fill="x", padx=12, pady=12)
        self.video_selector = FileSelector(files, "Source video(s)")
        self.video_selector.pack(fill="x", padx=12)
        self.output_selector = DirectorySelector(files, "Output folder")
        self.output_selector.pack(fill="x", padx=12)

        # Restore saved paths
        last_video = self.saved_settings.get("last_video", "")
        last_output = self.saved_settings.get("last_output", "")
        if last_video:
            self.video_selector.entry.insert(0, last_video)
            self.video_selector.paths = [last_video]
        if last_output:
            self.output_selector.entry.insert(0, last_output)

        # Cut mode
        mode = Section(self.left, "Cut Mode")
        mode.pack(fill="x", padx=12)

        self.cut_mode = ctk.StringVar(value="scene")
        self.cut_mode.trace_add("write", self._on_cut_mode_change)

        group = RadioGroup(mode, self.cut_mode, [
            ("Scene detection", "scene"),
            ("Every seconds", "interval"),
            ("Detect faces", "faces"),
        ], radio_width=150)
        group.pack(fill="x", padx=12, pady=(0, 6))
        self.mode_radios = group.radios

        if not TORCH_AVAILABLE:
            for rb in self.mode_radios:
                if rb.cget("value") == "faces":
                    rb.configure(state="disabled")

        vcmd = (self.register(self._validate_interval), "%P")
        self.interval_entry = ctk.CTkEntry(
            mode, height=25, width=90, fg_color=BG_MAIN,
            border_color=BORDER_SOFT, border_width=1, corner_radius=15,
            text_color="#ededed", font=("Consolas", 11),
            placeholder_text_color=TEXT_MUTED, placeholder_text="Seconds",
            validate="key", validatecommand=vcmd)
        self.interval_entry.pack(anchor="w", padx=(186, 0), pady=(0, 8))

        # Profile
        profile_sec = Section(self.left, "Detection Sensitivity")
        profile_sec.pack(fill="x", padx=12, pady=5)

        self.profile = ctk.StringVar(value="Normal")
        options = [(cfg["label"], key) for key, cfg in PROFILES.items()]
        group2 = RadioGroup(profile_sec, self.profile, options, radio_width=110)
        group2.pack(fill="x", padx=12)
        self.profile_radios = group2.radios

        # Acceleration
        accel_sec = Section(self.left, "Hardware Acceleration (Inference)")
        accel_sec.pack(fill="x", padx=12)
        self.accel = ctk.StringVar(value="cpu")
        group3 = RadioGroup(accel_sec, self.accel,
                            [(val.upper(), val) for val in ACCEL_OPTIONS],
                            radio_width=110)
        group3.pack(fill="x", padx=12)
        self.accel_radios = group3.radios
        self.update_accel_radios()

        # Start button
        self.start_btn = ctk.CTkButton(
            self.left, text="Start", height=13, corner_radius=420,
            fg_color=ACCENT, hover_color="#4f46e5", text_color="white",
            command=self.toggle_start)
        self.start_btn.pack(pady=(20, 10))

        self.log = LogBox(self.left, height=220)
        self.log.pack(fill="x", padx=10, pady=10)

        # Right panel
        self.right = ctk.CTkFrame(
            self, width=250, fg_color=BG_PANEL, border_width=1,
            border_color=BORDER_SOFT2, corner_radius=15)
        self.right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        section = Section(self.right, "Preview")
        section.pack(padx=12, pady=12, fill="x")
        self.preview_switch = ctk.CTkSwitch(
            section, text="Show Thumbnail", command=self.toggle_preview)
        self.preview_switch.pack(anchor="e", padx=10, pady=8)

        if self.preview_enabled:
            self.preview_switch.select()
            self.toggle_preview()

        self.preview_frame = PreviewFrame(self.right)
        self.preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.progress = ProgressBar(self.right)
        self.progress.pack(fill="x", padx=20, pady=10)

        self.left.pack_propagate(False)
        self.right.pack_propagate(False)
        self._on_cut_mode_change()

    # --- App callbacks ---
    def toggle_preview(self):
        self.preview_enabled = self.preview_switch.get()
        if not self.preview_enabled:
            self.preview_frame.clear_all()

    def toggle_start(self):
        if self.running:
            self.confirm_stop()
        else:
            if self.engine is not None:
                return
            self.start_process()

    def start_process(self):
        videos = self.video_selector.get_paths()
        output = self.output_selector.get().strip()
        first_video = videos[0] if videos else ""

        # Save paths for next session
        save_settings(video=first_video, output=output)

        if not videos and not output:
            return
        if not videos or not output:
            self.log.clear_status()
            self.log.status_lines[0] = "Select input video and output folder"
            self.log._render()
            return
        if not os.path.isdir(output):
            self.log.clear_status()
            self.log.status_lines[0] = "Invalid output folder"
            self.log._render()
            return

        valid_videos = []
        skipped = []
        for video in videos:
            name_no_ext = os.path.splitext(os.path.basename(video))[0].lower()
            if name_no_ext.endswith("_fixed") or "_fixed_fixed" in name_no_ext:
                skipped.append((video, "Temporary repaired file"))
                continue
            ext = os.path.splitext(video)[1].lower()
            if ext not in ALLOWED_VIDEO_EXTENSIONS:
                skipped.append((video, "Unsupported file type"))
                continue
            if not os.path.isfile(video):
                skipped.append((video, "File does not exist"))
                continue
            if not is_valid_video_file(video):
                skipped.append((video, "Invalid or unsupported video file"))
                continue
            valid_videos.append(video)

        self.cleanup_process(reason="reset")
        if self.log:
            self.log.clear_status()
            if len(videos) > 1:
                self.log.append_message(
                    f"Queue: {len(valid_videos)} videos ready, {len(skipped)} skipped",
                    kind="info")
            for video, reason in skipped[:8]:
                self.log.append_message(
                    f"Skipped: {os.path.basename(video)} ({reason})", kind="warning")
            if len(skipped) > 8:
                self.log.append_message(
                    f"Skipped: {len(skipped) - 8} more invalid file(s).", kind="warning")

        if not valid_videos:
            if self.log:
                self.log.append_message("Error: no valid videos to process.", kind="error")
            return

        self.running = True
        self.batch_stop = False
        self.set_ui_state(True)
        self.after(0, self._finalize_start_ui)

        cfg = PROFILES[self.profile.get()].copy()
        requested = self.accel.get()
        mode = self.cut_mode.get()
        compat = MODE_ACCEL_COMPAT.get(mode, {})

        encoder_allowed = compat.get("encoder", {"cpu"}) & self.available_accel
        inference_allowed = compat.get("inference", {"cpu"}) & self.available_accel

        encoder = requested if requested in encoder_allowed else "cpu"
        inference = requested if requested in inference_allowed else "cpu"

        cfg["ENCODER"] = encoder
        cfg["INFERENCE"] = inference
        cfg["DEBUG"] = DEBUG

        scene_mode = mode == "scene"
        if mode == "interval":
            value = self.interval_entry.get()
            if not value:
                self.log.clear_status()
                self.log.status_lines[0] = "Interval cannot be empty!"
                self.log.write_status()
                self.reset_ui()
                return
            cfg["FIXED_INTERVAL"] = int(value)

        threading.Thread(
            target=self.run_batch,
            args=(valid_videos, output, cfg, mode, scene_mode, inference),
            daemon=True
        ).start()

    def stop_process(self):
        self.batch_stop = True
        if not self.engine:
            return
        self.engine.stop()

    def _cooldown_between_videos(self):
        gc.collect()
        time.sleep(PROCESS_COOLDOWN_SECONDS)
        gc.collect()

    def run_batch(self, videos, output, cfg, mode, scene_mode, inference):
        started = time.time()
        completed = 0
        failed = 0
        partial = 0
        stopped = False

        try:
            for index, video in enumerate(videos, start=1):
                if self.batch_stop or self.stop_pending:
                    stopped = True
                    break

                basename = os.path.basename(video)
                if self.log and len(videos) > 1:
                    self.after(0, lambda i=index, n=len(videos), b=basename: self.log.append_message(
                        f"Processing {i}/{n}: {b}", kind="info"))
                if self.progress:
                    self.after(0, self.progress.reset)

                result = False
                error_msg = None
                error_type = None
                warning_msg = None
                temp_files = []
                engine = None
                try:
                    current_video = prepare_video_for_processing(video, temp_files=temp_files)
                    if mode == "faces":
                        engine = FaceDetectionEngine(
                            current_video, output, logbox=self.log, progressbar=self.progress,
                            previewer=self.preview_frame, profile=self.profile.get(),
                            accel=inference, preview_enabled=self.preview_enabled)
                        self.engine = engine
                        result = engine.run()
                    else:
                        engine = SceneEngine(
                            current_video, output, cfg.copy(), logbox=self.log, progressbar=self.progress,
                            previewer=self.preview_frame, preview_enabled=self.preview_enabled)
                        for temp_file in temp_files:
                            engine.add_temp_file(temp_file)
                        self.engine = engine
                        result = engine.run(scene_mode=scene_mode)

                    failed_cuts = int(getattr(engine, "failed", 0))
                    if result and failed_cuts:
                        partial += 1
                        warning_msg = (
                            f"Warning: {basename} finished with {failed_cuts} failed scene(s). "
                            "See cut_errors.json in its output folder."
                        )
                    if result:
                        completed += 1
                    elif not self.batch_stop:
                        failed += 1
                        error_msg = "the video could not be processed"
                        error_type = "ProcessingError"
                    else:
                        stopped = True
                        break
                except Exception as e:
                    failed += 1
                    error_msg = str(e) or "Unknown error"
                    error_type = type(e).__name__
                    print("Batch item error:", e)
                finally:
                    if warning_msg and self.log:
                        self.after(0, lambda m=warning_msg: self.log.append_message(m, kind="warning"))
                    if error_msg and self.log:
                        self.after(0, lambda b=basename, t=error_type, e=error_msg: self.log.append_message(
                            f"Error processing {b} [{t}]: {e}", kind="error"))
                    self.engine = None
                    engine = None
                    gc.collect()
                    for temp_file in temp_files:
                        remove_temp_file(temp_file)
                    if not self.batch_stop and not self.stop_pending and index < len(videos):
                        if self.progress:
                            self.after(0, self.progress.reset)
                        self._cooldown_between_videos()

            elapsed = max(1, int(time.time() - started))
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            total_time = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
            summary = (
                f"Queue finished: {completed}/{len(videos)} video(s) processed, "
                f"{failed} failed"
            )
            if partial:
                summary += f", {partial} with failed scene(s)"
            summary += "."
            final_warning = summary if (failed or partial) else None

            self.after(0, lambda: self.reset_ui(
                finished=completed > 0 and not stopped,
                total_time=total_time,
                stopped=stopped,
                warning_message=final_warning if not stopped else None))
            if completed <= 0 and not stopped and self.log:
                self.after(0, lambda: self.log.append_message(summary, kind="error"))
        finally:
            self.engine = None
            self.batch_stop = False
            self.stop_pending = False

    def run_engine(self, scene_mode):
        result = False
        error_msg = None
        error_type = None
        warning_msg = None
        temp_files = []
        try:
            self.engine.video = prepare_video_for_processing(self.engine.video, temp_files=temp_files)
            result = self.engine.run(scene_mode=scene_mode)
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            print("Error:", e)
        finally:
            total_time = None
            engine = self.engine
            stopped = engine._stop if engine else False
            if result and engine:
                total_time = engine.total_time()
                failed = len(getattr(engine, "_cut_failures", []))
                if failed:
                    warning_msg = (
                        f"Warning: {failed} scene(s) could not be exported. "
                        "See cut_errors.json in the output folder."
                    )
            self.engine = None
            self.stop_pending = False
            if error_msg:
                self.after(0, lambda e=error_msg[:160], t=error_type: self.log.append_message(
                    f"Error [{t}]: {e}", kind="error"))
            elif not result and not stopped:
                self.after(0, lambda: self.log.append_message(
                    "Error: the video could not be processed.", kind="error"))
            self.after(0, lambda: self.reset_ui(
                finished=result, total_time=total_time, stopped=stopped,
                warning_message=warning_msg))
            gc.collect()
            for temp_file in temp_files:
                remove_temp_file(temp_file)

    def run_face_engine(self):
        result = False
        error_msg = None
        error_type = None
        temp_files = []
        try:
            self.engine.video = prepare_video_for_processing(self.engine.video, temp_files=temp_files)
            result = self.engine.run()
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            print("Face engine error:", e)
        finally:
            total_time = None
            engine = self.engine
            stopped = engine._stop if engine else False
            if result and engine:
                total_time = engine.total_time()
            self.engine = None
            self.stop_pending = False
            if error_msg:
                self.after(0, lambda e=error_msg, t=error_type: self.log.append_message(
                    f"Error [{t}]: {e}", kind="error"))
            self.after(0, lambda: self.reset_ui(
                finished=result, total_time=total_time, stopped=stopped))
            gc.collect()
            for temp_file in temp_files:
                remove_temp_file(temp_file)

    def reset_ui(self, finished=False, total_time=None, stopped=False, warning_message=None):
        self.stop_pending = False
        self.running = False
        self.start_btn.configure(text="Start", fg_color=ACCENT,
                                 hover_color="#4f46e5", state="normal")
        self.set_ui_state(False)
        if stopped:
            self.cleanup_process(reason="stop")
        elif finished:
            self.cleanup_process(reason="finish", total_time=total_time,
                                 warning_message=warning_message)

    def set_ui_state(self, disabled):
        state = "disabled" if disabled else "normal"
        for widget in [self.video_selector.button, self.output_selector.button,
                       *self.mode_radios, *self.accel_radios]:
            widget.configure(state=state)
        self.video_selector.entry.configure(state=state)
        self.output_selector.entry.configure(state=state)

        if disabled:
            for rb in self.profile_radios:
                rb.configure(state="disabled")
        else:
            for rb in self.profile_radios:
                if self.cut_mode.get() == "faces" and rb.cget("value") == "Auto":
                    rb.configure(state="disabled")
                else:
                    rb.configure(state="normal")

        if self.cut_mode.get() == "interval":
            self.interval_entry.configure(state=state)
        self.preview_switch.configure(state="disabled" if self.running else "normal")

    def _on_cut_mode_change(self, *args):
        mode = self.cut_mode.get()

        if mode == "interval":
            self.interval_entry.pack(anchor="w", padx=(186, 0), pady=(0, 8))
        else:
            self.interval_entry.pack_forget()

        if mode == "interval":
            for rb in self.profile_radios:
                rb.configure(state="disabled")
        elif mode == "faces":
            if self.profile.get() == "Auto":
                self.profile.set("Normal")
            state = "disabled" if self.running else "normal"
            for rb in self.profile_radios:
                if rb.cget("value") == "Auto":
                    rb.configure(state="disabled")
                else:
                    rb.configure(state=state)
        else:
            state = "disabled" if self.running else "normal"
            for rb in self.profile_radios:
                rb.configure(state=state)

        self.update_accel_radios()

    def update_accel_radios(self):
        mode = self.cut_mode.get()
        compat = MODE_ACCEL_COMPAT.get(mode, {})
        allowed = compat.get("encoder", set()) | compat.get("inference", set())
        enabled = (allowed & self.available_accel) | {"cpu"}
        for rb in self.accel_radios:
            value = rb.cget("value")
            rb.configure(state="normal" if value in enabled else "disabled")
        if self.accel.get() not in enabled:
            self.accel.set("cpu")

    def cleanup_process(self, reason="reset", total_time=None, warning_message=None):
        if self.preview_frame and reason in ("stop", "finish"):
            self.preview_frame.clear_all()
        if self.progress and reason in ("stop", "reset"):
            # Reset progress bar
            self.after(0, self.progress.reset)
        elif self.progress and reason == "finish":
            # Ensure progress bar shows 100% before cleanup
            self.after(0, self.progress.mark_finished)
        if self.log:
            if reason == "stop":
                self.log.clear_status()
                self.log.status_lines[0] = "Process stopped"
                self.log.write_status()
            elif reason == "finish":
                msg = "Process finished"
                if total_time:
                    msg += f" {total_time}"
                self.log.write_finished(msg)
                if warning_message:
                    self.log.append_message(warning_message, kind="warning")
        self.after(1000, gc.collect)

    def confirm_stop(self):
        if not self.running or self.stop_pending:
            return
        answer = mb.askyesno("Confirm Stop",
                             "The process is still running.\nDo you really want to stop it?")
        if not answer:
            return
        self.stop_pending = True
        self.after(0, self.stop_process)

    def _validate_interval(self, value: str) -> bool:
        if value == "":
            return True
        if not value.isdigit():
            return False
        return 1 <= int(value) <= 18000

    def _finalize_start_ui(self):
        self.start_btn.configure(text="Stop ", fg_color=DANGER, hover_color="#dc2626")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    install_crash_logging()

    if not single_instance():
        mb.showerror("Application Already Running",
                     "This application is already running.")
        sys.exit(0)

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    app = SceneCutterApp()
    app.mainloop()
