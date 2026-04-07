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
import tkinter.filedialog as fd
import tkinter.messagebox as mb
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Imports: third-party (lightweight / always needed)
# ============================================================
import customtkinter as ctk
from PIL import Image

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
# Config: profiles
# ============================================================
PROFILES = {
    "Low":    {"label": "Low",    "THRESHOLD": 48.0, "MIN_FINAL_DURATION": 7.0},
    "Normal": {"label": "Normal", "THRESHOLD": 32.0, "MIN_FINAL_DURATION": 4.0},
    "High":   {"label": "High",   "THRESHOLD": 20.0, "MIN_FINAL_DURATION": 1.0},
    "Auto":   {"label": "Auto",   "MIN_FINAL_DURATION": 4.0, "ADAPTIVE": True},
}

ACCEL_OPTIONS = ["cpu", "nvidia", "amd", "intel"]
MAX_CUT_WORKERS = 2
ENABLE_PREVIEW_DEFAULT = True
PREVIEW_INTERVAL = 0.1
PREVIEW_FPS = 2
INSTANCE_SOCKET = None
INSTANCE_PORT = 54321
PREVIEW_MAX_WIDTH = 420
PREVIEW_MAX_HEIGHT = 240

# ============================================================
# Config: compatibility / constants
# ============================================================
MODE_ACCEL_COMPAT = {
    "scene":    {"encoder": {"cpu", "nvidia", "amd", "intel"}, "inference": {"cpu"}},
    "interval": {"encoder": {"cpu", "nvidia", "amd", "intel"}, "inference": {"cpu"}},
    "faces":    {"encoder": {"cpu"}, "inference": {"cpu", "nvidia"}},
}

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

MODE_ABBREV = {"faces": "FD", "scene": "SD", "interval": "ES"}

# ============================================================
# Config: UI theme palette
# ============================================================
BG_MAIN    = "#1a1a1a"
BG_PANEL   = "#313131"
BG_CARD    = "#404040"
BG_INPUT   = "#1a1a1a"
BORDER_SOFT  = "#787474"
BORDER_SOFT2 = "#4C4848"
TEXT_MAIN  = "#e5e7eb"
TEXT_MUTED = "#9ca3af"
ACCENT     = "#1f538d"
SUCCESS    = "#22c55e"
DANGER     = "#ef4444"


# ============================================================
# Utility: helpers used across the app
# ============================================================
def test_ffmpeg_encoder(encoder: str) -> bool:
    try:
        result = subprocess.run(
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
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = MODE_ABBREV.get(mode, mode.upper())
    path = os.path.join(base_output, f"{mode_tag}_{ts}_{profile}_{accel}")
    os.makedirs(path, exist_ok=True)
    return path


def is_valid_video_file(path: str) -> bool:
    try:
        out = subprocess.check_output(
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


def remux_if_needed(path):
    ext = os.path.splitext(path)[1].lower()
    if ext != ".mkv":
        return path

    fixed = path[:-4] + "_fixed.mkv"
    if os.path.exists(fixed) and is_valid_video_file(fixed):
        return fixed

    subprocess.run(
        ["ffmpeg", "-y", "-fflags", "+genpts+igndts", "-err_detect", "ignore_err",
         "-i", path, "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
         "-max_interleave_delta", "0", "-avoid_negative_ts", "make_zero", fixed],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if os.path.exists(fixed) and is_valid_video_file(fixed):
        return fixed
    return path


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
        return open_video, SceneManager, ContentDetector, StatsManager
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
        self.configure(state="disabled", font=("Consolas", 12))
        self.pack_propagate(False)
        self.status_lines = []
        self.initialized = False

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

    def write_finished(self, text):
        self.configure(state="normal")
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
        self._speed = 0.008

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
        path = fd.askopenfilename()
        if path:
            self.entry.delete(0, "end")
            self.entry.insert(0, path)

    def get(self):
        return self.entry.get()


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

    diff_y_list, diff_c_list, diff_hy_list, diff_hs_list, clip_motion = [], [], [], [], []

    for _i in range(num_clips):
        pos = min(_i * frame_interval, max(0, int(total) - frames_per_clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        prev_y, prev_cbcr, prev_hist_y, prev_hist_hs = None, None, None, None
        clip_y_diffs = []

        for _j in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.resize(cv2.cvtColor(frame, BGR2GRAY), dim)
            ycrcb = cv2.resize(cv2.cvtColor(frame, BGR2YCrCb), dim)
            hsv = cv2.resize(cv2.cvtColor(frame, BGR2HSV), dim)
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
                diff_y_list.append(d_y)
                diff_c_list.append(d_c)
                diff_hy_list.append(d_hy)
                diff_hs_list.append(d_hs)
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

    vars_list = [_var(diff_y_list), _var(diff_c_list),
                 _var(diff_hy_list), _var(diff_hs_list)]
    v_total = sum(vars_list)
    if v_total < 1e-15:
        weights = [0.50, 0.25, 0.15, 0.10]
    else:
        weights = [v / v_total for v in vars_list]
        weights = [0.10 + 0.40 * w for w in weights]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

    composite = [weights[0]*a + weights[1]*b + weights[2]*c + weights[3]*d
                 for a, b, c, d in zip(ry, rc, rhy, rhs)]

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

    t_norm = (threshold - 15.0) / 40.0
    min_dur = 1.0 + 10.0 * (1.0 - t_norm ** 1.5)
    min_dur = max(1.0, min(8.0, min_dur))
    min_dur = round(min_dur, 1)

    return threshold, min_dur


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
        self._start_time = None
        self._end_time = None
        self._video_info_shown = False
        self._fps = None
        self._thumb_container = None
        self._total_frames = None
        self._ffmpeg_proc = None
        self._ui_alive = True
        self._duration = None
        self._last_preview_ratio = -1
        self._keyframes_cache = None
        self._preview_cap = None

    def stop(self):
        self._stop = True
        self._ui_alive = False
        try:
            if self._ffmpeg_proc:
                try:
                    self._ffmpeg_proc.terminate()
                    time.sleep(0.3)
                    if self._ffmpeg_proc.poll() is None:
                        self._ffmpeg_proc.kill()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if self._thumb_container:
                self._thumb_container.close()
        except Exception:
            pass
        self._thumb_container = None
        try:
            if self._preview_cap:
                self._preview_cap.release()
        except Exception:
            pass
        self._preview_cap = None

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
        self._analysis_ratio = 0.0
        self._start_time = time.time()
        self._end_time = None
        self.done = 0
        self.detected = 0
        self.last_preview = 0
        self._last_thumb_time = 0

        if self.previewer and self.preview_enabled:
            try:
                img = self._get_preview_frame_at(0)
                if img and self._ui_alive:
                    self.previewer.after(0, lambda img=img: self.previewer.update_image(img))
            except Exception:
                pass

        if self.preview_enabled:
            try:
                self._preview_cap = cv2.VideoCapture(self.video)
            except Exception:
                self._preview_cap = None

        threading.Thread(target=self._preview_loop, daemon=True).start()

        if self.previewer and not self._video_info_shown:
            self.previewer.update_info(self._get_video_info_text())
            self._video_info_shown = True

        scenes = self._detect_scenes_progressive() if scene_mode else self._fixed_interval()
        if not scenes or self._stop:
            return False

        self._cut_scenes(scenes)
        self._end_time = time.time()

        if self._preview_cap:
            try:
                self._preview_cap.release()
            except Exception:
                pass
            self._preview_cap = None

        return True

    def _get_video_info_text(self):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,r_frame_rate:format=bit_rate",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        try:
            out = subprocess.check_output(cmd).decode().splitlines()
            width, height, fps, bitrate = out
            num, den = fps.split("/")
            fps_float = round(int(num) / int(den), 2)
            return f"{width}x{height} | FPS: {fps_float} | Bitrate: {int(bitrate)/1000:.0f} kbps"
        except Exception:
            return "Video info unavailable"

    def _detect_scenes_progressive(self):
        threshold, min_dur = self._map_threshold()
        fps = self._get_video_fps()

        result = _ensure_scenedetect()
        if result is None:
            raise RuntimeError("scenedetect not installed")
        open_video, SceneManager, ContentDetector, StatsManager = result

        backend = "pyav"
        if self.video.lower().endswith(".mkv"):
            backend = "pyav"

        if backend == "opencv":
            video = open_video(self.video, backend="opencv")
        else:
            video = open_video(self.video, backend=backend, suppress_output=True)

        if backend == "opencv":
            try:
                _ = video.frame_rate
            except Exception:
                video.close()
                raise RuntimeError("Failed to read video stream")

        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.auto_downscale = False
        scene_manager.downscale = 2
        scene_manager.add_detector(
            ContentDetector(threshold=threshold,
                           min_scene_len=int(min_dur * fps),
                           luma_only=True)
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
                video_duration = float(subprocess.check_output(cmd).decode().strip())
                self._total_frames = int(video_duration * fps)
            except Exception:
                if self._total_frames is None:
                    self._total_frames = int(video.frame_rate * 1)
                video_duration = max((self._total_frames or 1) / fps, 1.0)

        def _progress_cb(frame_num, _):
            if self._stop:
                return False
            if not self.progress:
                return True
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

            if not video_duration:
                return True

            ratio = min(current_time / video_duration, 1.0)
            ratio = max(self._analysis_ratio, ratio)
            self._analysis_ratio = ratio

            if self.progress:
                self.progress.after(0, lambda v=ratio * 0.4: self.progress.update(v))

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
        except RuntimeError:
            return []
        finally:
            try:
                video.close()
            except Exception:
                pass

        if not scene_list:
            total_frames = self._total_frames or int(fps)
            self.detected = 1
            return [(0, total_frames)]

        scenes = [(s.get_frames(), e.get_frames()) for s, e in scene_list]
        self.detected = len(scenes)
        return scenes

    def _fixed_interval(self):
        interval = self.cfg.get("FIXED_INTERVAL", 10)
        cmd = ["ffprobe", "-v", "error",
               "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        duration = float(subprocess.check_output(cmd).decode().strip())
        fps = self._get_video_fps()
        self._total_frames = int(duration * fps)

        scenes, t = [], 0.0
        while t < duration:
            scenes.append((int(t * fps), int(min(t + interval, duration) * fps)))
            t += interval
        self.detected = len(scenes)
        return scenes

    def _cut_scenes(self, scenes):
        outdir = build_output_dir(
            self.output,
            mode="scene" if self.scene_mode else "interval",
            profile=self.cfg.get("label", "NA"),
            accel=self.cfg.get("ENCODER", "cpu")
        )
        fps = self._get_video_fps()
        self.total = len(scenes)
        self.done = 0
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
            tasks.append((
                i, output_file, start_time, end_time,
                aligned_start, aligned_end,
                start_delta > (1 / fps) or end_delta > (1 / fps)
            ))

        def _cut_one(task):
            idx, output_file, start_time, end_time, aligned_start, aligned_end, precise = task
            if precise:
                self._run_ffmpeg_precise_cut(
                    self.video, start_time, end_time, output_file, aligned_start)
            else:
                self._run_ffmpeg_copy(aligned_start, aligned_end, output_file)

        if self.total <= 1:
            _cut_one(tasks[0])
            self.done = 1
        else:
            with ThreadPoolExecutor(max_workers=min(MAX_CUT_WORKERS, self.total)) as pool:
                futures = {pool.submit(_cut_one, task): task for task in tasks}
                for future in as_completed(futures):
                    if self._stop:
                        pool.shutdown(wait=False)
                        return
                    with self._cut_lock:
                        self.done += 1
                        if self.progress:
                            ratio = self.done / self.total
                            self.progress.after(0, lambda v=ratio: self.progress.update(v))
                        if self.log:
                            eta = self._calculate_eta()
                            self.log.after(0, lambda d=self.detected, c=self.done, e=eta:
                                self.log.write_status(d, c, e))

    def _get_progress_ratio(self):
        if self.total > 0:
            return self.done / self.total
        return self._analysis_ratio

    def _calculate_eta(self):
        if self.done == 0:
            return "--:--"
        elapsed = time.time() - self._start_time
        avg_per_scene = elapsed / self.done
        remaining = self.total - self.done
        eta_seconds = int(avg_per_scene * remaining)
        m, s = divmod(eta_seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _get_video_fps(self):
        if self._fps is not None:
            return self._fps
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=avg_frame_rate",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        num, den = subprocess.check_output(cmd).decode().strip().split("/")
        self._fps = float(num) / float(den)
        return self._fps

    def _map_threshold(self):
        if self.cfg.get("ADAPTIVE"):
            return _adaptive_threshold(self.video)
        base = self.cfg["THRESHOLD"]
        if base >= 45:
            return 42.0, self.cfg.get("MIN_FINAL_DURATION", 4.0)
        elif base >= 30:
            return 30.0, self.cfg.get("MIN_FINAL_DURATION", 4.0)
        else:
            return 18.0, self.cfg.get("MIN_FINAL_DURATION", 4.0)

    def _get_video_duration(self):
        if self._duration is not None:
            return self._duration
        cmd = ["ffprobe", "-v", "error",
               "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        self._duration = float(subprocess.check_output(cmd).decode().strip())
        return self._duration

    def _get_keyframes(self):
        if self._keyframes_cache is not None:
            return self._keyframes_cache
        cmd = ["ffprobe", "-select_streams", "v", "-skip_frame", "nokey",
               "-show_entries", "frame=pkt_pts_time",
               "-loglevel", "error", "-of", "csv=p=0", self.video]
        result = subprocess.run(cmd, capture_output=True, text=True)
        keyframes = []
        for line in result.stdout.splitlines():
            try:
                keyframes.append(float(line.strip()))
            except Exception:
                pass
        keyframes = sorted(set(keyframes))
        self._keyframes_cache = keyframes
        return keyframes

    def _run_ffmpeg_copy(self, start, end, output):
        duration = end - start
        cmd = ["ffmpeg", "-y", "-ss", f"{start:.6f}",
               "-i", self.video, "-t", f"{duration:.6f}",
               "-c", "copy", "-avoid_negative_ts", "make_zero",
               "-muxpreload", "0", "-muxdelay", "0", output]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.05)

    def _run_ffmpeg_precise_cut(self, input_file, start_time, end_time, output, aligned_start):
        encoder = self.cfg.get("ENCODER", "cpu")
        duration = end_time - start_time
        bitrate = self._get_video_bitrate()

        if encoder == "cpu":
            codec = ["-c:v", "libx264", "-crf", "18", "-preset", "slow"]
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

        cmd = ["ffmpeg", "-y", "-ss", f"{start_time:.6f}", "-i", input_file,
               "-t", f"{duration:.6f}", *codec,
               "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", output]

        if bitrate and encoder == "cpu":
            cmd += ["-maxrate", str(bitrate), "-bufsize", str(bitrate * 2)]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(0.05)

        if result.returncode != 0 and encoder != "cpu":
            codec_start = cmd.index(codec[0])
            codec_end = cmd.index(codec[-1])
            codec_fallback = ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]
            cmd[codec_start:codec_end + 1] = codec_fallback
            try:
                idx = cmd.index("-maxrate")
                del cmd[idx:idx + 2]
            except ValueError:
                pass
            result2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result2.returncode == 0:
                return
            print("FFMPEG ERROR:\n", result.stderr.decode())

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
            return int(subprocess.check_output(cmd).decode().strip())
        except Exception:
            return None

    def _preview_loop(self):
        while not self._stop:
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
        if not self._preview_cap:
            return None
        try:
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

        if not _ensure_torch():
            raise RuntimeError("Face detection requires PyTorch, but it is not installed.")

        YOLO = _ensure_yolo()
        if YOLO is None:
            raise RuntimeError("ultralytics package not found.")

        mp = _ensure_mediapipe()
        if mp is None:
            raise RuntimeError("mediapipe package not found.")

        self.profile = profile
        self.accel = accel
        self.device = "cuda:0" if accel == "nvidia" and torch.cuda.is_available() else "cpu"
        self._stop = False
        self._start_time = None
        self._end_time = None
        self.detected = 0
        self.done = 0
        self._face_ratio = 0.0

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", "yolov8n-face.pt")
        self.model = YOLO(model_path)

        self.profile_cfg = {
            "Low":    {"conf": 0.45, "min_size": 64, "ttl": 0.6,
                       "min_frames": 0.8, "min_valid_ratio": 0.75, "min_sharpness": 60},
            "Normal": {"conf": 0.35, "min_size": 40, "ttl": 1.2,
                       "min_frames": 0.5, "min_valid_ratio": 0.6, "min_sharpness": 40},
            "High":   {"conf": 0.22, "min_size": 24, "ttl": 2.5,
                       "min_frames": 0.25, "min_valid_ratio": 0.35, "min_sharpness": 20},
        }[profile]

        self.last_preview = 0
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=2, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def stop(self):
        self._stop = True
        self._ui_alive = False

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
        self._start_time = time.time()
        cap = cv2.VideoCapture(self.video)
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
                expand_y_top = int(h * 0.15)
                expand_y_bot = int(h * 0.15)
                cx1 = max(0, x1 - expand_x)
                cy1 = max(0, y1 - expand_y_top)
                cx2 = min(w_frame, x2 + expand_x)
                cy2 = min(h_frame, y2 + expand_y_bot)

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
                        center_face = face_raw[int(fh*0.25):int(fh*0.75),
                                               int(fw*0.25):int(fw*0.75)]
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
                    center_face = face_raw[int(fh*0.25):int(fh*0.75),
                                           int(fw*0.25):int(fw*0.75)]
                    if center_face.size == 0:
                        center_face = face_raw
                    new_tracks.append({
                        "id": track_id,
                        "box": (x1, y1, x2, y2),
                        "ttl": ttl_frames,
                        "frames": 1,
                        "valid": 0,
                        "score": cv2.Laplacian(center_face, cv2.CV_64F).var(),
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

            if self.log:
                self.log.write_status(detected=self.detected, cut=self.done,
                                      eta=self._calculate_eta(frame_idx, total_frames))

            if self.progress:
                ratio = max(self._face_ratio, frame_idx / total_frames)
                self._face_ratio = ratio
                if self._ui_alive:
                    self.progress.after(0, lambda v=ratio: self.progress.update(v))

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

        cap.release()
        self._end_time = time.time()
        return not self._stop

    def _calculate_eta(self, frame_idx, total_frames):
        if frame_idx == 0:
            return "--:--"
        elapsed = time.time() - self._start_time
        avg = elapsed / frame_idx
        remaining = total_frames - frame_idx
        eta = int(avg * remaining)
        m, s = divmod(eta, 60)
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
        self.resizable(False, False)
        self.preview_enabled = ENABLE_PREVIEW_DEFAULT
        self.available_accel = detect_available_accel()
        self._build_ui()

    def _build_ui(self):
        # Left panel
        self.left = ctk.CTkFrame(
            self, width=420, fg_color=BG_PANEL, border_width=1,
            border_color=BORDER_SOFT2, corner_radius=0)
        self.left.pack(side="left", fill="y", padx=10, pady=10)

        files = Section(self.left, "Files")
        files.pack(fill="x", padx=12, pady=12)
        self.video_selector = FileSelector(files, "Source video")
        self.video_selector.pack(fill="x", padx=12)
        self.output_selector = DirectorySelector(files, "Output folder")
        self.output_selector.pack(fill="x", padx=12)

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
        video = self.video_selector.get().strip()
        output = self.output_selector.get().strip()

        if not video and not output:
            return
        if not video or not output:
            self.log.clear_status()
            self.log.status_lines[0] = "Select input video and output folder"
            self.log._render()
            return

        ext = os.path.splitext(video)[1].lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            self.log.clear_status()
            self.log.status_lines[0] = "Unsupported file type"
            self.log._render()
            return

        if not os.path.isfile(video) or not os.path.isdir(output):
            self.log.clear_status()
            self.log.status_lines[0] = "Invalid paths!"
            self.log._render()
            return

        if not is_valid_video_file(video):
            self.log.clear_status()
            self.log.status_lines[0] = "Invalid or unsupported video file"
            self.log._render()
            return

        self.cleanup_process(reason="reset")
        if self.log:
            self.log.clear_status()

        self.running = True
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

        if mode == "faces":
            self.engine = FaceDetectionEngine(
                video, output, logbox=self.log, progressbar=self.progress,
                previewer=self.preview_frame, profile=self.profile.get(),
                accel=inference, preview_enabled=self.preview_enabled)
            threading.Thread(target=self.run_face_engine, daemon=True).start()
            return

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

        self.engine = SceneEngine(
            video, output, cfg, logbox=self.log, progressbar=self.progress,
            previewer=self.preview_frame, preview_enabled=self.preview_enabled)
        threading.Thread(target=self.run_engine, args=(scene_mode,), daemon=True).start()

    def stop_process(self):
        if not self.engine:
            return
        self.engine.stop()

    def run_engine(self, scene_mode):
        result = False
        try:
            self.engine.video = remux_if_needed(self.engine.video)
            result = self.engine.run(scene_mode=scene_mode)
        except Exception as e:
            print("Error:", e)
        finally:
            total_time = None
            engine = self.engine
            stopped = engine._stop if engine else False
            if result and engine:
                total_time = engine.total_time()
            self.engine = None
            self.stop_pending = False
            self.after(0, lambda: self.reset_ui(
                finished=result, total_time=total_time, stopped=stopped))

    def run_face_engine(self):
        result = False
        try:
            self.engine.video = remux_if_needed(self.engine.video)
            result = self.engine.run()
        except Exception as e:
            print("Face engine error:", e)
        finally:
            total_time = None
            engine = self.engine
            stopped = engine._stop if engine else False
            if result and engine:
                total_time = engine.total_time()
            self.engine = None
            self.stop_pending = False
            self.after(0, lambda: self.reset_ui(
                finished=result, total_time=total_time, stopped=stopped))

    def reset_ui(self, finished=False, total_time=None, stopped=False):
        self.stop_pending = False
        self.running = False
        self.start_btn.configure(text="Start", fg_color=ACCENT,
                                  hover_color="#4f46e5", state="normal")
        self.set_ui_state(False)
        if stopped:
            self.cleanup_process(reason="stop")
        elif finished:
            self.cleanup_process(reason="finish", total_time=total_time)

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

    def cleanup_process(self, reason="reset", total_time=None):
        if self.preview_frame and reason in ("stop", "finish"):
            self.preview_frame.clear_all()
        if self.progress and reason in ("stop", "reset"):
            self.after(0, self.progress.reset)
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
        self.after(1000, gc.collect)

    def confirm_stop(self):
        if not self.running or self.engine is None or self.stop_pending:
            return
        answer = mb.askyesno("Confirm stop",
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
    if not single_instance():
        mb.showerror("Aplicativo já em execução",
                      "Este aplicativo já está aberto.")
        sys.exit(0)

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    app = SceneCutterApp()
    app.mainloop()