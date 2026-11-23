import sys
import os
import ctypes
import ctypes.util
import socket
import subprocess
import threading
import time
import datetime
import json
import gc
import atexit
import bisect
import textwrap
import traceback
import faulthandler
import random
import importlib.util
import io
import shutil
import site
import platform
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageOps
import numpy as np
import cv2

torch = None
TORCH_AVAILABLE = False
LAST_IMPORT_ERRORS = {}


def _resource_root():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _user_data_dir():
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "Scenespy")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~/Library/Application Support"), "Scenespy")
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "scenespy")


def _ai_pack_dir():
    if os.environ.get("SCENESPY_AI_PACK"):
        return os.environ["SCENESPY_AI_PACK"]
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "Scenespy", "ai-pack")
    return os.path.join(_user_data_dir(), "ai-pack")


def _runtime_dir():
    if os.environ.get("SCENESPY_RUNTIME"):
        return os.environ["SCENESPY_RUNTIME"]
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "Scenespy", "runtime")
    return os.path.join(_user_data_dir(), "runtime")


APP_DIR = _resource_root()
USER_DATA_DIR = _user_data_dir()
try:
    os.makedirs(USER_DATA_DIR, exist_ok=True)
except Exception:
    USER_DATA_DIR = APP_DIR
SETTINGS_FILE = os.path.join(USER_DATA_DIR, "settings.json")
CRASH_LOG_FILE = os.path.join(USER_DATA_DIR, "scenespy_crash.log")
BIN_DIR = os.path.join(APP_DIR, "bin")
RUNTIME_DIR = _runtime_dir()
MODEL_FILE = os.path.join(APP_DIR, "models", "yolov8n-face.pt")
AI_PACK_DIR = _ai_pack_dir()
ASSETS_DIR = os.path.join(APP_DIR, "scenespy", "assets")
FONT_DIR = os.path.join(ASSETS_DIR, "fonts")
IMAGE_DIR = os.path.join(ASSETS_DIR, "images")
LOADING_GIF_FILE = os.path.join(IMAGE_DIR, "loading.gif")
UI_FONT_FAMILY = "JetBrains Mono"
UI_FONT_SCALE = 0.92
EXECUTABLE_PATHS = {}
_CRASH_LOG_HANDLE = None
_REGISTERED_FONT_PATHS = []
_ACTIVE_UI_FONT_FAMILY = UI_FONT_FAMILY
_CHILD_PROCS = set()
_CHILD_PROCS_LOCK = threading.Lock()
_AI_PACK_PATHS_ADDED = False
_AI_PACK_DLL_HANDLES = []


def _ai_pack_site_candidates():
    if sys.platform == "win32":
        yield os.path.join(AI_PACK_DIR, "Lib", "site-packages")
    else:
        lib_dir = os.path.join(AI_PACK_DIR, "lib")
        if os.path.isdir(lib_dir):
            for name in sorted(os.listdir(lib_dir)):
                if name.startswith("python"):
                    yield os.path.join(lib_dir, name, "site-packages")


def _ai_pack_dll_candidates():
    if sys.platform != "win32":
        return
    site_packages = os.path.join(AI_PACK_DIR, "Lib", "site-packages")
    yield AI_PACK_DIR
    yield os.path.join(AI_PACK_DIR, "Scripts")
    yield os.path.join(site_packages, "torch", "lib")
    nvidia_dir = os.path.join(site_packages, "nvidia")
    if os.path.isdir(nvidia_dir):
        for root, dirs, _files in os.walk(nvidia_dir):
            if os.path.basename(root).lower() in {"bin", "lib"}:
                yield root
            dirs[:] = [name for name in dirs if name not in {"__pycache__", "include"}]


def _add_ai_pack_dll_dirs():
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    for path in _ai_pack_dll_candidates() or ():
        if not path or not os.path.isdir(path):
            continue
        try:
            _AI_PACK_DLL_HANDLES.append(os.add_dll_directory(path))
        except Exception:
            pass


def add_ai_pack_to_path():
    global _AI_PACK_PATHS_ADDED
    if _AI_PACK_PATHS_ADDED:
        return
    for path in _ai_pack_site_candidates():
        if os.path.isdir(path) and path not in sys.path:
            site.addsitedir(path)
    _add_ai_pack_dll_dirs()
    _AI_PACK_PATHS_ADDED = True


add_ai_pack_to_path()


def _fallback_ui_font_family():
    if sys.platform == "win32":
        return "Consolas"
    if sys.platform == "darwin":
        return "Menlo"
    return "DejaVu Sans Mono"


def ui_font(size, weight=None):
    family = _ACTIVE_UI_FONT_FAMILY
    scaled_size = max(1, round(size * UI_FONT_SCALE))
    if weight:
        return (family, scaled_size, weight)
    return (family, scaled_size)


def _bundled_font_paths():
    return [
        os.path.join(FONT_DIR, "JetBrainsMono-Regular.ttf"),
        os.path.join(FONT_DIR, "JetBrainsMono-Bold.ttf"),
    ]


def _register_font_windows(path):
    FR_PRIVATE = 0x10
    return bool(ctypes.windll.gdi32.AddFontResourceExW(path, FR_PRIVATE, 0))


def _register_font_macos(path):
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    ct = ctypes.CDLL("/System/Library/Frameworks/CoreText.framework/CoreText")
    cf.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
    cf.CFURLCreateFromFileSystemRepresentation.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_bool
    ]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    ct.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool
    ct.CTFontManagerRegisterFontsForURL.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)
    ]
    encoded = os.fsencode(path)
    url = cf.CFURLCreateFromFileSystemRepresentation(None, encoded, len(encoded), False)
    if not url:
        return False
    try:
        error = ctypes.c_void_p()
        return bool(ctypes.c_bool(ct.CTFontManagerRegisterFontsForURL(url, 1, ctypes.byref(error))).value)
    finally:
        cf.CFRelease(url)


def _register_font_linux(path):
    lib_name = ctypes.util.find_library("fontconfig")
    if not lib_name:
        return False
    fc = ctypes.CDLL(lib_name)
    fc.FcInit()
    fc.FcConfigGetCurrent.restype = ctypes.c_void_p
    fc.FcConfigAppFontAddFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    fc.FcConfigAppFontAddFile.restype = ctypes.c_bool
    fc.FcConfigBuildFonts.argtypes = [ctypes.c_void_p]
    fc.FcConfigBuildFonts.restype = ctypes.c_bool
    config = fc.FcConfigGetCurrent()
    if not config:
        return False
    added = fc.FcConfigAppFontAddFile(config, os.fsencode(path))
    if added:
        fc.FcConfigBuildFonts(config)
    return bool(added)


def register_bundled_fonts():
    global _ACTIVE_UI_FONT_FAMILY
    registered_any = False
    for path in _bundled_font_paths():
        if not os.path.isfile(path) or path in _REGISTERED_FONT_PATHS:
            continue
        try:
            if sys.platform == "win32":
                registered = _register_font_windows(path)
            elif sys.platform == "darwin":
                registered = _register_font_macos(path)
            else:
                registered = _register_font_linux(path)
            if registered:
                _REGISTERED_FONT_PATHS.append(path)
                registered_any = True
        except Exception as e:
            log_crash(f"Font registration failed for {path}: {e}")
    if not registered_any and not _REGISTERED_FONT_PATHS:
        _ACTIVE_UI_FONT_FAMILY = _fallback_ui_font_family()


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
    """Load persisted UI paths."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"last_video": "", "last_output": ""}


def save_settings(video="", output=""):
    """Persist the last selected input and output paths."""
    try:
        settings = {"last_video": video, "last_output": output}
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


def total_physical_memory():
    """Best-effort physical RAM detection using only platform APIs."""
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical)
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * pages
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def adaptive_cache_profile(cpu_count=None, memory_bytes=None):
    """Return bounded cache/work limits appropriate for the host machine."""
    cpus = max(1, int(cpu_count or os.cpu_count() or 4))
    memory = int(total_physical_memory() if memory_bytes is None else memory_bytes)
    gib = memory / (1024 ** 3) if memory > 0 else 0

    if cpus <= 4 or (gib and gib < 8):
        tier = "compact"
        limits = (8, 384, 192, 8, 1)
    elif cpus >= 12 and gib >= 24:
        tier = "expanded"
        limits = (32, 2048, 1024, 48, 3)
    else:
        tier = "balanced"
        limits = (16, 1024, 512, 20, 2)
    frame_cache, clip_cache, face_cache, thumbnail_batch, cut_workers = limits
    return {
        "tier": tier,
        "cpu_count": cpus,
        "memory_bytes": memory,
        "frame_cache": frame_cache,
        "clip_thumbnail_cache": clip_cache,
        "face_thumbnail_cache": face_cache,
        "thumbnail_batch": thumbnail_batch,
        "cut_workers": cut_workers,
    }


CACHE_PROFILE = adaptive_cache_profile()


PROFILES = {
    "Low": {"label": "Low", "base_threshold": 42, "min_dur": 5.0, "dur_max_boost": 5.0},
    "Normal": {"label": "Normal", "base_threshold": 27, "min_dur": 2.0, "dur_max_boost": 4.0},
    "High": {"label": "High", "base_threshold": 18, "min_dur": 0.7, "dur_max_boost": 2.5},
    "Auto": {"label": "Auto", "ADAPTIVE": True},
}

# User-tunable defaults and runtime limits.
ACCEL_OPTIONS = ["cpu", "nvidia", "amd", "intel", "apple"]
MAX_CUT_WORKERS = CACHE_PROFILE["cut_workers"]
ENABLE_PREVIEW_DEFAULT = True
PREVIEW_INTERVAL = 0.2
INSTANCE_SOCKET = None
INSTANCE_PORT = 54321
PREVIEW_MAX_WIDTH = 640
PREVIEW_MAX_HEIGHT = 360
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
PROCESS_COOLDOWN_SECONDS = 1.25
PROCESS_START_DELAY_SECONDS = 0.1
ANALYSIS_MOSAIC_INTERVAL_MIN = 10.0
ANALYSIS_MOSAIC_INTERVAL_MAX = 20.0
ANALYSIS_MOSAIC_MAX_SOURCES = 1
ANALYSIS_MOSAIC_COLUMNS = 3
ANALYSIS_MOSAIC_ROWS = 2
ANALYSIS_MOSAIC_TILE_W = 208
ANALYSIS_MOSAIC_TILE_H = 162
ANALYSIS_MOSAIC_GAP = 4

MODE_ACCEL_COMPAT = {
    "scene": {"encoder": {"cpu", "nvidia", "amd", "intel", "apple"}, "inference": {"cpu"}},
    "interval": {"encoder": {"cpu", "nvidia", "amd", "intel", "apple"}, "inference": {"cpu"}},
    "faces": {"encoder": {"cpu"}, "inference": {"cpu", "nvidia", "apple"}},
}

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

MODE_ABBREV = {"faces": "FD", "scene": "SD", "interval": "ES"}


def status_labels(mode):
    return {
        "scene": ("Scenes detected", "Scenes cut"),
        "interval": ("Segments total", "Segments cut"),
        "faces": ("Faces detected", "Faces saved"),
    }.get(mode, ("Items detected", "Items done"))


def format_status_lines(mode, detected=None, cut=None, eta=None):
    detected_label, cut_label = status_labels(mode)
    eta_label = "Estimated time"
    width = max(len(eta_label), len(detected_label), len(cut_label))
    return [
        f"{detected_label:<{width}} : {detected if detected is not None else '-'}",
        f"{cut_label:<{width}} : {cut if cut is not None else '-'}",
        f"{eta_label:<{width}} : {eta if eta is not None else '--:--'}",
    ]

DEBUG = False

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


def test_ffmpeg_encoder(encoder: str) -> bool:
    options = {
        "libx264": ["-crf", "16", "-preset", "medium", "-tune", "film"],
        "h264_nvenc": [
            "-preset", "p5", "-cq", "19", "-rc", "vbr", "-b:v", "0",
            "-spatial_aq", "1", "-temporal_aq", "1", "-rc-lookahead", "20"],
        "h264_amf": ["-quality", "balanced", "-rc", "vbr_peak"],
        "h264_qsv": ["-preset", "medium", "-global_quality", "20"],
        "h264_videotoolbox": ["-q:v", "65"],
    }.get(encoder, [])
    output = None
    try:
        handle, output = tempfile.mkstemp(suffix=".mp4")
        os.close(handle)
        os.remove(output)
        result = run_hidden(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             "color=c=black:s=320x180:r=30:d=0.2",
             "-frames:v", "3", "-c:v", encoder, *options,
             "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
             output],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=10
        )
        return (result.returncode == 0 and os.path.isfile(output) and
                os.path.getsize(output) > 0)
    except Exception:
        return False
    finally:
        if output:
            try:
                os.remove(output)
            except OSError:
                pass


def detect_available_accel():
    return detect_available_encoder_accel() | detect_available_inference_accel()


def detect_available_encoder_accel():
    available = {"cpu"}
    if test_ffmpeg_encoder("h264_nvenc"):
        available.add("nvidia")
    if test_ffmpeg_encoder("h264_amf"):
        available.add("amd")
    if test_ffmpeg_encoder("h264_qsv"):
        available.add("intel")
    if test_ffmpeg_encoder("h264_videotoolbox"):
        available.add("apple")
    return available


def detect_available_inference_accel():
    available = {"cpu"}
    if importlib.util.find_spec("torch"):
        try:
            # Importing torch is the only reliable way to distinguish a CUDA
            # build with a usable driver/device from a CPU-only installation.
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                available.add("nvidia")
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and mps.is_available():
                available.add("apple")
        except Exception:
            # Capability probing must never prevent the application starting.
            pass
    return available


def _missing_modules(*names):
    add_ai_pack_to_path()
    return [name for name in names if importlib.util.find_spec(name) is None]


def _macos_major_version():
    if sys.platform != "darwin":
        return None
    try:
        return int(str(platform.mac_ver()[0]).split(".", 1)[0])
    except Exception:
        return None


def mediapipe_required():
    major = _macos_major_version()
    return sys.platform != "darwin" or major is None or major >= 13


def face_detection_required_modules():
    modules = ["torch", "ultralytics"]
    if mediapipe_required():
        modules.append("mediapipe")
    return modules


def missing_face_detection_modules():
    """Return AI modules absent from the app's private runtime."""
    return _missing_modules(*face_detection_required_modules())


def refresh_ai_runtime():
    """Make a newly installed private AI runtime visible to this process."""
    global _AI_PACK_PATHS_ADDED, torch, TORCH_AVAILABLE
    _AI_PACK_PATHS_ADDED = False
    torch = None
    TORCH_AVAILABLE = False
    LAST_IMPORT_ERRORS.clear()
    importlib.invalidate_caches()
    add_ai_pack_to_path()


def _missing_executables(*names):
    return [name for name in names if resolve_executable(name) is None]


def _platform_bin_dir():
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _executable_names(name):
    if sys.platform == "win32" and not name.lower().endswith(".exe"):
        return [f"{name}.exe", name]
    return [name]


def _bundled_executable_candidates(name):
    for filename in _executable_names(name):
        yield os.path.join(RUNTIME_DIR, "bin", _platform_bin_dir(), filename)
        yield os.path.join(RUNTIME_DIR, "bin", filename)
        yield os.path.join(BIN_DIR, _platform_bin_dir(), filename)
        yield os.path.join(BIN_DIR, filename)


def resolve_executable(name):
    """Return bundled executable path first, then system PATH."""
    key = name.lower()
    if key in EXECUTABLE_PATHS:
        return EXECUTABLE_PATHS[key]

    for candidate in _bundled_executable_candidates(name):
        if os.path.isfile(candidate):
            EXECUTABLE_PATHS[key] = candidate
            return candidate

    found = shutil.which(name)
    if found:
        EXECUTABLE_PATHS[key] = found
        return found

    return None


def ffmpeg_path():
    return resolve_executable("ffmpeg")


def ffprobe_path():
    return resolve_executable("ffprobe")


def normalize_command(cmd):
    if not cmd:
        return cmd
    normalized = list(cmd)
    exe_name = os.path.basename(str(normalized[0])).lower()
    exe_root, _ext = os.path.splitext(exe_name)
    if exe_root in {"ffmpeg", "ffprobe"}:
        resolved = resolve_executable(exe_root)
        if resolved:
            normalized[0] = resolved
    return normalized


def validate_runtime_dependencies():
    missing = _missing_executables("ffmpeg", "ffprobe")
    if not missing:
        return True, ""

    runtime_dir = os.path.join(RUNTIME_DIR, "bin", _platform_bin_dir())
    bundled_dir = os.path.join(BIN_DIR, _platform_bin_dir())
    message = (
        "Scenespy requires FFmpeg and FFprobe to process videos.\n\n"
        f"Missing: {', '.join(missing)}\n\n"
        "The automatic runtime setup did not complete. Reinstall Scenespy "
        "with an internet connection, or install FFmpeg in PATH.\n\n"
        "Expected runtime path:\n"
        f"{runtime_dir}\n\n"
        f"Bundled fallback path:\n{bundled_dir}"
    )
    return False, message


def next_output_path(base_output, mode, profile, accel, now=None):
    """Return a readable second-based output path without reusing a folder."""
    moment = now or datetime.datetime.now()
    ts = moment.strftime("%Y%m%d_%H%M%S")
    mode_tag = MODE_ABBREV.get(mode, mode.upper())
    name = f"{mode_tag}_{ts}_{profile}_{accel}"
    path = os.path.join(base_output, name)
    suffix = 2
    while os.path.exists(path):
        path = os.path.join(base_output, f"{name}_{suffix}")
        suffix += 1
    return path


def build_output_dir(base_output, mode, profile, accel):
    path = next_output_path(base_output, mode, profile, accel)
    os.makedirs(path, exist_ok=False)
    return path


def is_valid_video_file(path: str, stop_cb=None) -> bool:
    try:
        out = check_output_hidden_cancelable(
            ["ffprobe", "-v", "error", "-show_streams",
             "-select_streams", "v", "-of", "json", path],
            stop_cb=stop_cb, stderr=subprocess.DEVNULL
        )
        if stop_cb and stop_cb():
            return False
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


def remux_if_needed(path, temp_files=None, stop_cb=None):
    ext = os.path.splitext(path)[1].lower()
    if ext != ".mkv":
        return path

    fixed = path[:-4] + "_fixed.mkv"
    remove_temp_file(fixed)

    result = run_hidden_cancelable(
        ["ffmpeg", "-y", "-fflags", "+genpts+igndts", "-err_detect", "ignore_err",
         "-i", path, "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
         "-max_interleave_delta", "0", "-avoid_negative_ts", "make_zero", fixed],
        stop_cb=stop_cb, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if result.returncode == -9 or (stop_cb and stop_cb()):
        remove_temp_file(fixed)
        return path
    if os.path.exists(fixed) and is_valid_video_file(fixed):
        if temp_files is not None:
            temp_files.append(fixed)
        return fixed
    remove_temp_file(fixed)
    return path


def prepare_video_for_processing(path, temp_files=None, stop_cb=None):
    prepared = remux_if_needed(path, temp_files=temp_files, stop_cb=stop_cb)
    return prepared


def resize_for_preview(img, max_w=PREVIEW_MAX_WIDTH, max_h=PREVIEW_MAX_HEIGHT):
    w, h = img.size
    if w <= 0 or h <= 0:
        return None
    scale = min(max_w / w, max_h / h)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


def preview_frame_with_ffmpeg(path, seconds, stop_cb=None):
    """Decode one small preview frame when OpenCV cannot handle the source codec."""
    try:
        data = check_output_hidden_cancelable(
            ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, float(seconds)):.3f}",
             "-i", path, "-frames:v", "1", "-vf",
             f"scale={PREVIEW_MAX_WIDTH}:{PREVIEW_MAX_HEIGHT}:force_original_aspect_ratio=decrease",
             "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
            stop_cb=stop_cb,
        )
        if not data:
            return None
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB").copy()
    except Exception:
        return None


class AnalysisMosaicPreview:
    """Shows random video thumbnails while scene analysis is running."""
    def __init__(self, videos, previewer, stop_cb=None):
        self.videos = [v for v in videos if v and os.path.isfile(v)]
        self.previewer = previewer
        self.stop_cb = stop_cb or (lambda: False)
        self._stop = False
        self._thread = None
        self._caps = []
        self._durations = {}
        self._tiles = []
        self._lock = threading.Lock()

    def start(self):
        if not self.previewer or not self.videos:
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=False)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        self._thread = None
        self._release_caps()

    def is_running(self):
        return bool(self._thread and self._thread.is_alive())

    def _release_caps(self):
        with self._lock:
            for cap, _path in self._caps:
                try:
                    cap.release()
                except Exception:
                    pass
            self._caps = []

    def _open_sources(self):
        sources = list(self.videos)
        random.shuffle(sources)
        for path in sources[:ANALYSIS_MOSAIC_MAX_SOURCES]:
            if self._stop or self.stop_cb():
                break
            try:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    cap.release()
                    continue
                duration = self._probe_duration(path, cap)
                if duration <= 0:
                    cap.release()
                    continue
                self._caps.append((cap, path))
                self._durations[path] = duration
            except Exception:
                pass

    def _probe_duration(self, path, cap):
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps > 0 and frames > 0:
            return frames / fps
        try:
            return float(check_output_hidden_cancelable(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                stop_cb=self.stop_cb
            ).decode().strip())
        except Exception:
            return 0.0

    def _read_random_tile(self):
        with self._lock:
            if not self._caps:
                return None
            cap, path = random.choice(self._caps)
            duration = self._durations.get(path, 0.0)
            if duration <= 0:
                return None
            t = random.uniform(0.05, max(0.06, duration * 0.95))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
        if not ret or frame is None:
            return None
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return ImageOps.fit(
            image,
            (ANALYSIS_MOSAIC_TILE_W, ANALYSIS_MOSAIC_TILE_H),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5))

    def _compose(self):
        tile_count = ANALYSIS_MOSAIC_COLUMNS * ANALYSIS_MOSAIC_ROWS
        if len(self._tiles) < tile_count:
            return None
        width = (ANALYSIS_MOSAIC_COLUMNS * ANALYSIS_MOSAIC_TILE_W +
                 (ANALYSIS_MOSAIC_COLUMNS + 1) * ANALYSIS_MOSAIC_GAP)
        height = (ANALYSIS_MOSAIC_ROWS * ANALYSIS_MOSAIC_TILE_H +
                  (ANALYSIS_MOSAIC_ROWS + 1) * ANALYSIS_MOSAIC_GAP)
        canvas = Image.new("RGB", (width, height), BG_MAIN)
        for idx, img in enumerate(self._tiles[:tile_count]):
            x = ANALYSIS_MOSAIC_GAP + (idx % ANALYSIS_MOSAIC_COLUMNS) * (
                ANALYSIS_MOSAIC_TILE_W + ANALYSIS_MOSAIC_GAP)
            y = ANALYSIS_MOSAIC_GAP + (idx // ANALYSIS_MOSAIC_COLUMNS) * (
                ANALYSIS_MOSAIC_TILE_H + ANALYSIS_MOSAIC_GAP)
            canvas.paste(img, (x, y))
        return canvas

    def _wait_for_next_update(self):
        deadline = time.time() + random.uniform(
            ANALYSIS_MOSAIC_INTERVAL_MIN, ANALYSIS_MOSAIC_INTERVAL_MAX)
        while not self._stop and not self.stop_cb() and time.time() < deadline:
            time.sleep(min(0.25, max(0.0, deadline - time.time())))

    def _run(self):
        self._open_sources()
        if not self._caps:
            return
        self._tiles = []
        tile_count = ANALYSIS_MOSAIC_COLUMNS * ANALYSIS_MOSAIC_ROWS
        for _ in range(tile_count):
            tile = self._read_random_tile()
            if tile is not None:
                self._tiles.append(tile)
        if len(self._tiles) < tile_count:
            self._release_caps()
            return

        while not self._stop and not self.stop_cb():
            if not getattr(self.previewer, "enabled", True):
                time.sleep(0.1)
                continue
            image = self._compose()
            if image and self.previewer:
                self.previewer.after(0, lambda img=image: self.previewer.update_image(img))
            self._wait_for_next_update()
            if self._stop or self.stop_cb():
                break
            for idx in random.sample(range(len(self._tiles)), k=min(2, len(self._tiles))):
                tile = self._read_random_tile()
                if tile is not None:
                    self._tiles[idx] = tile
        self._release_caps()


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
    add_ai_pack_to_path()
    if torch is not None:
        return TORCH_AVAILABLE
    try:
        import torch as _t
        torch = _t
        TORCH_AVAILABLE = True
        LAST_IMPORT_ERRORS.pop("torch", None)
    except Exception:
        LAST_IMPORT_ERRORS["torch"] = traceback.format_exc()
        try:
            log_crash("PyTorch import failed:\n" + LAST_IMPORT_ERRORS["torch"])
        except Exception:
            pass
        torch = None
        TORCH_AVAILABLE = False
    return TORCH_AVAILABLE


def _ensure_yolo():
    """Lazy-load ultralytics.YOLO on first call."""
    add_ai_pack_to_path()
    try:
        from ultralytics import YOLO
        LAST_IMPORT_ERRORS.pop("ultralytics", None)
        return YOLO
    except Exception:
        LAST_IMPORT_ERRORS["ultralytics"] = traceback.format_exc()
        try:
            log_crash("ultralytics import failed:\n" + LAST_IMPORT_ERRORS["ultralytics"])
        except Exception:
            pass
        return None


def _ensure_mediapipe():
    """Lazy-load mediapipe on first call."""
    add_ai_pack_to_path()
    try:
        import mediapipe as mp
        LAST_IMPORT_ERRORS.pop("mediapipe", None)
        return mp
    except Exception:
        LAST_IMPORT_ERRORS["mediapipe"] = traceback.format_exc()
        try:
            log_crash("mediapipe import failed:\n" + LAST_IMPORT_ERRORS["mediapipe"])
        except Exception:
            pass
        return None


def runtime_import_error_message(package):
    detail = LAST_IMPORT_ERRORS.get(package)
    if not detail:
        return None
    tail = [line.strip() for line in detail.strip().splitlines() if line.strip()]
    cause = tail[-1] if tail else "unknown import error"
    return (
        f"{package} was found but could not be imported.\n"
        f"AI pack folder: {AI_PACK_DIR}\n"
        f"Cause: {cause}\n\n"
        f"Full details were written to:\n{CRASH_LOG_FILE}"
    )


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


def external_process_env(base=None):
    """Return an environment which is safe for non-bundled executables.

    PyInstaller prepends its private library directory to LD_LIBRARY_PATH on
    Linux.  Passing that path to system programs such as FFmpeg can make them
    load the bundled GLib instead of the distribution's matching version.
    """
    env = dict(os.environ if base is None else base)
    if sys.platform.startswith("linux") and getattr(sys, "frozen", False):
        original = env.get("LD_LIBRARY_PATH_ORIG")
        if original:
            env["LD_LIBRARY_PATH"] = original
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return env


def configure_linux_virtual_machine_graphics():
    """Avoid Qt's fatal GLX path on VirtualBox Linux guests."""
    if not sys.platform.startswith("linux"):
        return
    identity = ""
    for path in ("/sys/class/dmi/id/product_name", "/sys/class/dmi/id/sys_vendor"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                identity += " " + handle.read().lower()
        except OSError:
            continue
    if any(marker in identity for marker in ("virtualbox", "innotek", "oracle vm")):
        os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
        os.environ.setdefault("QT_OPENGL", "software")


def run_hidden(cmd, **kwargs):
    cmd = normalize_command(cmd)
    kwargs.setdefault("env", external_process_env())
    if sys.platform == "win32":
        kwargs.setdefault("creationflags", CREATE_NO_WINDOW)
    capture_output = kwargs.pop("capture_output", False)
    timeout = kwargs.pop("timeout", None)
    check = kwargs.pop("check", False)
    input_data = kwargs.pop("input", None)
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout and stderr arguments may not be used with capture_output.")
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if input_data is not None:
        kwargs.setdefault("stdin", subprocess.PIPE)
    proc = subprocess.Popen(cmd, **kwargs)
    register_child_process(proc)
    try:
        stdout, stderr = proc.communicate(input=input_data, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        terminate_process(proc)
        stdout, stderr = proc.communicate()
        e.output = stdout
        e.stderr = stderr
        raise
    finally:
        unregister_child_process(proc)
    result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    if check and result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=stdout, stderr=stderr)
    return result


def register_child_process(proc):
    if not proc:
        return
    with _CHILD_PROCS_LOCK:
        _CHILD_PROCS.add(proc)


def unregister_child_process(proc):
    if not proc:
        return
    with _CHILD_PROCS_LOCK:
        _CHILD_PROCS.discard(proc)


def terminate_process(proc, grace_seconds=0.5):
    if not proc or proc.poll() is not None:
        return True
    try:
        proc.terminate()
        proc.wait(timeout=grace_seconds)
        return proc.poll() is not None
    except Exception:
        pass
    try:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=grace_seconds)
    except Exception:
        pass
    return proc.poll() is not None


def terminate_all_child_processes(grace_seconds=0.5):
    with _CHILD_PROCS_LOCK:
        procs = list(_CHILD_PROCS)
    stopped = {
        proc for proc in procs
        if terminate_process(proc, grace_seconds=grace_seconds)
    }
    with _CHILD_PROCS_LOCK:
        # Keep resistant live processes registered so close/atexit can retry.
        _CHILD_PROCS.difference_update(stopped)


def run_hidden_cancelable(cmd, stop_cb=None, poll_interval=0.05,
                          low_priority=False, **kwargs):
    cmd = normalize_command(cmd)
    kwargs.setdefault("env", external_process_env())
    if sys.platform == "win32":
        flags = CREATE_NO_WINDOW
        if low_priority:
            flags |= 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS
        kwargs.setdefault("creationflags", flags)
    capture_output = kwargs.pop("capture_output", False)
    timeout = kwargs.pop("timeout", None)
    check = kwargs.pop("check", False)
    input_data = kwargs.pop("input", None)
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout and stderr arguments may not be used with capture_output.")
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if input_data is not None:
        kwargs.setdefault("stdin", subprocess.PIPE)
    proc = subprocess.Popen(cmd, **kwargs)
    register_child_process(proc)
    try:
        deadline = time.time() + timeout if timeout is not None else None
        communicate_input = input_data
        while True:
            remaining = poll_interval
            if deadline is not None:
                remaining = max(0.001, min(remaining, deadline - time.time()))
            try:
                stdout, stderr = proc.communicate(input=communicate_input, timeout=remaining)
                break
            except subprocess.TimeoutExpired:
                communicate_input = None
                if stop_cb and stop_cb():
                    terminate_process(proc)
                    stdout, stderr = proc.communicate()
                    return subprocess.CompletedProcess(cmd, -9, stdout, stderr)
                if deadline is not None and time.time() >= deadline:
                    terminate_process(proc)
                    stdout, stderr = proc.communicate()
                    raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
        result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        if check and result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=stdout, stderr=stderr)
        return result
    except Exception:
        terminate_process(proc)
        raise
    finally:
        unregister_child_process(proc)


def check_output_hidden(cmd, **kwargs):
    cmd = normalize_command(cmd)
    if sys.platform == "win32":
        kwargs.setdefault("creationflags", CREATE_NO_WINDOW)
    kwargs.setdefault("stdout", subprocess.PIPE)
    kwargs.setdefault("check", True)
    result = run_hidden(cmd, **kwargs)
    return result.stdout


def check_output_hidden_cancelable(cmd, stop_cb=None, **kwargs):
    kwargs.setdefault("stdout", subprocess.PIPE)
    kwargs.setdefault("check", True)
    result = run_hidden_cancelable(cmd, stop_cb=stop_cb, **kwargs)
    if result.returncode == -9:
        raise RuntimeError("Stopped")
    return result.stdout


atexit.register(terminate_all_child_processes)


__all__ = [name for name in globals() if not name.startswith("__")]
