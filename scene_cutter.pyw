import sys
import socket
import subprocess
import threading
import time
import datetime
import customtkinter as ctk
import av
import cv2
import os
import gc
import mediapipe as mp
import signal
from PIL import Image
from ultralytics import YOLO

from scenedetect import open_video
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.stats_manager import StatsManager

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False



# Config
PROFILES = {
    "Low": {"label": "Low", "THRESHOLD": 48.0, "MIN_FINAL_DURATION": 7.0},
    "Normal": {"label": "Normal", "THRESHOLD": 32.0, "MIN_FINAL_DURATION": 4.0},
    "High": {"label": "High", "THRESHOLD": 20.0, "MIN_FINAL_DURATION": 1.0},
}

ACCEL_OPTIONS = ["cpu", "nvidia", "amd", "intel"]
ENABLE_PREVIEW_DEFAULT = True
PREVIEW_INTERVAL = 0.15
PREVIEW_FPS = 1
INSTANCE_SOCKET = None
INSTANCE_PORT = 54321
PREVIEW_MAX_WIDTH = 420
PREVIEW_MAX_HEIGHT = 240

MODE_ACCEL_COMPAT = {
    "scene": {
        "encoder": {"cpu", "nvidia", "amd", "intel"},
        "inference": {"cpu"}
    },
    "interval": {
        "encoder": {"cpu"},
        "inference": {"cpu"}
    },
    "faces": {
        "encoder": {"cpu"},
        "inference": {"cpu", "nvidia"}
    },
}

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"
}


MODE_ABBREV = {
    "faces": "FD",
    "scene": "SD",
    "interval": "ES",
}


# Paleta base
BG_MAIN = "#1a1a1a"        # fundo geral
BG_PANEL = "#313131"      # painéis
BG_CARD = "#404040"       # seções internas
BG_INPUT = "#1a1a1a"      # entradas

BORDER_SOFT = "#787474"   # bordas finas
BORDER_SOFT2 = "#4C4848"
TEXT_MAIN = "#e5e7eb"
TEXT_MUTED = "#9ca3af"

ACCENT = "#6366f1"        # roxo/índigo moderno
SUCCESS = "#22c55e"
DANGER = "#ef4444"


def detect_available_accel():
    available = {"cpu"}

    # NVIDIA
    try:
        if torch.cuda.is_available():
            available.add("nvidia")
    except Exception:
        pass

    # AMD (AMF)
    if test_ffmpeg_encoder("h264_amf"):
        available.add("amd")

    # Intel (QSV)
    if test_ffmpeg_encoder("h264_qsv"):
        available.add("intel")

    return available

def build_output_dir(base_output, mode, profile, accel):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = MODE_ABBREV.get(mode, mode.upper())
    name = f"{mode_tag}_{ts}_{profile}_{accel}"
    path = os.path.join(base_output, name)
    os.makedirs(path, exist_ok=True)
    return path

def test_ffmpeg_encoder(encoder: str) -> bool:
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "color=c=black:s=160x120:d=0.1",
            "-c:v", encoder,
            "-f", "null",
            "-"
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )

        return result.returncode == 0
    except Exception:
        return False

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


# Widgets
class Section(ctk.CTkFrame):
    def __init__(self, master, title, **kwargs):
        super().__init__(
            master,
            fg_color=BG_CARD,
            border_width=1,
            border_color=BORDER_SOFT2,
            corner_radius=0
        )
        ctk.CTkLabel(
            self, text=title, font=("Consolas", 14, "bold")).pack(anchor="w", padx=12, pady=(8, 4))


class LabeledEntry(ctk.CTkFrame):
    def __init__(self, master, label, placeholder="", width=160):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, font=("Consolas", 12)).pack(anchor="w")
        self.entry = ctk.CTkEntry(
            width=width,
            corner_radius=15,
            fg_color=BG_MAIN,
            border_width=1,
            border_color=BORDER_SOFT,
            text_color=TEXT_MAIN,
            placeholder_text_color=TEXT_MUTED,
            placeholder_text = placeholder
        )
        self.entry.pack(pady=(2, 8), fill="x")

    def get(self):
        return self.entry.get()


class LogBox(ctk.CTkTextbox):
    def __init__(self, master, height=140):
        super().__init__(master, height=height, fg_color=BG_MAIN, corner_radius=15, border_color=BORDER_SOFT2, border_width=1)
        self.configure(state="disabled", font=("Consolas", 12))

        self.status_lines = [
            "",
            "",
            "",
            "",
            ""
        ]

        self._render()

    def write_status(self, detected=None, cut=None, eta=None):
        if detected is not None:
            self.status_lines[0] = f"Scenes detected: {detected}"
        if cut is not None:
            self.status_lines[1] = f"Scenes cut: {cut}"
        if eta is not None:
            self.status_lines[2] = f"Estimated time: {eta}"

        self._render()

    def clear_status(self):
        self.status_lines = [
            "Processing...",
            "",
            "",
            "",
            ""
        ]
        self._render()

    def write_finished(self, text):
        self.configure(state="normal")
        self.delete("1.0", "end")

        for line in self.status_lines:
            self.insert("end", line + "\n")

        self.insert("end", text + "\n", "finished")
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

        self._logical_value = 0.0      # valor real recebido
        self._visual_value = 0.0       # valor exibido
        self._animating = False

        self._speed = 0.008  # menor = mais suave, maior = mais rápido

    def update(self, value):
        if not self._enabled:
            return

        value = max(0.0, min(1.0, value))

        if value < self._logical_value:
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

        # interpolação suave (ease-out simples)
        delta = (self._logical_value - self._visual_value) * self._speed
        delta = max(delta, 0.001)  # evita travar perto do fim

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
        self._enabled = False  # 🔒 BLOQUEIA updates futuros

        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

        self._logical_value = 0.0
        self._visual_value = 0.0
        self._animating = False
        self.bar.configure(progress_color=self._normal_color)
        self.bar.set(0)
        self.label.configure(text="0%")


class PreviewFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(
            master,
            fg_color=BG_MAIN,
            border_width=1,
            border_color=BORDER_SOFT2,
            corner_radius=15
        )
        self.info_label = ctk.CTkLabel(self, text="", font=("Consolas", 10))
        self.info_label.pack(anchor="n", pady=4)
        self.label = ctk.CTkLabel(self, text="")
        self.label.pack(expand=True)
        self._img_ref = None

    def update_image(self, image):
        if not image:
            return
        w, h = image.size
        if w > PREVIEW_MAX_WIDTH or h > PREVIEW_MAX_HEIGHT:
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
    def __init__(self, master, label="File", width=420):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, font=("Consolas", 12)).pack(anchor="w")

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(4, 8))

        self.entry = ctk.CTkEntry(
            row,
            width=width,
            corner_radius=15,
            fg_color=BG_MAIN,
            border_width=1,
            border_color=BORDER_SOFT,
            text_color="#e6e6e6",
            placeholder_text_color=TEXT_MUTED
        )
        self.entry.pack(side="left", fill="x", expand=True)

        self.button = ctk.CTkButton(
            row,
            text="…",
            width=30,
            corner_radius=15,
            fg_color=BG_CARD,
            hover_color=BG_MAIN,
            border_width=1,
            border_color=BORDER_SOFT,
            text_color=TEXT_MUTED,
            command=self.select
        )
        self.button.pack(side="right", padx=(6, 0))

    def select(self):
        import tkinter.filedialog as fd
        path = fd.askopenfilename()
        if path:
            self.entry.delete(0, "end")
            self.entry.insert(0, path)

    def get(self):
        return self.entry.get()


class DirectorySelector(FileSelector):
    def select(self):
        import tkinter.filedialog as fd
        path = fd.askdirectory()
        if path:
            self.entry.delete(0, "end")
            self.entry.insert(0, path)


class RadioGroup(ctk.CTkFrame):
    def __init__(
        self,
        master,
        variable,
        options,
        columns=4,
        radio_width=120,
        height=32
    ):
        super().__init__(master, fg_color="transparent", height=height)

        self.grid_propagate(False)
        self.radios = []

        for i, (label, value) in enumerate(options):
            rb = ctk.CTkRadioButton(
                self,
                text=label,
                variable=variable,
                value=value,

                width=radio_width,
                radiobutton_width=10,
                radiobutton_height=10,

                fg_color=ACCENT,  # círculo interno quando marcado
                border_color="#4b5563",  # cinza suave (borda mais “fina”)
                hover_color="#6366f1",

                text_color=TEXT_MAIN,
                text_color_disabled=TEXT_MUTED,

                bg_color="transparent",
                font=("Consolas", 12)
            )

            row = 0
            col = i

            rb.grid(
                row=row,
                column=col,
                padx=(0, 12),
                pady=0,
                sticky="w"
            )

            self.radios.append(rb)


# Engine
class SceneEngine:
    def __init__(self, video, output, cfg, logbox=None, progressbar=None, previewer=None, preview_enabled=True):
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

    def stop(self):
        self._stop = True

        if self._ffmpeg_proc:
            try:
                self._ffmpeg_proc.send_signal(signal.CTRL_BREAK_EVENT)
                self._ffmpeg_proc.kill()
            except Exception:
                pass
            self._ffmpeg_proc = None

        try:
            if self._thumb_container:
                self._thumb_container.close()
        except Exception:
            pass

        self._thumb_container = None

    def total_time(self):
        if not self._start_time:
            return "--:--"
        end = self._end_time or time.time()
        elapsed = int(end - self._start_time)
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

        # Show video info in preview
        if self.previewer and not self._video_info_shown:
            info_text = self._get_video_info_text()
            self.previewer.update_info(info_text)
            self._video_info_shown = True

        scenes = self._detect_scenes_progressive() if scene_mode else self._fixed_interval()
        if not scenes or self._stop:
            return False

        self._cut_scenes(scenes)

        self._end_time = time.time()

        return True



    def _get_video_info_text(self):
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=bit_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            self.video
        ]
        try:
            out = subprocess.check_output(cmd).decode().splitlines()
            width, height, fps, bitrate = out
            num, den = fps.split("/")
            fps_float = round(int(num) / int(den), 2)
            return f"{width}x{height} | FPS: {fps_float} | Bitrate: {int(bitrate)/1000:.0f} kbps"
        except Exception:
            return "Video info unavailable"

    def _detect_scenes_progressive(self):
        threshold = self._map_threshold()
        min_dur = self.cfg["MIN_FINAL_DURATION"]
        fps = self._get_video_fps()

        video = open_video(self.video)

        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)

        scene_manager.add_detector(
            ContentDetector(
                threshold=threshold,
                min_scene_len=int(min_dur * fps)
            )
        )

        video_duration = video.duration.get_seconds()

        def _progress_cb(frame_num, _):
            if self._stop:
                return

            if not self.progress:
                return

            frame_idx = _safe_frame_index(frame_num)
            if frame_idx <= 0:
                return

            current_time = frame_idx / fps
            ratio = min(current_time / video_duration, 1.0)

            ratio = max(self._analysis_ratio, ratio)
            self._analysis_ratio = ratio

            self.progress.update(ratio * 0.4)

        try:
            scene_manager.detect_scenes(
                video=video,
                callback=_progress_cb
            )

            if self._stop:
                return []

            scene_list = scene_manager.get_scene_list()
            video_seconds = video.duration.get_seconds()
        finally:
            try:
                video.close()
            except Exception:
                pass

        if not scene_list:
            total_frames = int(video_seconds * fps)
            self.detected = 1
            return [(0, total_frames)]

        scenes = []
        for start, end in scene_list:
            scenes.append((
                start.get_frames(),
                end.get_frames()
            ))

        self.detected = len(scenes)
        return scenes

    def _fixed_interval(self):
        interval = self.cfg.get("FIXED_INTERVAL", 10)
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            self.video
        ]
        duration = float(subprocess.check_output(cmd).decode().strip())
        fps = self._get_video_fps()

        self._total_frames = int(duration * fps)

        scenes, t = [], 0.0

        while t < duration:
            scenes.append((
                int(t * fps),
                int(min(t + interval, duration) * fps)
            ))

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

        self.total, self.done = len(scenes), 0
        encoder = self.cfg.get("ENCODER", "cpu")

        fps = self._get_video_fps()

        for idx, (start_frame, end_frame) in enumerate(scenes, 1):

            if self._stop:
                break

            # proteção contra cenas degeneradas
            if end_frame <= start_frame:
                continue

            if self.previewer and self.preview_enabled:
                mid_frame = (start_frame + end_frame) // 2
                mid_time = mid_frame / fps
                thumb = self._generate_thumbnail(mid_time)

                if thumb:
                    self.previewer.update_image(thumb)

            OFFSET_BY_PROFILE = {
                "Low": 0.12,
                "Normal": 0.08,
                "High": 0.05
            }

            SCENE_START_OFFSET_FRAMES = max(
                1,
                int(OFFSET_BY_PROFILE[self.cfg["label"]] * fps)
            )

            start_frame = min(
                start_frame + SCENE_START_OFFSET_FRAMES,
                end_frame - 1
            )


            start_time = start_frame / fps
            end_time = end_frame / fps

            duration = max((end_frame - start_frame) / fps, 0.01)
            outfile = os.path.join(outdir, f"scene_{idx:04d}.mp4")
            cmd = [
                "ffmpeg",
                "-y",
                "-i", self.video,
                "-ss", f"{start_time:.6f}",
                "-t", f"{duration:.6f}",
                "-map", "0:v:0",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-reset_timestamps", "1",
                outfile
            ]

            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )

            self._ffmpeg_proc.wait()
            self._ffmpeg_proc = None

            self.done += 1

            if self.log:
                self.log.write_status(
                    detected=self.detected,
                    cut=self.done,
                    eta=self._calculate_eta()
                )

            if self.progress and self.total:
                cut_ratio = self.done / self.total
                # análise = 40%, corte = 60%
                self.progress.update(0.4 + cut_ratio * 0.6)

        if self.progress and not self._stop and self.done == self.total:
            self.progress.mark_finished()

        if self._thumb_container:
            try:
                self._thumb_container.close()
            except Exception:
                pass
            self._thumb_container = None

    def _generate_thumbnail(self, timestamp):
        try:
            if self._thumb_container:
                try:
                    self._thumb_container.close()
                except Exception:
                    pass

            self._thumb_container = av.open(self.video)

            stream = self._thumb_container.streams.video[0]
            time_base = float(stream.time_base)

            target_pts = int(timestamp / time_base)

            self._thumb_container.seek(
                target_pts,
                backward=True,
                any_frame=True
            )

            for frame in self._thumb_container.decode(video=0):
                if frame.pts is None:
                    continue

                if frame.pts >= target_pts:
                    img = frame.to_image()
                    img = resize_for_preview(img)
                    return img

        except Exception:
            return None

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

        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            self.video
        ]
        num, den = subprocess.check_output(cmd).decode().strip().split("/")
        self._fps = float(num) / float(den)
        return self._fps

    def _map_threshold(self):
        # Ajuste empírico baseado no ContentDetector
        base = self.cfg["THRESHOLD"]

        if base >= 45:
            return 42.0  # Low
        elif base >= 30:
            return 30.0  # Normal
        else:
            return 18.0  # High


class FaceDetectionEngine:
    def __init__(self, video, output, logbox=None, progressbar=None,
                 previewer=None, profile="Normal", accel="cpu", preview_enabled=True):

        self.video = video
        self.output = output
        self.log = logbox
        self.progress = progressbar
        self.previewer = previewer
        self.preview_enabled = preview_enabled

        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "Face detection requires PyTorch, but it is not installed."
            )

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
            "Low": {
                "conf": 0.45,
                "min_size": 64,
                "ttl": 0.6,
                "min_frames": 0.8,
                "min_valid_ratio": 0.75,
                "min_sharpness": 60
            },
            "Normal": {
                "conf": 0.35,
                "min_size": 40,
                "ttl": 1.2,
                "min_frames": 0.5,
                "min_valid_ratio": 0.6,
                "min_sharpness": 40
            },
            "High": {
                "conf": 0.22,
                "min_size": 24,
                "ttl": 2.5,
                "min_frames": 0.25,
                "min_valid_ratio": 0.35,
                "min_sharpness": 20
            }
        }[profile]

        self.last_preview = 0

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def stop(self):
        self._stop = True

    def total_time(self):
        if not self._start_time:
            return "--:--"
        end = self._end_time or time.time()
        elapsed = int(end - self._start_time)
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
        left_eye = lm[33]
        right_eye = lm[263]
        nose = lm[1]
        mouth = lm[13]

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

        outdir = build_output_dir(
            self.output,
            mode="faces",
            profile=self.profile,
            accel=self.accel
        )

        frame_idx = 0
        track_id = 0
        min_lm_frames = 1 if self.profile == "High" else 2

        while cap.isOpened():
            if self._stop:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            use_cuda = self.device.startswith("cuda")

            iou = 0.45

            results = self.model.predict(
                frame,
                conf=self.profile_cfg["conf"],
                iou=iou,
                imgsz=640 if frame.shape[1] <= 1280 else 800,
                device=self.device,
                half=use_cuda,
                verbose=False
            )[0]

            new_tracks = []

            for box in results.boxes.xyxy:
                box = box.squeeze()  # garante 1D
                x1, y1, x2, y2 = map(int, box.tolist())

                face_raw = frame[y1:y2, x1:x2]
                w, h = x2 - x1, y2 - y1

                if w < self.profile_cfg["min_size"] or h < self.profile_cfg["min_size"]:
                    continue

                aspect = w / h
                if not (0.65 <= aspect <= 1.35):
                    continue

                h_frame, w_frame, _ = frame.shape

                expand_x = int((x2 - x1) * 0.15)  # 15% lateral (perfil)
                expand_y_top = int((y2 - y1) * 0.15)  # 35% para cima (cabelo)
                expand_y_bot = int((y2 - y1) * 0.15)  # 10% para baixo

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
                    if self._iou(t["box"], (x1, y1, x2, y2)) > iou:
                        t["box"] = (x1, y1, x2, y2)
                        t["ttl"] = ttl_frames
                        t["frames"] += 1

                        h, w = face_raw.shape[:2]
                        cx1 = int(w * 0.25)
                        cx2 = int(w * 0.75)
                        cy1 = int(h * 0.25)
                        cy2 = int(h * 0.75)

                        center_face = face_raw[cy1:cy2, cx1:cx2]
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

                    h, w = face_raw.shape[:2]
                    cx1 = int(w * 0.25)
                    cx2 = int(w * 0.75)
                    cy1 = int(h * 0.25)
                    cy2 = int(h * 0.75)

                    center_face = face_raw[cy1:cy2, cx1:cx2]
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

                    if (
                        t["frames"] >= min_required and
                        (t["valid"] / max(t["frames"], 1)) >= cfg["min_valid_ratio"] and
                        t["score"] >= cfg["min_sharpness"]
                    ):
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
                    h, w, _ = draw.shape

                    for t in tracks:
                        x1, y1, x2, y2 = t["box"]
                        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    img = Image.fromarray(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
                    img = resize_for_preview(img)

                    if img:
                        self.previewer.update_image(img)
                        self.last_preview = now

            if self.log:
                self.log.write_status(
                    detected=self.detected,
                    cut=self.done,
                    eta=self._calculate_eta(frame_idx, total_frames)
                )

            if self.progress:
                ratio = frame_idx / total_frames
                ratio = max(self._face_ratio, ratio)  # impede regressão
                self._face_ratio = ratio
                self.progress.update(ratio)


        for t in tracks:
            cfg = self.profile_cfg
            min_required = max(3, int(fps * cfg["min_frames"]))

            if (
                t["frames"] >= min_required and
                (t["valid"] / max(t["frames"], 1)) >= cfg["min_valid_ratio"] and
                t["score"] >= cfg["min_sharpness"]
            ):
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


# App
class SceneCutterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Scenespy - Scene Cutter")
        self.geometry("1000x650")
        self.engine = None
        self.running = False
        self.stop_pending = False

        self.preview_enabled = ENABLE_PREVIEW_DEFAULT
        self.available_accel = detect_available_accel()
        self._build_ui()

    def _build_ui(self):
        # Left panel
        self.left = ctk.CTkFrame(
            self,
            width=300,
            fg_color=BG_PANEL,
            border_width=1,
            border_color=BORDER_SOFT2,
            corner_radius=0
        )
        self.left.pack(side="left", fill="y", padx=10, pady=10)

        files = Section(self.left, "Files")
        files.pack(fill="x", padx=10, pady=8)

        self.video_selector = FileSelector(files, "Source video")
        self.video_selector.pack(fill="x", padx=12)

        self.output_selector = DirectorySelector(files, "Output folder")
        self.output_selector.pack(fill="x", padx=12)

        mode = Section(self.left, "Cut Mode")
        mode.pack(fill="x", padx=12)

        self.cut_mode = ctk.StringVar(value="scene")
        self.cut_mode.trace_add("write", self._on_cut_mode_change)

        options = [
            ("Scene detection", "scene"),
            ("Every seconds", "interval"),
            ("Detect faces", "faces"),
        ]

        group = RadioGroup(
            mode,
            self.cut_mode,
            options,
            radio_width=150
        )
        group.pack(fill="x", padx=12, pady=(0, 6))

        self.mode_radios = group.radios

        if not TORCH_AVAILABLE:
            for rb in self.mode_radios:
                if rb.cget("value") == "faces":
                    rb.configure(state="disabled")

        vcmd = (self.register(self._validate_interval), "%P")

        self.interval_entry = ctk.CTkEntry(
            mode,
            height=25,
            width=90,
            fg_color=BG_MAIN,
            border_color=BORDER_SOFT,
            border_width=1,
            corner_radius=15,
            text_color=TEXT_MAIN,
            placeholder_text_color=TEXT_MUTED,
            placeholder_text="Seconds",
            validate="key",
            validatecommand=vcmd
        )

        profile = Section(self.left, "Detection Sensitivity")
        profile.pack(fill="x", padx=12, pady=8)

        self.profile = ctk.StringVar(value="Normal")
        options = [(cfg["label"], key) for key, cfg in PROFILES.items()]

        group = RadioGroup(
            profile,
            self.profile,
            options,
            radio_width=90,
        )
        group.pack(fill="x", padx=12)

        self.profile_radios = group.radios

        accel_section = Section(self.left, "Hardware Acceleration (Inference)")
        accel_section.pack(fill="x", padx=12)

        self.accel = ctk.StringVar(value="cpu")
        options = [(val.upper(), val) for val in ACCEL_OPTIONS]

        group = RadioGroup(
            accel_section,
            self.accel,
            options,
            radio_width=85
        )
        group.pack(fill="x", padx=12)

        self.accel_radios = group.radios
        self.update_accel_radios()

        self.start_btn = ctk.CTkButton(
            self.left,
            text="Start",
            corner_radius=15,
            fg_color=ACCENT,
            hover_color="#4f46e5",
            text_color="white",
            command=self.toggle_start
        )
        self.start_btn.pack(pady=20)

        self.log = LogBox(self.left, height=220)
        self.log.pack(fill="x", padx=10, pady=10)

        # Right panel
        self.right = ctk.CTkFrame(
            self,
            width=300,
            fg_color=BG_PANEL,
            border_width=1,
            border_color=BORDER_SOFT2,
            corner_radius=15
        )
        self.right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        section = Section(self.right, "Preview")
        section.pack(padx=10, pady=10, fill="x")

        self.preview_switch = ctk.CTkSwitch(section, text="Show Thumbnail", command=self.toggle_preview)
        self.preview_switch.pack(anchor="e", padx=10, pady=8)

        if self.preview_enabled:
            self.preview_switch.select()
            self.toggle_preview()

        self.preview_frame = PreviewFrame(self.right)
        self.preview_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.progress = ProgressBar(self.right)
        self.progress.pack(fill="x", padx=20, pady=10)

        self._on_cut_mode_change()

    def toggle_preview(self):
        self.preview_enabled = self.preview_switch.get()
        if not self.preview_enabled:
            self.preview_frame.clear_all()

    def toggle_start(self):
        if self.running:
            self.confirm_stop()
        else:
            if self.engine is not None:
                return  # proteção contra clique duplo
            self.start_process()

    def start_process(self):
        video = self.video_selector.get()
        ext = os.path.splitext(video)[1].lower()
        output = self.output_selector.get()

        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            self.log.clear_status()
            self.log.status_lines[0] = "Unsupported file type"
            self.log.write_status()
            return

        if not os.path.isfile(video) or not os.path.isdir(output):
            self.log.clear_status()
            self.log.status_lines[0] = "Invalid paths!"
            self.log.write_status()
            return

        if not is_valid_video_file(video):
            self.log.clear_status()
            self.log.status_lines[0] = "Invalid or unsupported video file"
            self.log.write_status()
            return

        self.cleanup_process(reason="reset")

        if self.log:
            self.log.clear_status()

        self.progress._enabled = True
        self.running = True

        self.set_ui_state(True)
        self.start_btn.configure(text="Stop", fg_color=DANGER, hover_color="#dc2626")

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

        mode = self.cut_mode.get()

        if mode == "faces":
            self.engine = FaceDetectionEngine(
                video,
                output,
                logbox=self.log,
                progressbar=self.progress,
                previewer=self.preview_frame,
                profile=self.profile.get(),
                accel=inference,
                preview_enabled=self.preview_enabled
            )

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
                video,
                output,
                cfg,
                logbox=self.log,
                progressbar=self.progress,
                previewer=self.preview_frame,
                preview_enabled=self.preview_enabled
            )

        threading.Thread(target=self.run_engine, args=(scene_mode,), daemon=True).start()

    def stop_process(self):
        if not self.engine:
            return
        self.engine.stop()

    def run_engine(self, scene_mode):
        result = False
        try:
            result = self.engine.run(scene_mode=scene_mode)
        except Exception as e:
            print("Error:", e)
        finally:
            total_time = None
            engine = self.engine
            stopped = engine._stop if engine else False

            if result and self.engine:
                total_time = self.engine.total_time()

            self.engine = None
            self.stop_pending = False

            self.after(
                0,
                lambda: self.reset_ui(
                    finished=result,
                    total_time=total_time,
                    stopped=stopped
                )
            )

    def reset_ui(self, finished=False, total_time=None, stopped=False):
        self.stop_pending = False
        self.running = False
        self.start_btn.configure(
            text="Start",
            fg_color="#4ade80",
            state="normal"
        )
        self.set_ui_state(False)

        if stopped:
            self.cleanup_process(reason="stop")
        elif finished:
            self.cleanup_process(reason="finish", total_time=total_time)

    def set_ui_state(self, disabled):
        state = "disabled" if disabled else "normal"

        for widget in [
            self.video_selector.button,
            self.output_selector.button,
            *self.mode_radios,
            *self.profile_radios,
            *self.accel_radios
        ]:
            widget.configure(state=state)

        self.video_selector.entry.configure(state=state)
        self.output_selector.entry.configure(state=state)

        if self.cut_mode.get() == "interval":
            self.interval_entry.configure(
                state=state if disabled else "normal"
            )

        # Preview switch
        self.preview_switch.configure(
            state="disabled" if self.running else "normal"
        )

    def _on_cut_mode_change(self, *args):
        if self.cut_mode.get() == "interval":
            self.interval_entry.configure(state="normal")
            self.interval_entry.pack(anchor="n", padx=24, pady=(0, 6))
        else:
            self.interval_entry.pack_forget()

        self.update_accel_radios()

    def run_face_engine(self):
        result = False
        try:
            result = self.engine.run()
        except Exception as e:
            print("Face engine error:", e)
        finally:
            total_time = None
            if result and self.engine:
                total_time = self.engine.total_time()

            engine = self.engine
            stopped = engine._stop if engine else False
            self.engine = None

            self.stop_pending = False


            self.after(
                0,
                lambda: self.reset_ui(
                    finished=result,
                    total_time=total_time,
                    stopped=stopped
                )
            )

    def update_accel_radios(self):
        mode = self.cut_mode.get()
        compat = MODE_ACCEL_COMPAT.get(mode, {})

        allowed = set()
        allowed |= compat.get("encoder", set())
        allowed |= compat.get("inference", set())

        enabled = allowed & self.available_accel

        enabled.add("cpu")

        for rb in self.accel_radios:
            value = rb.cget("value")
            rb.configure(state="normal" if value in enabled else "disabled")

        if self.accel.get() not in enabled:
            self.accel.set("cpu")

    def cleanup_process(self, reason="reset", total_time=None):
        if self.preview_frame:
            self.preview_frame.clear_all()

        if self.progress and reason in ("stop", "reset"):
            self.progress.reset()

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

        gc.collect()

    def confirm_stop(self):
        if not self.running or self.engine is None:
            return
        if self.stop_pending:
            return

        import tkinter.messagebox as mb

        answer = mb.askyesno(
            "Confirm stop",
            "The process is still running.\nDo you really want to stop it?"
        )

        if not answer:
            return

        self.stop_pending = True

        self.after(50, self.stop_process)

    def _validate_interval(self, value: str) -> bool:
        if value == "":
            return True

        if not value.isdigit():
            return False

        v = int(value)
        return 1 <= v <= 18000

def resize_for_preview(img, max_w=PREVIEW_MAX_WIDTH, max_h=PREVIEW_MAX_HEIGHT):
    w, h = img.size

    if w <= 0 or h <= 0:
        return None

    scale = min(max_w / w, max_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return img.resize((new_w, new_h), Image.BILINEAR)


def is_valid_video_file(path: str) -> bool:
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_streams",
            "-select_streams", "v",
            "-of", "json",
            path
        ]

        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        import json
        data = json.loads(out)

        streams = data.get("streams", [])
        if not streams:
            return False

        for s in streams:
            if s.get("codec_type") == "video":
                codec = s.get("codec_name", "").lower()
                if codec != "gif":
                    return True

        return False

    except Exception:
        return False


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

# Main
if __name__ == "__main__":
    if not single_instance():
        import tkinter.messagebox as mb
        mb.showerror(
            "Aplicativo já em execução",
            "Este aplicativo já está aberto."
        )
        sys.exit(0)

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    app = SceneCutterApp()
    app.mainloop()