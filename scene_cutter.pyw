import sys
import socket
import subprocess
import threading
import time
import datetime
import customtkinter as ctk
import av
import cv2
import csv
import os
import mediapipe as mp
import statistics
from collections import deque
from PIL import Image
from ultralytics import YOLO

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False



# Config
PROFILES = {
    "Low": {"label": "Low", "THRESHOLD": 45.0, "MIN_FINAL_DURATION": 5.0},
    "Normal": {"label": "Normal", "THRESHOLD": 30.0, "MIN_FINAL_DURATION": 2.5},
    "High": {"label": "High", "THRESHOLD": 20.0, "MIN_FINAL_DURATION": 1.0},
}

ACCEL_OPTIONS = ["cpu", "nvidia", "amd", "intel"]
ENABLE_PREVIEW_DEFAULT = True
PREVIEW_INTERVAL = 0.15
PREVIEW_FPS = 1
INSTANCE_SOCKET = None
INSTANCE_PORT = 54321


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


# Widgets
class Section(ctk.CTkFrame):
    def __init__(self, master, title, **kwargs):
        super().__init__(master, fg_color="gray17", corner_radius=5, **kwargs)
        ctk.CTkLabel(
            self, text=title, font=("Consolas", 14, "bold")).pack(anchor="w", padx=12, pady=(8, 4))


class LabeledEntry(ctk.CTkFrame):
    def __init__(self, master, label, placeholder="", width=160):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, font=("Consolas", 12)).pack(anchor="w")
        self.entry = ctk.CTkEntry(self, placeholder_text=placeholder, width=width)
        self.entry.pack(pady=(2, 8), fill="x")

    def get(self):
        return self.entry.get()


class LogBox(ctk.CTkTextbox):
    def __init__(self, master, height=140):
        super().__init__(master, height=height, fg_color="#0f0f14", corner_radius=3)
        self.configure(state="disabled", font=("Consolas", 11))
        self.status_lines = [""] * 3
        self._init_status_lines()

    def write_status(self, detected=None, cut=None, eta=None):
        self.configure(state="normal")

        if detected is not None:
            self.status_lines[0] = f"Scenes detected: {detected}"
        if cut is not None:
            self.status_lines[1] = f"Scenes cut: {cut}"
        if eta is not None:
            self.status_lines[2] = f"Estimated time: {eta}"

        self.delete("1.0", "end")
        for line in self.status_lines:
            self.insert("end", line + "\n")
        self.see("end")
        self.configure(state="disabled")

    def write_message(self, text, color=None):
        self.configure(state="normal")
        self.insert("end", text + "\n")
        if color:
            tag = f"msg_{color}"
            self.tag_add(tag, "end-2l", "end-1l")
            self.tag_config(tag, foreground=color)
        self.see("end")
        self.configure(state="disabled")

    def clear_status(self):
        self.configure(state="normal")
        self.status_lines = [""] * 4
        self.delete("1.0", "end")
        for line in self.status_lines:
            self.insert("end", line + "\n")
        self.configure(state="disabled")

    def _init_status_lines(self):
        self.configure(state="normal")
        for _ in self.status_lines:
            self.insert("end", "\n")
        self.configure(state="disabled")


class ProgressBar(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.bar = ctk.CTkProgressBar(self)
        self.bar.pack(fill="x", pady=4)
        self.bar.set(0)
        self.label = ctk.CTkLabel(self, text="0%", font=("Consolas", 11))
        self.label.pack(anchor="e")

    def update(self, value):
        self.bar.set(value)
        self.label.configure(text=f"{int(value * 100)}%")


class PreviewFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="gray25", corner_radius=3)
        self.info_label = ctk.CTkLabel(self, text="", font=("Consolas", 10))
        self.info_label.pack(anchor="n", pady=4)
        self.label = ctk.CTkLabel(self, text="")
        self.label.pack(expand=True)
        self._img_ref = None

    def update_image(self, image):
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

        self.entry = ctk.CTkEntry(row, width=width, corner_radius=2)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.configure(state="disabled")

        self.button = ctk.CTkButton(row, text="...", width=30, command=self.select,
                                    corner_radius=5, fg_color="gray25", hover_color="gray35")
        self.button.pack(side="right", padx=(6, 0))

    def select(self):
        import tkinter.filedialog as fd
        path = fd.askopenfilename()
        if path:
            self.entry.configure(state="normal")
            self.entry.delete(0, "end")
            self.entry.insert(0, path)
            self.entry.configure(state="disabled")

    def get(self):
        return self.entry.get()


class DirectorySelector(FileSelector):
    def select(self):
        import tkinter.filedialog as fd
        path = fd.askdirectory()
        if path:
            self.entry.configure(state="normal")
            self.entry.delete(0, "end")
            self.entry.insert(0, path)
            self.entry.configure(state="disabled")


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

    def run(self, scene_mode=True):
        self._start_time = time.time()

        # Show video info in preview
        if self.previewer and not self._video_info_shown:
            info_text = self._get_video_info_text()
            self.previewer.update_info(info_text)
            self._video_info_shown = True

        scenes = self._detect_scenes_progressive() if scene_mode else self._fixed_interval()
        if not scenes or self._stop:
            return False

        if self.log:
            self.log.write_message(f"🎬 Scenes detected: {len(scenes)}")

        self._cut_scenes(scenes)
        self._end_time = time.time()
        return not self._stop

    def _get_video_info_text(self):
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,bit_rate",
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
        container = av.open(self.video)
        stream = container.streams.video[0]

        threshold = self.cfg["THRESHOLD"]
        min_dur = self.cfg["MIN_FINAL_DURATION"]

        diff_window = deque(maxlen=12)
        adaptive_threshold = threshold
        calibrated = False

        HARD_CUT_MULT = 2.4
        calibration_time = 3.0  # segundos iniciais só para medir

        scenes = []
        last_cut_time = 0.0
        last_frame_time = 0.0
        last_scene_change_time = None

        prev_hist = None
        frame_idx = 0

        for packet in container.demux(stream):
            if self._stop:
                break

            for frame in packet.decode():
                frame_idx += 1

                tail_window = 2.0  # segundos finais

                if frame.time is not None and container.duration:
                    video_end = container.duration / av.time_base
                    if frame.time < video_end - tail_window:
                        if frame_idx % 2 != 0:
                            continue

                t = frame.time
                if t is None:
                    continue

                last_frame_time = t

                img = frame.to_ndarray(format="bgr24")
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                hist = cv2.calcHist(
                    [hsv], [0, 1],
                    None,
                    [50, 60],
                    [0, 180, 0, 256]
                )
                cv2.normalize(hist, hist)

                if prev_hist is not None:
                    diff = cv2.compareHist(
                        prev_hist, hist,
                        cv2.HISTCMP_CHISQR
                    )

                    diff_window.append(diff)

                    # calibração inicial
                    if not calibrated:
                        if t >= calibration_time and len(diff_window) >= 8:
                            mean = statistics.mean(diff_window)
                            std = statistics.pstdev(diff_window)

                            adaptive_threshold = mean + 2.2 * std
                            adaptive_threshold = max(
                                threshold * 0.7,
                                min(adaptive_threshold, threshold * 1.3)
                            )
                            calibrated = True

                        prev_hist = hist
                        continue

                    # confirmação temporal
                    strong_hits = sum(
                        1 for d in diff_window
                        if d > adaptive_threshold
                    )

                    hard_cut = diff > adaptive_threshold * HARD_CUT_MULT

                    if hard_cut:
                        last_scene_change_time = t
                    elif strong_hits >= 2 and last_scene_change_time is None:
                        last_scene_change_time = t

                    # corte confirmado
                    if (
                            (hard_cut or strong_hits >= 2) and
                            last_scene_change_time is not None and
                            (last_scene_change_time - last_cut_time) >= min_dur
                    ):
                        cut_time = last_scene_change_time
                        cut_time = max(cut_time, last_cut_time + 0.05)

                        scenes.append((last_cut_time, cut_time))
                        last_cut_time = cut_time
                        last_scene_change_time = None

                        if self.log:
                            self.log.write_status(
                                detected=len(scenes),
                                cut=self.done
                            )

                prev_hist = hist

                # preview
                if self.previewer and self.preview_enabled:
                    if frame_idx % int(PREVIEW_FPS / PREVIEW_INTERVAL) == 0:
                        img_pil = frame.to_image().resize(
                            (420, int(420 * frame.height / frame.width))
                        )
                        self.previewer.update_image(img_pil)

        if last_scene_change_time and last_scene_change_time > last_cut_time:
            cut = max(last_scene_change_time, last_cut_time + 0.05)
            scenes.append((last_cut_time, cut))
            if last_frame_time - cut >= 0.05:
                scenes.append((cut, last_frame_time))
        else:
            if last_frame_time - last_cut_time >= 0.05:
                scenes.append((last_cut_time, last_frame_time))

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

        scenes, t = [], 0.0
        while t < duration:
            scenes.append((t, min(t + interval, duration)))
            t += interval

        self.detected = len(scenes)
        return scenes

    def _cut_scenes(self, scenes):
        outdir = os.path.join(
            self.output,
            datetime.datetime.now().strftime("scenes_%Y%m%d_%H%M%S")
        )
        os.makedirs(outdir, exist_ok=True)

        self.total, self.done = len(scenes), 0
        encoder = self.cfg.get("ENCODER", "cpu")

        for idx, (start, end) in enumerate(scenes, 1):
            if self._stop:
                break

            # proteção contra cenas degeneradas
            if end <= start:
                continue

            if self.previewer and self.preview_enabled:
                mid_time = (start + end) / 2
                thumb = self._generate_thumbnail(mid_time)
                if thumb:
                    self.previewer.update_image(thumb)

            epsilon = 0.001  # 1 ms real
            safe_end = max(start, end - epsilon)

            cmd = [
                "ffmpeg",
                "-y",
                "-i", self.video,
                "-filter_complex",
                (
                    f"[0:v]trim=start={start:.6f}:end={safe_end:.6f},"
                    f"setpts=PTS-STARTPTS[v];"
                    f"[0:a]atrim=start={start:.6f}:end={safe_end:.6f},"
                    f"asetpts=PTS-STARTPTS[a]"

                ),
                "-map", "[v]",
                "-map", "[a]",
                "-reset_timestamps", "1",
            ]

            if encoder == "nvidia":
                cmd += [
                    "-c:v", "h264_nvenc",
                    "-g", "1",
                    "-bf", "0"
                ]
            elif encoder == "amd":
                cmd += [
                    "-c:v", "h264_amf",
                    "-g", "1",
                    "-bf", "0"
                ]
            elif encoder == "intel":
                cmd += [
                    "-c:v", "h264_qsv",
                    "-g", "1",
                    "-bf", "0"
                ]
            else:
                cmd += [
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-x264-params", "keyint=1:no-scenecut=1"
                ]

            # NÃO copiar áudio para evitar drift temporal
            cmd += [
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                os.path.join(outdir, f"scene_{idx:03d}.mp4")
            ]

            cmd += ["-vsync", "0"]

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            self.done += 1

            if self.log:
                self.log.write_status(
                    detected=self.detected,
                    cut=self.done,
                    eta=self._calculate_eta()
                )

            if self.progress:
                self.progress.update(self.done / self.total)

    def _generate_thumbnail(self, timestamp):
        try:
            container = av.open(self.video)
            container.seek(int(timestamp * av.time_base))
            for frame in container.decode(video=0):
                img = frame.to_image()
                img = img.resize((420, int(420 * frame.height / frame.width)))
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

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", "yolov8n-face.pt")
        self.model = YOLO(model_path)

        self.profile_cfg = {
            "Low":    {"conf": 0.45, "min_size": 56, "ttl": 0.6},
            "Normal": {"conf": 0.35, "min_size": 40, "ttl": 1.2},
            "High":   {"conf": 0.25, "min_size": 32, "ttl": 2.0},
        }[profile]

        self.last_preview = 0

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
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

        if abs(left_eye.x - right_eye.x) < 0.035:
            return False

        return True

    def run(self):
        self._start_time = time.time()

        cap = cv2.VideoCapture(self.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        ttl_frames = int(fps * self.profile_cfg["ttl"])
        tracks = []

        outdir = os.path.join(
            self.output,
            datetime.datetime.now().strftime("faces_%Y%m%d_%H%M%S")
        )
        os.makedirs(outdir, exist_ok=True)

        csv_path = os.path.join(outdir, "faces.csv")
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["file", "track_id"])

        frame_idx = 0
        track_id = 0

        while cap.isOpened():
            if self._stop:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            use_cuda = self.device.startswith("cuda")


            results = self.model.predict(
                frame,
                conf=self.profile_cfg["conf"],
                iou=0.45,
                imgsz=800,
                device=self.device,
                half=use_cuda,
                verbose=False
            )[0]

            new_tracks = []

            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                w, h = x2 - x1, y2 - y1

                if w < self.profile_cfg["min_size"] or h < self.profile_cfg["min_size"]:
                    continue

                aspect = w / h
                if not (0.65 <= aspect <= 1.35):
                    continue

                face = frame[y1:y2, x1:x2]
                if face.size == 0 or self._skin_ratio(face) < 0.15:
                    continue

                matched = False

                for t in tracks:
                    if self._iou(t["box"], (x1, y1, x2, y2)) > 0.35:
                        t["box"] = (x1, y1, x2, y2)
                        t["ttl"] = ttl_frames
                        t["frames"] += 1

                        sharp = cv2.Laplacian(face, cv2.CV_64F).var()
                        if sharp > t["score"]:
                            t["score"] = sharp
                            t["face"] = face.copy()

                        if t["frames"] >= 3 and self._valid_landmarks(face):
                            t["valid"] += 1

                        matched = True
                        break

                if not matched:
                    track_id += 1
                    self.detected += 1

                    new_tracks.append({
                        "id": track_id,
                        "box": (x1, y1, x2, y2),
                        "ttl": ttl_frames,
                        "frames": 1,
                        "valid": 0,
                        "score": cv2.Laplacian(face, cv2.CV_64F).var(),
                        "face": face.copy()
                    })

            for t in tracks:
                t["ttl"] -= 1
                if t["ttl"] <= 0:
                    if (
                        t["frames"] >= fps * 0.5 and
                        (t["valid"] / max(t["frames"], 1)) >= 0.6 and
                        t["score"] > 40
                    ):
                        self.done += 1
                        fname = f"face_{self.done:04d}.png"
                        cv2.imwrite(os.path.join(outdir, fname), t["face"])
                        writer.writerow([fname, t["id"]])

            tracks = [t for t in tracks if t["ttl"] > 0] + new_tracks

            if self.previewer and self.preview_enabled:
                now = time.time()
                if now - self.last_preview >= PREVIEW_INTERVAL:
                    draw = frame.copy()
                    for t in tracks:
                        x1, y1, x2, y2 = t["box"]
                        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    h, w, _ = draw.shape
                    draw = cv2.resize(draw, (420, int(420 * h / w)))
                    img = Image.fromarray(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
                    self.previewer.update_image(img)
                    self.last_preview = now

            if self.log:
                self.log.write_status(
                    detected=self.detected,
                    cut=self.done,
                    eta=self._calculate_eta(frame_idx, total_frames)
                )

            if self.progress:
                self.progress.update(frame_idx / total_frames)

        csv_file.close()
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
        self.preview_enabled = ENABLE_PREVIEW_DEFAULT
        self.available_accel = detect_available_accel()
        self._build_ui()

    def _build_ui(self):
        # Left panel
        self.left = ctk.CTkFrame(self, width=300)
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

        self.interval_entry = ctk.CTkEntry(
            mode,
            width=90,
            corner_radius=15,
            placeholder_text="Secounds"
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

        self.start_btn = ctk.CTkButton(self.left, text="Start", command=self.toggle_start, corner_radius=50, fg_color="#67679C")
        self.start_btn.pack(pady=20)

        self.log = LogBox(self.left, height=220)
        self.log.pack(fill="x", padx=10, pady=10)

        # Right panel
        self.right = ctk.CTkFrame(self)
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
            self.preview_frame.clear_image()

    def toggle_start(self):
        if self.running:
            self.stop_process()
        else:
            self.start_process()

    def start_process(self):
        video = self.video_selector.get()
        output = self.output_selector.get()

        if not os.path.isfile(video) or not os.path.isdir(output):
            self.log.write_message("Invalid paths!", color="red")
            return

        self.progress.update(0)
        self.running = True
        self.set_ui_state(True)
        self.start_btn.configure(text="Stop", fg_color="#dc2626")

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
            try:
                cfg["FIXED_INTERVAL"] = float(self.interval_entry.get())
            except ValueError:
                self.log.write_message("Invalid interval!", color="red")
                self.reset_ui()
                return

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
        self.running = False
        if self.engine:
            self.engine.stop()
        self.preview_frame.clear_all()
        self.log.clear_status()
        self.log.write_message("Process stopped", color="#facc15")
        self.reset_ui()

    def run_engine(self, scene_mode):
        result = False
        try:
            result = self.engine.run(scene_mode=scene_mode)
        except Exception as e:
            print("Error:", e)
        finally:
            self.after(0, self.reset_ui if not result else lambda: self.reset_ui(finished=True))

    def reset_ui(self, finished=False):
        self.running = False
        self.start_btn.configure(text="Start", fg_color="#4ade80")
        self.set_ui_state(False)
        if finished:
            self.log.write_message(
                f"Process finished [{self.engine.total_time()}]",
                color="#22c55e"
            )
            self.progress.update(1.0)

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

        if self.cut_mode.get() == "interval":
            self.interval_entry.configure(state=state)

        self.preview_switch.configure(state="disabled" if self.running else "normal")

    def _on_cut_mode_change(self, *args):
        if self.cut_mode.get() == "interval":
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
            self.after(0, self.reset_ui if not result else lambda: self.reset_ui(finished=True))

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