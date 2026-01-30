import os
import sys
import socket
import subprocess
import threading
import time
import datetime
import customtkinter as ctk
import av
import cv2
import numpy as np
import csv
from PIL import Image

# Config
PROFILES = {
    "Low": {"label": "Low", "THRESHOLD": 45.0, "MIN_FINAL_DURATION": 5.5},
    "Normal": {"label": "Normal", "THRESHOLD": 28.0, "MIN_FINAL_DURATION": 1.8},
    "High": {"label": "High", "THRESHOLD": 18.0, "MIN_FINAL_DURATION": 0.9},
}

ACCEL_OPTIONS = ["cpu", "nvidia", "amd", "intel"]
ENABLE_PREVIEW_DEFAULT = True
PREVIEW_INTERVAL = 0.15
PREVIEW_FPS = 1
INSTANCE_SOCKET = None
INSTANCE_PORT = 54321


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
        min_dur = self.cfg["MIN_FINAL_DURATION"]

        scenes = []
        frame_idx = 0

        for packet in container.demux(stream):
            if self._stop:
                break
            for frame in packet.decode():
                frame_idx += 1
                if frame_idx % 10 == 0:
                    scenes.append(frame.time)
                    if self.log:
                        self.log.write_status(detected=len(scenes), cut=self.done)

                    if self.previewer and self.preview_enabled and frame_idx % int(PREVIEW_FPS / PREVIEW_INTERVAL) == 0:
                        img = frame.to_image().resize((420, int(420 * frame.height / frame.width)))
                        self.previewer.update_image(img)

        #
        result = []
        last = 0
        for t in scenes:
            if t - last >= min_dur:
                result.append((last, t))
                last = t

        self.detected = len(result)
        return result

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
        accel = self.cfg.get("ACCEL", "cpu")

        for idx, (start, end) in enumerate(scenes, 1):
            if self._stop:
                break

            if self.previewer and self.preview_enabled:
                mid_time = (start + end) / 2
                thumb = self._generate_thumbnail(mid_time)
                if thumb:
                    self.previewer.update_image(thumb)

            cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", self.video, "-t", f"{end - start:.3f}"]

            if accel == "nvidia":
                cmd += ["-c:v", "h264_nvenc"]
            elif accel == "amd":
                cmd += ["-c:v", "h264_amf"]
            elif accel == "intel":
                cmd += ["-c:v", "h264_qsv"]
            else:
                cmd += ["-c:v", "libx264", "-preset", "veryfast"]

            cmd += ["-crf", "23", "-c:a", "copy", os.path.join(outdir, f"scene_{idx:03d}.mp4")]
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
    def __init__(self, video, output, logbox=None, progressbar=None, previewer=None):
        self.video = video
        self.output = output
        self.log = logbox
        self.progress = progressbar
        self.previewer = previewer

        self._stop = False
        self._start_time = None
        self._end_time = None

        # === contrato igual ao SceneEngine ===
        self.detected = 0   # "Scenes detected" → rostos válidos encontrados
        self.done = 0       # "Scenes cut" → rostos salvos
        self.total = 0

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "models")

        proto = os.path.join(model_dir, "deploy.prototxt")
        weights = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        if not os.path.isfile(proto) or not os.path.isfile(weights):
            raise RuntimeError(
                "Face detection model not found.\n"
                f"Expected:\n{proto}\n{weights}"
            )

        self.detector = cv2.dnn.readNetFromCaffe(proto, weights)

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

    def _is_sharp(self, img):
        return cv2.Laplacian(img, cv2.CV_64F).var() > 120

    # =========================
    # PHASE 1 — ANALYSIS
    # =========================
    def _analyze_faces(self):
        container = av.open(self.video)
        stream = container.streams.video[0]
        total_frames = stream.frames or 1
        processed = 0

        for frame in container.decode(video=0):
            if self._stop:
                break

            processed += 1
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]

            blob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0)
            )

            self.detector.setInput(blob)
            detections = self.detector.forward()

            for i in range(detections.shape[2]):
                if detections[0, 0, i, 2] < 0.85:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                face = img[y1:y2, x1:x2]

                if face.size == 0 or face.shape[0] < 80:
                    continue
                if not self._is_sharp(face):
                    continue

                self.detected += 1

            if self.log:
                self.log.write_status(
                    detected=self.detected,
                    cut=self.done
                )

            if self.previewer and processed % 5 == 0:
                preview = frame.to_image().resize(
                    (420, int(420 * frame.height / frame.width))
                )
                self.previewer.update_image(preview)

            if self.progress:
                self.progress.update(processed / total_frames)

        self.total = self.detected

    # =========================
    # PHASE 2 — CUT & SAVE
    # =========================
    def _cut_faces(self):
        outdir = os.path.join(
            self.output,
            datetime.datetime.now().strftime("faces_%Y%m%d_%H%M%S")
        )
        os.makedirs(outdir, exist_ok=True)

        csv_path = os.path.join(outdir, "faces.csv")
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["file", "id"])

            container = av.open(self.video)
            stream = container.streams.video[0]

            for frame in container.decode(video=0):
                if self._stop:
                    break

                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]

                blob = cv2.dnn.blobFromImage(
                    cv2.resize(img, (300, 300)),
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0)
                )

                self.detector.setInput(blob)
                detections = self.detector.forward()

                for i in range(detections.shape[2]):
                    if detections[0, 0, i, 2] < 0.85:
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    face = img[y1:y2, x1:x2]

                    if face.size == 0 or face.shape[0] < 80:
                        continue
                    if not self._is_sharp(face):
                        continue

                    self.done += 1
                    fname = f"face_{self.done:04d}.png"
                    cv2.imwrite(os.path.join(outdir, fname), face)
                    writer.writerow([fname, self.done])

                    if self.previewer:
                        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        self.previewer.update_image(Image.fromarray(rgb))

                    if self.log:
                        self.log.write_status(
                            detected=self.detected,
                            cut=self.done,
                            eta=self._calculate_eta()
                        )

                    if self.progress and self.total:
                        self.progress.update(self.done / self.total)

    def _calculate_eta(self):
        if self.done == 0:
            return "--:--"

        elapsed = time.time() - self._start_time
        avg = elapsed / self.done
        remaining = max(self.total - self.done, 0)
        eta_seconds = int(avg * remaining)

        m, s = divmod(eta_seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    # =========================
    # RUN
    # =========================
    def run(self):
        self._start_time = time.time()

        self._analyze_faces()
        if self._stop:
            return False

        self._cut_faces()

        self._end_time = time.time()
        return not self._stop



# App
class SceneCutterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Scenespy - Scene Cutter")
        self.geometry("1000x650")
        self.engine = None
        self.running = False
        self.preview_enabled = ENABLE_PREVIEW_DEFAULT
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

        accel_section = Section(self.left, "Hardware Acceleration")
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
        cfg["ACCEL"] = self.accel.get()

        mode = self.cut_mode.get()

        if mode == "faces":
            self.engine = FaceDetectionEngine(
                video,
                output,
                logbox=self.log,
                progressbar=self.progress,
                previewer=self.preview_frame
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

    def run_face_engine(self):
        result = False
        try:
            result = self.engine.run()
        except Exception as e:
            print("Face engine error:", e)
        finally:
            self.after(0, self.reset_ui if not result else lambda: self.reset_ui(finished=True))


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