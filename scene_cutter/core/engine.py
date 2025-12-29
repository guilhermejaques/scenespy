import os
import shutil
import subprocess
import datetime
import time
import threading
import concurrent.futures

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from core.preview_ffmpeg import FFmpegPreview


# ============================================================
# ENGINE (GUI-SAFE | ALTO DESEMPENHO)
# ============================================================
class SceneEngine:
    def __init__(
        self,
        video_path,
        output_dir,
        profile_cfg,
        progress_cb=None,
        enable_terminal=False
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.cfg = profile_cfg

        self.progress_cb = progress_cb or (lambda msg, pct=None, img=None: None)

        self._stop_flag = False
        self._stop_event = threading.Event()
        self._start_time = None

        self._detected_scenes = 0
        self._last_preview_time = 0.0
        self._preview_interval = 0.8

        self._enable_terminal = enable_terminal

    # ========================== CONTROLE ==========================
    def stop(self):
        self._stop_flag = True
        self._stop_event.set()

    def _emit(self, msg=None, pct=None, img=None):
        if self._stop_flag:
            return
        self.progress_cb(msg, pct, img)

    # ========================== VALIDAÇÃO ==========================
    def validate(self):
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError("Vídeo não encontrado")

        if not shutil.which("ffmpeg"):
            raise EnvironmentError("FFmpeg não encontrado no PATH")

    def create_output_directory(self):
        name = os.path.splitext(os.path.basename(self.video_path))[0]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"{name}_{ts}")
        os.makedirs(path, exist_ok=True)
        return path

    # ========================== DETECÇÃO ==========================
    def detect_scenes(self):
        self._emit("🔍 Detectando cenas...", 5)

        video = open_video(self.video_path)
        video.downscale = self.cfg["DOWNSCALE"]

        manager = SceneManager()
        manager.add_detector(
            ContentDetector(
                threshold=self.cfg["THRESHOLD"],
                min_scene_len=self.cfg["MIN_SCENE_LEN_FRAMES"],
            )
        )

        manager.detect_scenes(video)
        scenes = manager.get_scene_list()

        self._detected_scenes = len(scenes)
        self._emit(f"🎬 {self._detected_scenes} cenas detectadas", 25)

        return scenes

    # ========================== NORMALIZAÇÃO ======================
    def normalize_scenes(self, scenes):
        self._emit("✨ Normalizando timeline...", 40)

        min_dur = self.cfg["MIN_FINAL_DURATION"]
        normalized = []

        buffer_start = None
        buffer_end = None

        for s in scenes:
            if self._stop_flag:
                return []

            start = s[0].get_seconds()
            end = s[1].get_seconds()

            if buffer_start is None:
                buffer_start = start
                buffer_end = end
            else:
                buffer_end = end

            if (buffer_end - buffer_start) >= min_dur:
                normalized.append((buffer_start, buffer_end))
                buffer_start = None
                buffer_end = None

        if buffer_start is not None:
            normalized.append((buffer_start, buffer_end))

        self._emit(f"🎞 {len(normalized)} cenas finais", 55)
        return normalized

    # ========================== CORTE ==============================
    def cut_scenes(self, scenes, out_dir, preview=None):
        total = len(scenes)
        max_workers = min(4, os.cpu_count() or 1)

        progress_lock = threading.Lock()
        completed = 0

        def cut_scene_task(index_scene):
            nonlocal completed

            i, (start, end) = index_scene
            if self._stop_flag:
                return

            duration = end - start
            output = os.path.join(out_dir, f"scene_{i:03d}.mp4")

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", self.video_path,
                    "-ss", f"{start:.3f}",
                    "-t", f"{duration:.3f}",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-c:a", "copy",
                    output,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            with progress_lock:
                completed += 1
                pct = (completed / total) * 100 if total else 0
                self._emit(
                    msg=f"Cortando cenas: {completed}/{total} | Detectadas: {self._detected_scenes}",
                    pct=pct
                )

            if preview:
                now = time.time()
                if now - self._last_preview_time >= self._preview_interval:
                    img = preview.get_frame_at(start + duration / 2)
                    if img:
                        self._last_preview_time = now
                        self._emit(img=img)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(cut_scene_task, enumerate(scenes, start=1)))

    # ========================== RUN ===============================
    def run(self, scene_mode=True):
        self.validate()
        self._start_time = time.time()

        out_dir = self.create_output_directory()
        preview = FFmpegPreview(self.video_path) if self.cfg.get("ENABLE_PREVIEW") else None

        try:
            if scene_mode:
                scenes = self.detect_scenes()
                if self._stop_flag:
                    return None

                scenes = self.normalize_scenes(scenes)
                if self._stop_flag:
                    return None
            else:
                duration = open_video(self.video_path).duration.get_seconds()
                interval = self.cfg.get("FIXED_INTERVAL", 10)
                scenes = [
                    (t, min(t + interval, duration))
                    for t in self._frange(0, duration, interval)
                ]

            self.cut_scenes(scenes, out_dir, preview)

        finally:
            if preview:
                preview.release()

        return None if self._stop_flag else out_dir

    # ========================== UTIL ===============================
    @staticmethod
    def _frange(start, stop, step):
        t = start
        while t < stop:
            yield t
            t += step
