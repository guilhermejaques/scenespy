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


class TerminalProgressReporter:
    """
    Terminal fixo para mostrar progresso e ETA.
    """
    def __init__(self, update_interval=1.5):
        self.update_interval = update_interval
        self._last = 0
        self._start = time.time()

    def update(self, phase, current, total):
        now = time.time()
        if now - self._last < self.update_interval:
            return
        self._last = now

        elapsed = now - self._start
        avg = elapsed / max(current, 1)
        eta = avg * (total - current)

        pct = (current / total) * 100
        eta_min = int(eta // 60)
        eta_sec = int(eta % 60)

        msg = f"[{phase.upper():>4}] {current}/{total} | {pct:5.1f}% | ETA ~{eta_min}m {eta_sec}s"
        print("\r" + msg, end="", flush=True)

    def finish(self):
        print("\n", flush=True)


class SceneEngine:
    def __init__(self, video_path, output_dir, profile_cfg, progress_cb=None, enable_terminal=True):
        self.video_path = video_path
        self.output_dir = output_dir
        self.cfg = profile_cfg
        self.progress_cb = progress_cb or (lambda msg, pct=None, img=None: None)
        self._stop_flag = False
        self._start_time = None
        self._terminal = TerminalProgressReporter() if enable_terminal else None

    # ========================== Controle ==========================
    def stop(self):
        self._stop_flag = True

    def _emit(self, msg, pct=None, img=None):
        if self._stop_flag:
            return
        self.progress_cb(msg, pct, img)
        if self._terminal and msg:
            print("\r" + msg + " " * 20, end="", flush=True)

    # ========================== Validação =========================
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

    # ========================== Detecção de cenas ==================
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

        # Heartbeat leve, sem spam
        def heartbeat():
            last = 0
            while not self._stop_flag:
                now = time.time()
                if now - last > 2.0:
                    elapsed = int(now - self._start_time)
                    self._emit(f"⏳ Detectando cenas... ({elapsed}s)")
                    last = now
                time.sleep(0.5)

        threading.Thread(target=heartbeat, daemon=True).start()
        manager.detect_scenes(video)
        scenes = manager.get_scene_list()
        self._emit(f"🎬 {len(scenes)} cenas detectadas", 25)
        return scenes

    # ========================== Normalização =====================
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

    # ========================== Corte + ETA ======================
    def cut_scenes(self, scenes, out_dir, preview=None):
        total = len(scenes)
        preview_every = self.cfg.get("PREVIEW_EVERY_N_SCENES", 5)
        max_workers = min(4, os.cpu_count())  # até 4 threads FFmpeg simultâneas

        start_time = time.time()

        def cut_scene_task(i_start_end):
            i, (start, end) = i_start_end
            if self._stop_flag:
                return None

            duration = end - start
            output = os.path.join(out_dir, f"scene_{i:03d}.mp4")

            cmd = [
                "ffmpeg", "-y",
                "-i", self.video_path,
                "-ss", f"{start:.3f}",
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-c:a", "copy",
                output,
            ]

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Atualiza progresso CLI
            elapsed = time.time() - start_time
            avg = elapsed / max(i,1)
            eta = avg * (total - i)
            eta_min, eta_sec = divmod(int(eta), 60)
            pct = 55 + int((i / total) * 45)
            msg = f"Cortando cena {i}/{total} | ETA ~{eta_min}m {eta_sec}s"
            self._emit(msg, pct)

            # Preview controlado
            if preview and i % preview_every == 0:
                img = preview.get_frame_at(start + duration / 2)
                if img:
                    self._emit(None, None, img)

        # Executor limitado para threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(cut_scene_task, enumerate(scenes, start=1)))

        if self._terminal:
            self._terminal.finish()

    # ========================== Execução ==========================
    def run(self, scene_mode=True):
        self.validate()
        self._start_time = time.time()
        out_dir = self.create_output_directory()

        preview = FFmpegPreview(self.video_path) if self.cfg.get("ENABLE_PREVIEW") else None

        try:
            if scene_mode:
                scenes = self.detect_scenes()
                if self._stop_flag: return None
                scenes = self.normalize_scenes(scenes)
                if self._stop_flag: return None
            else:
                duration = open_video(self.video_path).duration.get_seconds()
                interval = self.cfg.get("FIXED_INTERVAL", 10)
                scenes = [(t, min(t+interval, duration)) for t in self._frange(0, duration, interval)]

            self.cut_scenes(scenes, out_dir, preview)
        finally:
            if preview:
                preview.release()

        return None if self._stop_flag else out_dir

    @staticmethod
    def _frange(start, stop, step):
        t = start
        while t < stop:
            yield t
            t += step
