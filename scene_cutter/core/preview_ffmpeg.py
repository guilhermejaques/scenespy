import subprocess
from PIL import Image
import io
import threading
import time

class FFmpegPreview:
    def __init__(self, video_path, min_interval=0.15):
        self.video_path = video_path
        self.min_interval = min_interval
        self._last_emit_time = 0.0
        self._lock = threading.Lock()
        self._released = False

    def get_frame_at(self, seconds, size=(420, 240)):
        if self._released:
            return None
        now = time.time()
        with self._lock:
            if now - self._last_emit_time < self.min_interval:
                return None
            self._last_emit_time = now
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{seconds:.3f}", "-i", self.video_path,
            "-frames:v", "1", "-vf", f"scale={size[0]}:-1",
            "-f", "image2pipe", "-vcodec", "png", "-"
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
            if not proc.stdout:
                return None
            img = Image.open(io.BytesIO(proc.stdout))
            img.load()
            return img
        except Exception:
            return None

    def release(self):
        self._released = True
