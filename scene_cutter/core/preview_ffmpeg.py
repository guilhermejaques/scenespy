import subprocess
from PIL import Image
import io


class FFmpegPreview:
    """
    Gera preview de frame usando FFmpeg (rápido e sem OpenCV).
    """

    def __init__(self, video_path):
        self.video_path = video_path

    def get_frame_at(self, seconds, size=(420, 240)):
        cmd = [
            "ffmpeg",
            "-ss", f"{seconds:.3f}",
            "-i", self.video_path,
            "-frames:v", "1",
            "-vf", f"scale={size[0]}:-1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-"
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return Image.open(io.BytesIO(proc.stdout))
        except Exception:
            return None

    def release(self):
        pass  # compatibilidade com engine
