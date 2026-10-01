"""Small media probing helpers used before loading a file in the player."""

from __future__ import annotations

import json
import subprocess

from .project import MediaItem
from .shared import CREATE_NO_WINDOW, resolve_executable


def _rate(value: str) -> float:
    try:
        numerator, denominator = str(value).split("/", 1)
        return float(numerator) / float(denominator)
    except Exception:
        return 0.0


def probe_media(path: str) -> MediaItem:
    command = [
        resolve_executable("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", path,
    ]
    kwargs = {"capture_output": True, "text": True, "check": True}
    if CREATE_NO_WINDOW:
        kwargs["creationflags"] = CREATE_NO_WINDOW
    result = subprocess.run(command, **kwargs)
    data = json.loads(result.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    duration = float(stream.get("duration") or (data.get("format") or {}).get("duration") or 0)
    fps = _rate(stream.get("avg_frame_rate") or "0/1") or 30.0
    frames = int(stream.get("nb_frames") or round(duration * fps))
    return MediaItem(
        path=path,
        duration=duration,
        fps=fps,
        total_frames=max(1, frames),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
    )
