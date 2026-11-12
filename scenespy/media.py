"""Small media probing helpers used before loading a file in the player."""

from __future__ import annotations

import json
import statistics
import subprocess

from .project import MediaItem
from .shared import CREATE_NO_WINDOW, resolve_executable


def _rate(value: str) -> float:
    try:
        numerator, denominator = str(value).split("/", 1)
        return float(numerator) / float(denominator)
    except Exception:
        return 0.0


def _frame_times(path: str) -> list[float]:
    command = [
        resolve_executable("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv=p=0", path,
    ]
    kwargs = {"capture_output": True, "text": True, "check": True}
    if CREATE_NO_WINDOW:
        kwargs["creationflags"] = CREATE_NO_WINDOW
    result = subprocess.run(command, **kwargs)
    values = []
    for line in (result.stdout or "").splitlines():
        try:
            values.append(float(line.strip().split(",", 1)[0]))
        except (TypeError, ValueError):
            continue
    if not values:
        return []
    origin = values[0]
    return [max(0.0, value - origin) for value in values]


def _is_genuinely_vfr(times: list[float]) -> bool:
    deltas = [right - left for left, right in zip(times, times[1:])
              if right > left]
    if len(deltas) < 3:
        return False
    typical = statistics.median(deltas)
    tolerance = max(0.0005, typical * 0.02)
    unusual = sum(abs(delta - typical) > tolerance for delta in deltas)
    return unusual / len(deltas) >= 0.01


def probe_media(path: str) -> MediaItem:
    command = [
        resolve_executable("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
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
    nominal_fps = _rate(stream.get("r_frame_rate") or "0/1")
    suspect_vfr = bool(
        nominal_fps and abs(nominal_fps - fps) / max(fps, 0.001) > 0.005)
    try:
        times = _frame_times(path) if suspect_vfr else []
    except (OSError, subprocess.SubprocessError):
        # Timing enrichment must never make an otherwise playable import fail.
        times = []
    variable = _is_genuinely_vfr(times)
    if variable:
        frames = len(times)
    else:
        times = []
    return MediaItem(
        path=path,
        duration=duration,
        fps=fps,
        total_frames=max(1, frames),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        variable_frame_rate=variable,
        frame_times=times,
    )
