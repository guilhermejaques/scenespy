"""Frame-exact mapping for non-destructive pre-analysis video edits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .shared import ffmpeg_path, run_hidden_cancelable


@dataclass(frozen=True)
class FrameRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


class FrameMap:
    """Map a concatenated analysis stream back onto original source frames."""

    def __init__(self, ranges, total_frames):
        self.total_frames = max(0, int(total_frames))
        normalized = []
        for start, end in ranges:
            start = max(0, min(int(start), self.total_frames))
            end = max(start, min(int(end), self.total_frames))
            if end > start:
                normalized.append(FrameRange(start, end))
        self.ranges = tuple(normalized)
        self._pieces = []
        cursor = 0
        for source in self.ranges:
            self._pieces.append((cursor, cursor + source.length, source))
            cursor += source.length
        self.analysis_frames = cursor

    @classmethod
    def from_segments(cls, segments, excluded, removed, total_frames):
        unavailable = {int(value) for value in excluded} | {
            int(value) for value in removed}
        ranges = [segment for index, segment in enumerate(segments, 1)
                  if index not in unavailable]
        return cls(ranges, total_frames)

    @property
    def is_identity(self):
        if not self.ranges or self.ranges[0].start != 0:
            return False
        return (self.ranges[-1].end == self.total_frames and
                all(left.end == right.start
                    for left, right in zip(self.ranges, self.ranges[1:])))

    @property
    def join_frames(self):
        return tuple(end for _start, end, _source in self._pieces[:-1])

    def to_source(self, analysis_frame, allow_endpoint=False):
        if not self._pieces:
            return None
        frame = int(analysis_frame)
        if allow_endpoint and frame >= self.analysis_frames:
            return self._pieces[-1][2].end
        frame = max(0, min(frame, self.analysis_frames - 1))
        for analysis_start, analysis_end, source in self._pieces:
            if analysis_start <= frame < analysis_end:
                return source.start + frame - analysis_start
        return self._pieces[-1][2].end - 1

    def seconds_to_source(self, seconds, fps):
        fps = max(float(fps or 0), 0.001)
        frame = max(0, round(float(seconds or 0) * fps))
        source = self.to_source(frame, allow_endpoint=True)
        return 0.0 if source is None else source / fps

    def mapped_boundaries(self, analysis_boundaries):
        boundaries = {0, self.total_frames}
        for source in self.ranges:
            boundaries.update((source.start, source.end))
        for frame in analysis_boundaries:
            mapped = self.to_source(frame, allow_endpoint=True)
            if mapped is not None:
                boundaries.add(mapped)
        return sorted(frame for frame in boundaries
                      if 0 <= frame <= self.total_frames)

    def segment_is_included(self, start, end):
        return any(source.start <= start and end <= source.end
                   for source in self.ranges)

    def rebuild_segments(self, analysis_boundaries):
        boundaries = self.mapped_boundaries(analysis_boundaries)
        segments = list(zip(boundaries, boundaries[1:]))
        excluded = [index for index, (start, end) in enumerate(segments, 1)
                    if not self.segment_is_included(start, end)]
        return boundaries, excluded


def create_analysis_proxy(source, target, mapping, fps, stop_cb=None):
    """Create an exact-frame, video-only concatenation for model analysis."""
    if not mapping.ranges:
        raise ValueError("At least one included sequence is required")
    executable = ffmpeg_path()
    if not executable:
        raise RuntimeError("FFmpeg is required for pre-analysis video edits")
    target = str(Path(target))
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    count = len(mapping.ranges)
    if count == 1:
        source_label = "[0:v]"
        prefix = ""
    else:
        labels = "".join(f"[s{index}]" for index in range(count))
        prefix = f"[0:v]split={count}{labels};"
        source_label = None
    trims = []
    outputs = []
    for index, frame_range in enumerate(mapping.ranges):
        input_label = source_label or f"[s{index}]"
        output_label = f"[v{index}]"
        trims.append(
            f"{input_label}trim=start_frame={frame_range.start}:"
            f"end_frame={frame_range.end},setpts=PTS-STARTPTS{output_label}")
        outputs.append(output_label)
    concat = "".join(outputs) + f"concat=n={count}:v=1:a=0[joined]"
    graph = prefix + ";".join(
        trims + [concat, "[joined]pad=ceil(iw/2)*2:ceil(ih/2)*2[outv]"])
    command = [
        executable, "-y", "-v", "error", "-noautorotate", "-i", str(source),
        "-filter_complex", graph, "-map", "[outv]", "-an",
        "-r", f"{float(fps):.12g}", "-fps_mode", "cfr",
        "-frames:v", str(mapping.analysis_frames),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
        "-pix_fmt", "yuv420p", target,
    ]
    result = run_hidden_cancelable(
        command, stop_cb=stop_cb, capture_output=True)
    if result.returncode:
        detail = (result.stderr or b"").decode(errors="ignore").strip()
        raise RuntimeError(detail or "Could not prepare the edited analysis stream")
    return target

