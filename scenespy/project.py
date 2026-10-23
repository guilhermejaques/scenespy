"""Non-destructive project model shared by the editor UI and exporters."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


PROJECT_VERSION = 1


@dataclass
class CutEvent:
    frame: int
    source: str = "scene"
    confidence: float | None = None
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    original_frame: int = 0


@dataclass
class FaceEvent:
    start_frame: int
    end_frame: int
    best_frame: int
    crop_file: str = ""
    confidence: float = 0.0
    quality: float = 0.0
    box: list[int] = field(default_factory=list)
    crop_box: list[int] = field(default_factory=list)
    enabled: bool = True
    review_frames: list[dict] = field(default_factory=list)
    original_best_frame: int = 0
    original_box: list[int] = field(default_factory=list)
    original_crop_box: list[int] = field(default_factory=list)
    rejection_reason: str = ""
    auto_included: bool = True
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class MediaItem:
    path: str
    duration: float
    fps: float
    total_frames: int
    width: int = 0
    height: int = 0
    mode: str = "scene"
    sensitivity: str = "Normal"
    analyzed: bool = False
    exported: bool = False
    cuts: list[CutEvent] = field(default_factory=list)
    faces: list[FaceEvent] = field(default_factory=list)
    excluded_segments: list[int] = field(default_factory=list)
    removed_segments: list[int] = field(default_factory=list)
    media_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def seconds_for_frame(self, frame: int | float) -> float:
        """Map a frame onto the player's canonical media duration."""
        if float(frame) >= self.total_frames:
            return max(0.0, self.duration)
        return max(0.0, min(self.duration, float(frame) / max(self.fps, 0.001)))

    def frame_for_seconds(self, seconds: int | float) -> int:
        """Map player time back onto the frame-based project model."""
        if self.duration <= 0 or self.fps <= 0:
            return 0
        seconds = max(0.0, min(self.duration, float(seconds)))
        return max(0, min(self.total_frames, round(seconds * self.fps)))

    def normalize_cuts(self) -> None:
        last_frame = max(0, self.total_frames - 1)
        unique = {}
        for event in self.cuts:
            event.frame = max(1, min(int(event.frame), last_frame))
            unique[event.frame] = event
        self.cuts = [unique[frame] for frame in sorted(unique) if frame < self.total_frames]

    def segments(self) -> list[tuple[int, int]]:
        self.normalize_cuts()
        boundaries = [0, *(event.frame for event in self.cuts), self.total_frames]
        return [(start, end) for start, end in zip(boundaries, boundaries[1:]) if end > start]


@dataclass
class ScenespyProject:
    media: list[MediaItem] = field(default_factory=list)
    output_folder: str = ""
    project_path: str = ""
    version: int = PROJECT_VERSION

    def save(self, path: str | os.PathLike | None = None) -> str:
        target = str(path or self.project_path)
        if not target:
            raise ValueError("A project path is required")
        if not target.lower().endswith(".scenespy"):
            target += ".scenespy"
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["project_path"] = target
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.project_path = target
        return target

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ScenespyProject":
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        media = []
        for item in raw.get("media", []):
            item = dict(item)
            if "analyzed" not in item:
                item["analyzed"] = bool(item.get("cuts") or item.get("faces"))
            if "exported" not in item:
                item["exported"] = False
            item["cuts"] = [CutEvent(**event) for event in item.get("cuts", [])]
            item["faces"] = [FaceEvent(**event) for event in item.get("faces", [])]
            media.append(MediaItem(**item))
        return cls(
            media=media,
            output_folder=raw.get("output_folder", ""),
            project_path=str(path),
            version=int(raw.get("version", PROJECT_VERSION)),
        )
