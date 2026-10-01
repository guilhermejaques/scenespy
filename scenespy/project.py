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


@dataclass
class FaceEvent:
    start_frame: int
    end_frame: int
    best_frame: int
    crop_file: str = ""
    confidence: float = 0.0
    quality: float = 0.0
    box: list[int] = field(default_factory=list)
    enabled: bool = True
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
    cuts: list[CutEvent] = field(default_factory=list)
    faces: list[FaceEvent] = field(default_factory=list)
    media_id: str = field(default_factory=lambda: uuid.uuid4().hex)

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
            item["cuts"] = [CutEvent(**event) for event in item.get("cuts", [])]
            item["faces"] = [FaceEvent(**event) for event in item.get("faces", [])]
            media.append(MediaItem(**item))
        return cls(
            media=media,
            output_folder=raw.get("output_folder", ""),
            project_path=str(path),
            version=int(raw.get("version", PROJECT_VERSION)),
        )
