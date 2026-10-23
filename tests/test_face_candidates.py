import tempfile
import unittest
from pathlib import Path

import numpy as np

from scenespy.face_engine import FaceDetectionEngine


def sample_track(track_id=1, frames=5):
    return {
        "id": track_id,
        "frames": frames,
        "valid": 0,
        "face": np.full((16, 16, 3), 127, dtype=np.uint8),
        "box": (2, 2, 12, 12),
        "best_box": (2, 2, 12, 12),
        "best_crop_box": (0, 0, 16, 16),
        "first_frame": 1,
        "last_frame": frames,
        "best_frame": 2,
        "confidence": 0.5,
        "quality": 0.2,
        "score": 4.0,
        "first_seen": 0.0,
        "last_seen": 0.2,
        "best_time": 0.1,
        "review_candidates": [],
    }


class FaceCandidateTests(unittest.TestCase):
    def test_rejected_track_keeps_one_lightweight_review_crop(self):
        engine = FaceDetectionEngine(
            "video.mp4", "output", keep_rejected=True)
        engine._track_passes_face_gate = lambda _track, _cfg: False

        with tempfile.TemporaryDirectory() as folder:
            engine._save_face_track_if_valid(
                sample_track(), folder, min_required=3, cfg={})
            candidate = engine._face_candidates[0]
            crop = Path(folder, candidate["file"])
            self.assertTrue(crop.is_file())
            self.assertEqual(crop.suffix, ".png")

        self.assertFalse(candidate["accepted"])
        self.assertEqual(candidate["rejection_reason"],
                         "Automatic quality filter")
        self.assertEqual(engine.done, 0)
        self.assertEqual(engine._saved_faces, [])

    def test_public_pipeline_does_not_retain_rejected_tracks_by_default(self):
        engine = FaceDetectionEngine("video.mp4", "output")
        engine._track_passes_face_gate = lambda _track, _cfg: False

        with tempfile.TemporaryDirectory() as folder:
            engine._save_face_track_if_valid(
                sample_track(), folder, min_required=3, cfg={})

        self.assertEqual(engine._face_candidates, [])

    def test_accepted_track_is_available_to_saved_and_review_lists(self):
        engine = FaceDetectionEngine(
            "video.mp4", "output", keep_rejected=True)
        engine._track_passes_face_gate = lambda _track, _cfg: True

        with tempfile.TemporaryDirectory() as folder:
            engine._save_face_track_if_valid(
                sample_track(), folder, min_required=3, cfg={})

        self.assertEqual(engine.done, 1)
        self.assertEqual(len(engine._saved_faces), 1)
        self.assertEqual(len(engine._face_candidates), 1)
        self.assertTrue(engine._face_candidates[0]["accepted"])


if __name__ == "__main__":
    unittest.main()
