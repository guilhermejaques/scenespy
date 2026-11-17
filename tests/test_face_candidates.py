import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scenespy.face_engine import FaceDetectionEngine, _FramePrefetcher


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
    def test_apple_mps_is_validated_and_falls_back_before_processing(self):
        created_models = []

        class Model:
            def __init__(self, _path):
                created_models.append(self)

            def predict(self, *_args, **_kwargs):
                raise RuntimeError("unsupported MPS operation")

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: True)))
        engine = FaceDetectionEngine(
            "video.mp4", "output", accel="apple")

        with patch("scenespy.face_engine._ensure_torch", return_value=True), \
                patch("scenespy.face_engine._ensure_yolo", return_value=Model), \
                patch("scenespy.face_engine._ensure_mediapipe", return_value=None), \
                patch("scenespy.face_engine.shared.torch", fake_torch):
            engine._load_deps()

        self.assertEqual(engine.requested_accel, "apple")
        self.assertEqual(engine.accel, "cpu")
        self.assertEqual(engine.device, "cpu")
        self.assertIn("unsupported MPS", engine.acceleration_fallback_reason)
        self.assertEqual(len(created_models), 2)

    def test_cuda_frame_prefetch_preserves_order_and_end_of_stream(self):
        class Capture:
            def __init__(self):
                self.frames = iter([1, 2, 3, 4])

            def read(self):
                try:
                    return True, next(self.frames)
                except StopIteration:
                    return False, None

        prefetcher = _FramePrefetcher(
            Capture(), stop_cb=lambda: False, capacity=2)
        frames = []
        try:
            while True:
                ok, frame = prefetcher.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            prefetcher.close()

        self.assertEqual(frames, [1, 2, 3, 4])
        self.assertGreaterEqual(prefetcher.decode_seconds, 0.0)

    def test_cuda_frame_prefetch_propagates_decoder_errors(self):
        class BrokenCapture:
            def read(self):
                raise RuntimeError("decoder failed")

        prefetcher = _FramePrefetcher(
            BrokenCapture(), stop_cb=lambda: False, capacity=2)
        try:
            with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                prefetcher.read()
        finally:
            prefetcher.close()

    def test_face_json_prioritizes_summary_and_audits_saved_files(self):
        engine = FaceDetectionEngine(
            "video.mp4", "output", keep_rejected=True)
        engine._track_passes_face_gate = lambda _track, _cfg: True
        engine._start_time = 1.0
        engine._end_time = 2.0

        with tempfile.TemporaryDirectory() as folder:
            engine._save_face_track_if_valid(
                sample_track(), folder, min_required=3, cfg={})
            engine._write_face_metadata(folder, fps=24.0, frames_processed=48)
            payload = json.loads(Path(folder, "faces.json").read_text(
                encoding="utf-8"))

        self.assertEqual(next(iter(payload)), "summary")
        self.assertEqual(payload["summary"]["faces_saved"], 1)
        self.assertTrue(payload["summary"]["audit_passed"])
        self.assertTrue(payload["audit"]["saved_files_present"])
        self.assertEqual(payload["faces"][0]["file"], "face_1.png")
        self.assertEqual(payload["process"]["acceleration_requested"], "cpu")
        self.assertEqual(payload["process"]["acceleration_used"], "cpu")
        self.assertFalse(payload["process"]["acceleration_fallback"])
        self.assertEqual(payload["process"]["timings"], {})

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
            self.assertEqual(candidate["file"], "candidate_1.png")

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
        self.assertEqual(engine._face_candidates[0]["file"], "face_1.png")

    def test_finalized_tracks_are_published_once_with_saved_state(self):
        published = []
        engine = FaceDetectionEngine(
            "video.mp4", "output", keep_rejected=True,
            face_callback=published.append)

        with tempfile.TemporaryDirectory() as folder:
            engine._track_passes_face_gate = lambda _track, _cfg: True
            engine._save_face_track_if_valid(
                sample_track(1), folder, min_required=3, cfg={})
            engine._track_passes_face_gate = lambda _track, _cfg: False
            engine._save_face_track_if_valid(
                sample_track(2), folder, min_required=3, cfg={})

        self.assertEqual(len(published), 2)
        self.assertTrue(published[0]["accepted"])
        self.assertFalse(published[1]["accepted"])
        self.assertEqual(published[0]["file"], "face_1.png")
        self.assertEqual(published[1]["file"], "candidate_2.png")


if __name__ == "__main__":
    unittest.main()
