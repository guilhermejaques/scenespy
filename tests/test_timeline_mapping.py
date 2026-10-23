import unittest
from types import SimpleNamespace

from scenespy.frame_map import FrameMap
from scenespy.qt_app import ScenespyApp


class TimelineMappingTests(unittest.TestCase):
    def test_rejected_face_candidate_is_mapped_as_disabled(self):
        engine = SimpleNamespace(
            _output_dir="cache",
            _face_candidates=[{
                "file": "candidate_000001.png",
                "first_frame": 1,
                "last_frame": 8,
                "best_frame": 4,
                "box": [1, 2, 10, 12],
                "crop_box": [0, 0, 14, 16],
                "confidence": 0.6,
                "quality": 0.2,
                "accepted": False,
                "rejection_reason": "Automatic quality filter",
            }],
        )

        events = ScenespyApp._mapped_face_events(
            engine, FrameMap([(100, 200)], 300), "cache")

        self.assertEqual(len(events), 1)
        self.assertFalse(events[0].enabled)
        self.assertFalse(events[0].auto_included)
        self.assertEqual(events[0].best_frame, 104)
        self.assertEqual(events[0].rejection_reason,
                         "Automatic quality filter")

    def test_preanalysis_exclusion_is_removed_from_result_timeline(self):
        mapping = FrameMap([(0, 330)], 108_000)

        cuts, excluded, removed = ScenespyApp._mapped_timeline(
            mapping,
            analysis_boundaries=(),
            mode="faces",
            excluded_ranges=[(330, 108_000)],
            removed_ranges=[],
        )

        self.assertEqual([cut.frame for cut in cuts], [330])
        self.assertEqual(excluded, [])
        self.assertEqual(removed, [2])

    def test_disjoint_enabled_ranges_keep_order_and_hide_the_gap(self):
        mapping = FrameMap([(100, 200), (500, 650)], 1_000)

        cuts, excluded, removed = ScenespyApp._mapped_timeline(
            mapping,
            analysis_boundaries=mapping.join_frames,
            mode="scene",
            excluded_ranges=[(0, 100), (200, 500), (650, 1_000)],
            removed_ranges=[],
        )

        self.assertEqual([cut.frame for cut in cuts], [100, 200, 500, 650])
        self.assertEqual(excluded, [])
        self.assertEqual(removed, [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
