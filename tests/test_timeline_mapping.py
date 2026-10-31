import unittest
from types import SimpleNamespace

from scenespy.frame_map import FrameMap
from scenespy.project import CutEvent, MediaItem
from scenespy.qt_app import ScenespyApp, face_event_source_frame


class TimelineMappingTests(unittest.TestCase):
    @staticmethod
    def _timeline_host(media):
        host = SimpleNamespace(current_media=media)
        host._timeline_ranges = lambda selected=None: ScenespyApp._timeline_ranges(
            host, selected)
        host._timeline_duration = lambda selected=None: ScenespyApp._timeline_duration(
            host, selected)
        return host

    def test_transport_uses_real_media_duration_and_skips_removed_gap(self):
        media = MediaItem(
            "video.mp4", duration=10.2, fps=29.97, total_frames=300,
            cuts=[CutEvent(100), CutEvent(200)], removed_segments=[2])
        host = self._timeline_host(media)

        first_duration = media.seconds_for_frame(100)
        final_duration = media.duration - media.seconds_for_frame(200)
        expected_duration = first_duration + final_duration
        self.assertAlmostEqual(
            ScenespyApp._timeline_duration(host), expected_duration, places=9)
        self.assertAlmostEqual(
            ScenespyApp._source_to_timeline_seconds(host, media.duration),
            expected_duration, places=9)

        # The exact ripple join resolves to the first frame after the removed
        # range, never to a timestamp inside that range.
        self.assertAlmostEqual(
            ScenespyApp._timeline_to_source_seconds(host, first_duration),
            media.seconds_for_frame(200), places=9)
        self.assertAlmostEqual(
            ScenespyApp._timeline_to_source_seconds(host, expected_duration),
            media.duration, places=9)

    def test_visible_transport_time_round_trips_away_from_ripple_join(self):
        media = MediaItem(
            "video.mp4", duration=20.0, fps=30.0, total_frames=600,
            cuts=[CutEvent(180), CutEvent(360)], removed_segments=[2])
        host = self._timeline_host(media)
        source_seconds = 15.0

        timeline_seconds = ScenespyApp._source_to_timeline_seconds(
            host, source_seconds)
        restored = ScenespyApp._timeline_to_source_seconds(
            host, timeline_seconds)
        self.assertAlmostEqual(restored, source_seconds, places=9)

    def test_interval_preview_boundaries_match_frame_rounding(self):
        self.assertEqual(
            ScenespyApp._interval_analysis_boundaries(301, 30.0, 3),
            [90, 180, 270])

    def test_face_indicator_uses_zero_based_media_frame(self):
        event = SimpleNamespace(best_frame=1)
        self.assertEqual(face_event_source_frame(event), 0)
        event.best_frame = 105
        self.assertEqual(face_event_source_frame(event), 104)

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

        live_event = ScenespyApp._mapped_face_event(
            engine._face_candidates[0], FrameMap([(100, 200)], 300),
            "cache", engine._output_dir)
        self.assertEqual(live_event.best_frame, events[0].best_frame)
        self.assertEqual(live_event.crop_file, events[0].crop_file)
        self.assertEqual(live_event.enabled, events[0].enabled)

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
