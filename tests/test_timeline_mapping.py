import unittest
from types import SimpleNamespace

from scenespy.frame_map import FrameMap
from scenespy.project import CutEvent, FaceEvent, MediaItem
from scenespy.qt_app import (ScenespyApp, TimelineView, clip_preview_candidates,
                             clip_preview_frame,
                             face_event_source_frame)


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

    def test_event_navigation_uses_visible_clip_starts_and_exact_face_frames(self):
        media = MediaItem(
            "video.mp4", duration=10.0, fps=30.0, total_frames=300,
            mode="scene", cuts=[CutEvent(100), CutEvent(200)],
            removed_segments=[2])
        host = self._timeline_host(media)
        host.timeline = SimpleNamespace(
            show_saved_faces=True, show_detected_faces=True)
        self.assertEqual(
            ScenespyApp._navigation_event_frames(host), [200])

        media.mode = "faces"
        media.removed_segments = []
        media.faces = [
            FaceEvent(140, 160, 151, enabled=True),
            FaceEvent(240, 260, 251, enabled=False),
        ]
        host.timeline.show_detected_faces = False
        self.assertEqual(
            ScenespyApp._navigation_event_frames(host), [100, 150, 200])
        host.timeline.show_detected_faces = True
        self.assertEqual(
            ScenespyApp._navigation_event_frames(host), [100, 150, 200, 250])

    def test_next_event_does_not_skip_an_adjacent_frame(self):
        media = MediaItem(
            "video.mp4", duration=10.0, fps=30.0, total_frames=300)
        sought = []
        host = SimpleNamespace(
            current_media=media,
            player=SimpleNamespace(
                position=lambda: round(media.seconds_for_frame(99) * 1000)),
            _navigation_event_frames=lambda: [100, 101],
            seek_player_ms=sought.append,
        )

        ScenespyApp.jump_event(host, 1)

        self.assertEqual(
            sought, [round(media.seconds_for_frame(100) * 1000)])

    def test_playback_face_boxes_use_short_cached_candidate_windows(self):
        face = FaceEvent(
            90, 110, 101, box=[10, 20, 50, 70],
            review_frames=[{
                "frame": 101, "box": [10, 20, 50, 70],
                "crop_box": [5, 10, 55, 80],
            }])
        media = MediaItem(
            "video.mp4", duration=10.0, fps=30.0, total_frames=300,
            mode="faces", faces=[face])
        host = SimpleNamespace(
            current_media=media,
            timeline=SimpleNamespace(
                show_saved_faces=True, show_detected_faces=True))

        index = ScenespyApp._build_playback_face_index(host)

        self.assertEqual(index[100][1], 100)
        self.assertEqual(index[100][2], [10, 20, 50, 70])
        self.assertIn(97, index)
        self.assertIn(103, index)
        self.assertNotIn(96, index)
        self.assertNotIn(104, index)

    def test_interval_preview_boundaries_match_frame_rounding(self):
        self.assertEqual(
            ScenespyApp._interval_analysis_boundaries(301, 30.0, 3),
            [90, 180, 270])

    def test_clip_selection_requires_at_least_one_cut(self):
        media = MediaItem(
            "video.mp4", duration=10.0, fps=30.0, total_frames=300)
        host = SimpleNamespace(media=media)
        self.assertFalse(TimelineView.clip_selection_available(host))
        media.cuts.append(CutEvent(150))
        self.assertTrue(TimelineView.clip_selection_available(host))

    def test_cut_uses_the_exact_displayed_source_frame_and_reindexes_segments(self):
        media = MediaItem(
            "video.mp4", duration=10.2, fps=29.97, total_frames=300,
            cuts=[CutEvent(100), CutEvent(200)],
            excluded_segments=[2], removed_segments=[3])
        timeline = SimpleNamespace(rebuild=lambda: None)
        host = SimpleNamespace(
            current_media=media,
            selected_segment=None,
            timeline=timeline,
            _current_media_is_processing=lambda: False,
            _mark_current_unexported=lambda: None,
            _refresh_media_metrics=lambda: None,
            _refresh_action_button=lambda: None,
            _refresh_cut_action_state=lambda: None,
        )
        host._can_add_cut_at = lambda seconds: ScenespyApp._can_add_cut_at(
            host, seconds)

        displayed_seconds = media.seconds_for_frame(150)
        self.assertFalse(host._can_add_cut_at(media.seconds_for_frame(100)))
        self.assertTrue(host._can_add_cut_at(displayed_seconds))
        ScenespyApp.add_cut_at(host, displayed_seconds)
        self.assertFalse(host._can_add_cut_at(displayed_seconds))

        self.assertEqual([cut.frame for cut in media.cuts], [100, 150, 200])
        self.assertEqual(media.excluded_segments, [2, 3])
        self.assertEqual(media.removed_segments, [4])
        self.assertAlmostEqual(
            media.seconds_for_frame(media.frame_for_seconds(displayed_seconds)),
            displayed_seconds, places=9)

    def test_segment_frame_buttons_use_exact_first_and_last_frames(self):
        media = MediaItem(
            "video.mp4", duration=10.2, fps=29.97, total_frames=300)
        shown = []
        host = SimpleNamespace(
            current_media=media,
            selected_segment=(1, media.seconds_for_frame(100),
                              media.seconds_for_frame(200)),
            _show_segment_still=shown.append,
        )

        ScenespyApp.step_face_frame(host, -1)
        ScenespyApp.step_face_frame(host, 1)

        self.assertEqual(shown, [100, 199])

    def test_selection_preview_reuses_timeline_thumbnail_source_frame(self):
        media = MediaItem(
            "video.mp4", duration=10.2, fps=29.97, total_frames=300)
        requested = []
        shown = []
        timeline = SimpleNamespace(
            clip_thumbnail_source_frame=lambda start, end: (
                requested.append((start, end)) or 102))
        preview = SimpleNamespace(
            clear_source=lambda: None, hide=lambda: None)
        host = SimpleNamespace(
            current_media=media, timeline=timeline,
            selection_preview=preview, _show_segment_still=shown.append)

        ScenespyApp._show_segment_thumbnail(
            host, media.seconds_for_frame(100), media.seconds_for_frame(200))

        self.assertEqual(requested, [(100, 200)])
        self.assertEqual(shown, [102])

    def test_face_indicator_uses_zero_based_media_frame(self):
        event = SimpleNamespace(best_frame=1)
        self.assertEqual(face_event_source_frame(event), 0)
        event.best_frame = 105
        self.assertEqual(face_event_source_frame(event), 104)

    def test_clip_indicator_uses_the_same_first_frame_as_selection(self):
        self.assertEqual(clip_preview_frame(0, 90), 0)
        self.assertEqual(clip_preview_frame(150, 275), 150)
        self.assertEqual(list(clip_preview_candidates(150, 154)),
                         [150, 151, 152, 153])
        candidates = list(clip_preview_candidates(150, 500))
        self.assertEqual(candidates[0], 150)
        self.assertEqual(candidates[-1], 179)
        self.assertNotIn(500, candidates)

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
