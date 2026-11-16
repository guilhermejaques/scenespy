import unittest
from types import SimpleNamespace

from scenespy.frame_map import FrameMap
from scenespy.project import CutEvent, FaceEvent, MediaItem
from scenespy.qt_app import (ScenespyApp, TimelineView, clip_preview_candidates,
                             clip_preview_frame,
                             face_event_source_frame)


class TimelineMappingTests(unittest.TestCase):
    def test_timeline_invalidates_playhead_before_scene_destroys_items(self):
        observations = []
        host = SimpleNamespace(playhead=object())
        host.scene = lambda: SimpleNamespace(
            clear=lambda: observations.append(host.playhead))

        TimelineView._clear_scene_items(host)

        self.assertEqual(observations, [None])
        self.assertIsNone(host.playhead)

    def test_processing_lock_visibly_dims_and_restores_timeline(self):
        class Effect:
            opacity = None

            def setOpacity(self, value):
                self.opacity = value

        effect = Effect()
        host = SimpleNamespace(timeline_lock_effect=effect)

        ScenespyApp._set_timeline_locked_visual(host, True)
        self.assertEqual(effect.opacity, .58)
        ScenespyApp._set_timeline_locked_visual(host, False)
        self.assertEqual(effect.opacity, 1.0)

    def test_live_queue_segments_use_full_rebuild_without_committing(self):
        original_cut = CutEvent(10)
        media = MediaItem(
            "video.mp4", duration=4, fps=25, total_frames=100,
            cuts=[original_cut], excluded_segments=[1], removed_segments=[2],
            analyzed=False)
        live_cuts = [CutEvent(30), CutEvent(70)]
        rendered = []
        timeline = SimpleNamespace(media=media)

        def rebuild():
            rendered.append((
                [cut.frame for cut in media.cuts],
                list(media.removed_segments),
                list(media.excluded_segments),
                media.analyzed,
            ))

        timeline.rebuild = rebuild
        TimelineView.show_live_segments(
            timeline, media, live_cuts, [2], load_thumbnails=True)

        self.assertEqual(rendered, [([30, 70], [2], [], True)])
        self.assertEqual(media.cuts, [original_cut])
        self.assertEqual(media.excluded_segments, [1])
        self.assertEqual(media.removed_segments, [2])
        self.assertFalse(media.analyzed)
        self.assertEqual(timeline._live_segment_media_id, media.media_id)

    def test_final_live_queue_segments_enable_indicators_and_thumbnails(self):
        media = MediaItem(
            "video.mp4", duration=4, fps=25, total_frames=100,
            analyzed=False)
        calls = []
        task = SimpleNamespace(live_segments={})
        host = SimpleNamespace(
            task=task,
            current_media=media,
            timeline=SimpleNamespace(
                show_live_segments=lambda *args, **kwargs:
                calls.append((args, kwargs))))
        cuts = [CutEvent(25), CutEvent(50)]

        ScenespyApp._live_segments_found(
            host, media, (cuts, [], True))

        self.assertTrue(calls[0][1]["load_thumbnails"])
        self.assertEqual(task.live_segments[media.media_id],
                         (cuts, [], True))

    def test_progressive_live_segments_defer_thumbnails(self):
        media = MediaItem(
            "video.mp4", duration=4, fps=25, total_frames=100,
            analyzed=False)
        calls = []
        host = SimpleNamespace(
            task=SimpleNamespace(live_segments={}),
            current_media=media,
            timeline=SimpleNamespace(
                show_live_segments=lambda *args, **kwargs:
                calls.append((args, kwargs))))

        ScenespyApp._live_segments_found(
            host, media, ([CutEvent(25)], [], False))

        self.assertFalse(calls[0][1]["load_thumbnails"])

    def test_queue_transport_uses_uncommitted_live_cut_layout(self):
        media = MediaItem(
            "video.mp4", duration=4, fps=25, total_frames=100,
            cuts=[], analyzed=False)
        live_cuts = [CutEvent(25), CutEvent(50), CutEvent(75)]
        host = SimpleNamespace(
            current_media=media,
            processing_media=media,
            task=SimpleNamespace(live_segments={
                media.media_id: (live_cuts, [2], True)}))

        ranges = ScenespyApp._timeline_ranges(host, media)

        self.assertEqual(ranges, [(0, 25), (50, 75), (75, 100)])

    def test_queue_cut_preview_rewinds_transport_before_export(self):
        media = MediaItem(
            "video.mp4", duration=4, fps=25, total_frames=100)
        events = []
        ready = SimpleNamespace(set=lambda: events.append("ready"))
        task = SimpleNamespace(
            cut_preview_ready=ready, media_progress={})
        animation = SimpleNamespace(stop=lambda: events.append("stop"))
        label = SimpleNamespace(
            setText=lambda text: events.append(("label", text)))
        host = SimpleNamespace(
            task=task,
            batch_active=True,
            processing_media=media,
            processing_position_animation=animation,
            progress_label=label,
            _processing_preview_selected=lambda: True,
            _set_progress_value=lambda value, immediate=False:
            events.append(("progress", value, immediate)),
            _sync_processing_position=lambda value:
            events.append(("position", value)))

        ScenespyApp._prepare_queue_cut_preview(host, media)

        self.assertIn(("position", 0.0), events)
        self.assertIn(("progress", 0, True), events)
        self.assertEqual(task.media_progress[media.media_id],
                         (0, "Cutting clips"))
        self.assertEqual(events[-1], "ready")

    def test_stopping_clip_preview_preserves_timed_duration_label(self):
        class Status:
            def __init__(self):
                self.message = "Previewing clip · 00:01.000 / 00:03.000"
                self.timeout = None

            def currentMessage(self):
                return self.message

            def showMessage(self, message, timeout=0):
                self.message, self.timeout = message, timeout

            def clearMessage(self):
                self.message = ""

        status = Status()
        cleared = []
        host = SimpleNamespace(
            preview_end_ms=3000,
            preview_start_ms=1000,
            preview_armed=True,
            _segment_preview_status_active=True,
            _segment_preview_last_status_ms=2000,
            timeline=SimpleNamespace(
                clear_clip_duration=lambda: cleared.append(True)),
            statusBar=lambda: status,
        )

        ScenespyApp._stop_segment_preview(host, "Clip preview stopped")

        self.assertIsNone(host.preview_end_ms)
        self.assertIsNone(host.preview_start_ms)
        self.assertFalse(host.preview_armed)
        self.assertFalse(host._segment_preview_status_active)
        self.assertEqual(status.message, "Clip preview stopped")
        self.assertEqual(status.timeout, 3000)
        self.assertEqual(cleared, [])

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
        timeline_end = media.seconds_for_frame(media.total_frames)
        final_duration = timeline_end - media.seconds_for_frame(200)
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
            timeline_end, places=9)

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
        timeline = SimpleNamespace(
            rebuild=lambda: None, set_position=lambda _seconds: None)
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

    def test_cut_uses_playhead_frame_while_player_seek_is_still_behind(self):
        media = MediaItem(
            "video.mp4", duration=10.0, fps=30.0, total_frames=300)
        playhead = SimpleNamespace(x=lambda: 1500.0)
        rebuilt_at = []
        positioned_at = []
        timeline = SimpleNamespace(
            playhead=playhead,
            frame_for_x=lambda _x: 150,
            current_seconds=149.6 / 30.0,
            rebuild=lambda: rebuilt_at.append(timeline.current_seconds),
            set_position=positioned_at.append,
        )
        host = SimpleNamespace(
            current_media=media,
            selected_segment=None,
            timeline=timeline,
            player=SimpleNamespace(position=lambda: 4967),
            _current_media_is_processing=lambda: False,
            _mark_current_unexported=lambda: None,
            _refresh_media_metrics=lambda: None,
            _refresh_action_button=lambda: None,
            _refresh_cut_action_state=lambda: None,
        )
        host._playhead_cut_frame = lambda: ScenespyApp._playhead_cut_frame(host)
        host._can_add_cut_frame = lambda frame: ScenespyApp._can_add_cut_frame(
            host, frame)
        host._add_cut_frame = lambda frame: ScenespyApp._add_cut_frame(host, frame)

        ScenespyApp.add_cut(host)

        self.assertEqual([cut.frame for cut in media.cuts], [150])
        self.assertEqual(rebuilt_at, [media.seconds_for_frame(150)])
        self.assertEqual(positioned_at, [media.seconds_for_frame(150)])

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
