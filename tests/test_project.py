import tempfile
import unittest
from pathlib import Path

from scenespy.project import CutEvent, FaceEvent, MediaItem, ScenespyProject
from scenespy.qt_app import (final_face_crop_box, parse_timecode, scrub_timecode,
                             timecode, transport_timecode)


class ProjectModelTests(unittest.TestCase):
    def test_editor_timecodes_are_parseable(self):
        self.assertEqual(parse_timecode("01:02:03.500"), 3723.5)
        self.assertEqual(parse_timecode("02:10"), 130.0)
        self.assertIsNone(parse_timecode("not-a-time"))
        self.assertEqual(transport_timecode(3723.987), "01:02:03")

    def test_editor_timecodes_hide_unused_hours_consistently(self):
        self.assertEqual(transport_timecode(65, 3599), "01:05")
        self.assertEqual(timecode(65.25, 3599), "01:05.250")
        self.assertEqual(transport_timecode(65, 3600), "00:01:05")
        self.assertEqual(timecode(3599.999, 3599.999), "59:59.999")
        self.assertEqual(timecode(3599.9996, 3599.9996), "01:00:00.000")

    def test_scrub_timecodes_cover_very_short_and_long_media(self):
        self.assertEqual(scrub_timecode(.125, .8), "00:00.125")
        self.assertEqual(scrub_timecode(5, 30), "00:05.000")
        self.assertEqual(scrub_timecode(3661, 360_000), "01:01:01")
        self.assertEqual(scrub_timecode(360_000, 360_000), "100:00:00")

    def test_final_face_crop_box_matches_export_framing(self):
        self.assertEqual(final_face_crop_box([100, 100, 200, 200], 1920, 1080),
                         [65, 58, 235, 248])
        self.assertEqual(final_face_crop_box([0, 0, 100, 100], 200, 200),
                         [0, 0, 170, 190])

    def test_cuts_are_sorted_deduplicated_and_bounded(self):
        media = MediaItem("video.mp4", 10, 30, 300, cuts=[
            CutEvent(200), CutEvent(100), CutEvent(100), CutEvent(999), CutEvent(-4)
        ])
        media.normalize_cuts()
        self.assertEqual([cut.frame for cut in media.cuts], [1, 100, 200, 299])
        self.assertEqual(media.segments(), [(0, 1), (1, 100), (100, 200), (200, 299), (299, 300)])

    def test_frame_and_player_time_conversion_is_precise_and_bounded(self):
        media = MediaItem("video.mp4", 10.2, 29.97, 300)
        self.assertAlmostEqual(media.seconds_for_frame(150), 150 / 29.97, places=9)
        self.assertEqual(media.frame_for_seconds(150 / 29.97), 150)
        self.assertEqual(media.frame_for_seconds(-5), 0)
        self.assertEqual(media.frame_for_seconds(999), 300)
        self.assertEqual(media.seconds_for_frame(300), 10.2)

    def test_project_round_trip_preserves_editor_events(self):
        project = ScenespyProject(media=[MediaItem(
            "video.mp4", 5, 24, 120,
            analyzed=True, exported=True,
            cuts=[CutEvent(48, "manual", original_frame=42)],
            faces=[FaceEvent(
                12, 30, 20, "face.png", .9, .8, [1, 2, 3, 4], [0, 0, 8, 10],
                review_frames=[
                    {"frame": 19, "box": [1, 2, 3, 4], "crop_box": [0, 0, 8, 10]},
                    {"frame": 20, "box": [1, 2, 3, 4], "crop_box": [0, 0, 8, 10]},
                ],
                original_best_frame=20,
                original_box=[1, 2, 3, 4],
                original_crop_box=[0, 0, 8, 10],
                enabled=False,
                rejection_reason="Automatic quality filter",
                auto_included=False,
            )],
            excluded_segments=[2],
            removed_segments=[3],
        )])
        with tempfile.TemporaryDirectory() as folder:
            path = project.save(Path(folder, "review"))
            loaded = ScenespyProject.load(path)
        self.assertTrue(path.endswith(".scenespy"))
        self.assertEqual(loaded.media[0].cuts[0].frame, 48)
        self.assertEqual(loaded.media[0].cuts[0].original_frame, 42)
        self.assertEqual(loaded.media[0].faces[0].box, [1, 2, 3, 4])
        self.assertEqual(loaded.media[0].faces[0].crop_box, [0, 0, 8, 10])
        self.assertEqual(len(loaded.media[0].faces[0].review_frames), 2)
        self.assertEqual(loaded.media[0].faces[0].original_best_frame, 20)
        self.assertEqual(loaded.media[0].faces[0].original_crop_box, [0, 0, 8, 10])
        self.assertFalse(loaded.media[0].faces[0].enabled)
        self.assertFalse(loaded.media[0].faces[0].auto_included)
        self.assertEqual(loaded.media[0].faces[0].rejection_reason,
                         "Automatic quality filter")
        self.assertEqual(loaded.media[0].excluded_segments, [2])
        self.assertEqual(loaded.media[0].removed_segments, [3])
        self.assertTrue(loaded.media[0].analyzed)
        self.assertTrue(loaded.media[0].exported)


if __name__ == "__main__":
    unittest.main()
