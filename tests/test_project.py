import tempfile
import unittest
from pathlib import Path

from scenespy.project import CutEvent, FaceEvent, MediaItem, ScenespyProject
from scenespy.qt_app import parse_timecode, transport_timecode


class ProjectModelTests(unittest.TestCase):
    def test_editor_timecodes_are_parseable(self):
        self.assertEqual(parse_timecode("01:02:03.500"), 3723.5)
        self.assertEqual(parse_timecode("02:10"), 130.0)
        self.assertIsNone(parse_timecode("not-a-time"))
        self.assertEqual(transport_timecode(3723.987), "01:02:03")

    def test_cuts_are_sorted_deduplicated_and_bounded(self):
        media = MediaItem("video.mp4", 10, 30, 300, cuts=[
            CutEvent(200), CutEvent(100), CutEvent(100), CutEvent(999), CutEvent(-4)
        ])
        media.normalize_cuts()
        self.assertEqual([cut.frame for cut in media.cuts], [1, 100, 200, 299])
        self.assertEqual(media.segments(), [(0, 1), (1, 100), (100, 200), (200, 299), (299, 300)])

    def test_project_round_trip_preserves_editor_events(self):
        project = ScenespyProject(media=[MediaItem(
            "video.mp4", 5, 24, 120,
            cuts=[CutEvent(48, "manual")],
            faces=[FaceEvent(12, 30, 20, "face.png", .9, .8, [1, 2, 3, 4])],
        )])
        with tempfile.TemporaryDirectory() as folder:
            path = project.save(Path(folder, "review"))
            loaded = ScenespyProject.load(path)
        self.assertTrue(path.endswith(".scenespy"))
        self.assertEqual(loaded.media[0].cuts[0].frame, 48)
        self.assertEqual(loaded.media[0].faces[0].box, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
