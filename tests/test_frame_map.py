import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scenespy.frame_map import FrameMap, create_playback_proxy


class FrameMapTests(unittest.TestCase):
    def test_playback_proxy_is_small_seek_friendly_and_frame_limited(self):
        with TemporaryDirectory() as folder, \
                patch("scenespy.frame_map.ffmpeg_path", return_value="ffmpeg"), \
                patch("scenespy.frame_map.run_hidden_cancelable",
                      return_value=SimpleNamespace(returncode=0, stderr=b"")) as run:
            target = f"{folder}/proxy.mp4"
            create_playback_proxy("source.mp4", target, 29.97, 300)

        command = run.call_args.args[0]
        self.assertIn(
            "scale=960:540:force_original_aspect_ratio=decrease:force_divisible_by=2",
            command)
        self.assertEqual(command[command.index("-frames:v") + 1], "300")
        self.assertEqual(command[command.index("-g") + 1], "12")
        self.assertIn("0:a:0?", command)

    def test_maps_concatenated_frames_to_original_ranges(self):
        mapping = FrameMap([(0, 100), (200, 260)], 300)
        self.assertEqual(mapping.analysis_frames, 160)
        self.assertEqual(mapping.to_source(0), 0)
        self.assertEqual(mapping.to_source(99), 99)
        self.assertEqual(mapping.to_source(100), 200)
        self.assertEqual(mapping.to_source(159), 259)
        self.assertEqual(mapping.to_source(160, allow_endpoint=True), 260)

    def test_rebuilds_original_timeline_and_excluded_gap(self):
        mapping = FrameMap([(0, 100), (200, 300)], 300)
        boundaries, excluded = mapping.rebuild_segments([0, 40, 100, 150, 200])
        self.assertEqual(boundaries, [0, 40, 100, 200, 250, 300])
        self.assertEqual(excluded, [3])

    def test_from_segments_uses_one_based_sequence_indexes(self):
        mapping = FrameMap.from_segments(
            [(0, 30), (30, 60), (60, 90)], [2], [], 90)
        self.assertEqual([(item.start, item.end) for item in mapping.ranges],
                         [(0, 30), (60, 90)])
        self.assertFalse(mapping.is_identity)

    def test_fractional_fps_conversion_is_frame_based(self):
        mapping = FrameMap([(0, 300), (600, 900)], 900)
        fps = 30000 / 1001
        self.assertAlmostEqual(
            mapping.seconds_to_source(300 / fps, fps), 600 / fps, places=9)


if __name__ == "__main__":
    unittest.main()
