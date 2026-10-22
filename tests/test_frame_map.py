import unittest

from scenespy.frame_map import FrameMap


class FrameMapTests(unittest.TestCase):
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
