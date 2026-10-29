import unittest

from scenespy.scene_engine import SceneEngine


class ScenePreviewTests(unittest.TestCase):
    def test_scene_preview_ratio_matches_the_displayed_analysis_progress(self):
        engine = SceneEngine("video.mp4", "output", {})

        engine._analysis_ratio = 0.32
        engine._write_analysis_status("finding transitions", 0.58)

        self.assertEqual(engine._get_progress_ratio(), 0.58)

    def test_analysis_progress_never_moves_preview_backwards(self):
        engine = SceneEngine("video.mp4", "output", {})

        engine._analysis_ratio = 0.70
        engine._write_analysis_status("stale update", 0.58)

        self.assertEqual(engine._get_progress_ratio(), 0.70)

    def test_cut_preview_follows_attempted_progress_after_a_failed_cut(self):
        engine = SceneEngine("video.mp4", "output", {})
        engine.total = 4
        engine.done = 1
        engine.completed_attempts = 2

        self.assertEqual(engine._get_progress_ratio(), 0.5)

    def test_export_preview_uses_ordered_sequence_edge_frames(self):
        engine = SceneEngine(
            "video.mp4", "output", {}, preview_frames=[10, 19, 30, 49])
        engine._get_video_fps = lambda: 10.0

        self.assertEqual(engine._preview_time_for_ratio(0.0), 1.0)
        self.assertEqual(engine._preview_time_for_ratio(0.5), 3.0)
        self.assertEqual(engine._preview_time_for_ratio(1.0), 4.9)

    def test_final_analysis_scenes_are_published_as_an_isolated_snapshot(self):
        published = []
        engine = SceneEngine(
            "video.mp4", "output", {}, analysis_callback=published.append)
        scenes = [(0, 90), (90, 210), (210, 300)]

        engine._publish_analysis_scenes(scenes)
        scenes.append((300, 330))

        self.assertEqual(published, [[(0, 90), (90, 210), (210, 300)]])

    def test_stopped_engine_does_not_publish_preview_results(self):
        published = []
        engine = SceneEngine(
            "video.mp4", "output", {}, analysis_callback=published.append)
        engine._stop = True

        engine._publish_analysis_scenes([(0, 90)])

        self.assertEqual(published, [])

    def test_preview_scenes_use_a_snapshot_too(self):
        published = []
        engine = SceneEngine(
            "video.mp4", "output", {},
            preview_analysis_callback=published.append)
        scenes = [(0, 90), (90, 210)]

        engine._publish_preview_scenes(scenes)
        scenes.clear()

        self.assertEqual(published, [[(0, 90), (90, 210)]])

    def test_preview_boundaries_are_normalized_and_deduplicated(self):
        published = []
        engine = SceneEngine(
            "video.mp4", "output", {},
            preview_analysis_callback=published.append)

        engine._publish_preview_boundaries([90, 90, 210], 300, fps=30)
        engine._publish_preview_boundaries([210, 90], 300, fps=30)

        self.assertEqual(
            published, [[(0, 90), (90, 210), (210, 300)]])


if __name__ == "__main__":
    unittest.main()
