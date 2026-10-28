import unittest

from scenespy.scene_engine import SceneEngine


class ScenePreviewTests(unittest.TestCase):
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
