import os
import shutil
import tempfile
import unittest
from pathlib import Path


RUN_INTEGRATION = os.environ.get("SCENESPY_RUN_INTEGRATION") == "1"


@unittest.skipUnless(RUN_INTEGRATION, "set SCENESPY_RUN_INTEGRATION=1 to run video smoke tests")
class VideoPipelineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise unittest.SkipTest("FFmpeg and FFprobe are required")

        import cv2
        import numpy as np

        cls._temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temp.name)
        cls.video = cls.root / "solid-scenes.mp4"
        writer = cv2.VideoWriter(
            str(cls.video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            12.0,
            (320, 240),
        )
        if not writer.isOpened():
            cls._temp.cleanup()
            raise RuntimeError("OpenCV could not create the integration video")
        try:
            for color in ((0, 0, 255), (255, 0, 0), (0, 255, 0)):
                frame = np.full((240, 320, 3), color, dtype=np.uint8)
                for _ in range(24):
                    writer.write(frame)
        finally:
            writer.release()

    @classmethod
    def tearDownClass(cls):
        cls._temp.cleanup()

    def test_scene_and_interval_outputs(self):
        from scenespy.api import detect_scenes, split_video

        scene = detect_scenes(
            self.video, self.root / "scenes", sensitivity="High", verbose=False
        )
        interval = split_video(
            self.video, self.root / "intervals", interval=2, verbose=False
        )

        self.assertTrue(scene["success"], scene)
        self.assertGreaterEqual(scene["saved"], 2, scene)
        self.assertEqual(scene["failed"], 0, scene)
        self.assertTrue(interval["success"], interval)
        self.assertEqual(interval["saved"], 3, interval)
        self.assertEqual(interval["failed"], 0, interval)

    def test_face_pipeline_loads_and_completes(self):
        from scenespy.api import extract_faces

        result = extract_faces(
            self.video, self.root / "faces", sensitivity="High", verbose=False
        )

        self.assertTrue(result["success"], result)
        self.assertEqual(result["failed"], 0, result)

