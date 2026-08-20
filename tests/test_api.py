import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scenespy import api
from scenespy.shared import format_status_lines


class ApiValidationTests(unittest.TestCase):
    def test_public_import_without_gui_modules(self):
        code = (
            "import sys; "
            "sys.modules['tkinter'] = None; "
            "sys.modules['customtkinter'] = None; "
            "from scenespy import detect_scenes, split_video, extract_faces; "
            "print(detect_scenes.__module__, split_video.__module__, extract_faces.__module__)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "scenespy.api scenespy.api scenespy.api")

    def test_gui_imports_remain_available(self):
        from scenespy import ScenespyApp, main
        from scenespy.app import ScenespyApp as DirectApp

        self.assertIs(ScenespyApp, DirectApp)
        self.assertTrue(callable(main))

    def test_modes_and_aliases(self):
        self.assertEqual(api._normalize_mode("scene"), "scene")
        self.assertEqual(api._normalize_mode("Scenes"), "scene")
        self.assertEqual(api._normalize_mode("every_seconds"), "interval")
        self.assertEqual(api._normalize_mode("extract_faces"), "faces")

    def test_invalid_mode_values(self):
        for value in (None, 1, True, [], {}):
            with self.assertRaises(TypeError):
                api._normalize_mode(value)
        for value in ("", "unknown", "10"):
            with self.assertRaises(ValueError):
                api._normalize_mode(value)

    def test_sensitivities_are_case_insensitive(self):
        self.assertEqual(api._normalize_sensitivity("low", "scene"), "Low")
        self.assertEqual(api._normalize_sensitivity("NORMAL", "scene"), "Normal")
        self.assertEqual(api._normalize_sensitivity("High", "faces"), "High")
        self.assertEqual(api._normalize_sensitivity("auto", "scene"), "Auto")

    def test_invalid_sensitivity_values(self):
        for value in (None, 1, True, [], {}):
            with self.assertRaises(TypeError):
                api._normalize_sensitivity(value, "scene")
        with self.assertRaises(ValueError):
            api._normalize_sensitivity("maximum", "scene")
        with self.assertRaises(ValueError):
            api._normalize_sensitivity("Auto", "faces")

    def test_empty_paths_are_rejected(self):
        for value in ("", " ", "\t", "\n"):
            with self.assertRaises(ValueError):
                api._normalize_path(value, "video")
        for value in (None, 1, True, [], {}):
            with self.assertRaises(TypeError):
                api._normalize_path(value, "video")

    def test_interval_validation_happens_before_file_access(self):
        for value in (True, False, 1.5, "10", None, [], {}):
            with self.assertRaises(TypeError):
                api.process_video("missing.mp4", "output", mode="interval",
                                  interval=value, verbose=False)
        for value in (0, -1, 18001):
            with self.assertRaises(ValueError):
                api.process_video("missing.mp4", "output", mode="interval",
                                  interval=value, verbose=False)

    def test_callback_and_boolean_options(self):
        with self.assertRaises(TypeError):
            api.process_video("missing.mp4", "output", progress="callback", verbose=False)
        with self.assertRaises(TypeError):
            api.process_video("missing.mp4", "output", verbose=1)
        with self.assertRaises(TypeError):
            api.process_videos(["missing.mp4"], "output", continue_on_error=1, verbose=False)

    def test_output_file_is_rejected_as_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir, "video.mp4")
            output = Path(temp_dir, "output")
            video.write_bytes(b"video")
            output.write_text("file", encoding="utf-8")
            with self.assertRaises(NotADirectoryError):
                api._validate_input(video, output, "interval")

    def test_temporary_repaired_video_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir, "video_fixed.mp4")
            video.write_bytes(b"video")
            with self.assertRaises(ValueError):
                api._validate_input(video, Path(temp_dir, "output"), "interval")

    def test_batch_keeps_invalid_items_as_results(self):
        values = ["one.mp4", None, 42]
        with patch("scenespy.api.process_video", side_effect=ValueError("invalid video")):
            results = api.process_videos(values, "output", verbose=False)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result["success"] is False for result in results))
        self.assertTrue(all(result["error_type"] == "ValueError" for result in results))

    def test_batch_can_stop_on_first_error(self):
        with patch("scenespy.api.process_video", side_effect=ValueError("invalid video")):
            with self.assertRaisesRegex(ValueError, "invalid video"):
                api.process_videos(["one.mp4"], "output", verbose=False,
                                   continue_on_error=False)

    def test_status_text_matches_app_format(self):
        self.assertEqual(format_status_lines("faces", 14, 7, "00:00"), [
            "Faces detected : 14",
            "Faces saved    : 7",
            "Estimated time : 00:00",
        ])
        self.assertEqual(format_status_lines("scene", 8, 8, "--:--"), [
            "Scenes detected : 8",
            "Scenes cut      : 8",
            "Estimated time  : --:--",
        ])
        self.assertEqual(format_status_lines("interval", 3, 3, "00:00"), [
            "Segments total : 3",
            "Segments cut   : 3",
            "Estimated time : 00:00",
        ])


if __name__ == "__main__":
    unittest.main()
