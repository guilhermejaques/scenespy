import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scenespy import shared
from scenespy.qt_app import (FramePreviewDecoder, ScenespyApp, Task,
                             remove_owned_directory)


class LifecycleCleanupTests(unittest.TestCase):
    def test_adaptive_cache_profile_is_bounded_by_hardware_tier(self):
        gib = 1024 ** 3
        compact = shared.adaptive_cache_profile(2, 4 * gib)
        balanced = shared.adaptive_cache_profile(8, 16 * gib)
        expanded = shared.adaptive_cache_profile(16, 32 * gib)

        self.assertEqual(compact["tier"], "compact")
        self.assertEqual(balanced["tier"], "balanced")
        self.assertEqual(expanded["tier"], "expanded")
        self.assertLess(compact["frame_cache"], balanced["frame_cache"])
        self.assertLess(
            balanced["clip_thumbnail_cache"],
            expanded["clip_thumbnail_cache"])
        self.assertLessEqual(expanded["cut_workers"], 3)

    def test_frame_cache_evicts_oldest_entry_at_its_limit(self):
        decoder = FramePreviewDecoder(cache_size=2)
        decoder._remember(("video", 1), "one")
        decoder._remember(("video", 2), "two")
        decoder._remember(("video", 3), "three")
        self.assertEqual(
            list(decoder._cache.items()),
            [(("video", 2), "two"), (("video", 3), "three")],
        )

    def test_clip_preview_status_omits_percentage(self):
        class StatusBar:
            message = "Previewing clip"

            def currentMessage(self):
                return self.message

            def showMessage(self, message, *_args):
                self.message = message

        status = StatusBar()
        window = SimpleNamespace(
            _segment_preview_status_active=True,
            _segment_preview_last_status_ms=None,
            preview_start_ms=1000,
            preview_end_ms=3000,
            current_media=SimpleNamespace(duration=10),
            statusBar=lambda: status,
            _player_timecode=lambda seconds, _duration: f"{seconds:.1f}",
        )
        ScenespyApp._update_segment_preview_status(window, 2000, force=True)
        self.assertNotIn("%", status.message)
        self.assertIn("2.0 / 3.0", status.message)

    def test_stopping_short_preview_does_not_clear_duration_label(self):
        class Timeline:
            cleared = False

            def clear_clip_duration(self):
                self.cleared = True

        class StatusBar:
            def currentMessage(self):
                return "Previewing clip"

            def clearMessage(self):
                pass

            def showMessage(self, *_args):
                pass

        timeline = Timeline()
        window = SimpleNamespace(
            _segment_preview_status_active=True,
            _segment_preview_last_status_ms=1000,
            preview_start_ms=0,
            preview_end_ms=1000,
            preview_armed=True,
            timeline=timeline,
            statusBar=lambda: StatusBar(),
        )
        ScenespyApp._stop_segment_preview(window)
        self.assertFalse(timeline.cleared)

    def test_owned_directory_cleanup_is_confined_to_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "exports"
            owned = root / "SCENE_generated"
            outside = Path(folder) / "keep"
            owned.mkdir(parents=True)
            outside.mkdir()
            (owned / "partial.mp4").write_bytes(b"partial")
            (outside / "original.mp4").write_bytes(b"original")

            self.assertTrue(remove_owned_directory(str(owned), str(root)))
            self.assertFalse(owned.exists())
            self.assertTrue(outside.exists())
            self.assertFalse(remove_owned_directory(str(outside), str(root)))
            self.assertFalse(remove_owned_directory(str(root), str(root)))
            self.assertTrue(root.exists())
            self.assertTrue((outside / "original.mp4").exists())

    def test_task_runs_registered_cleanup_after_success(self):
        cleaned = []

        def work(task):
            task.register_cleanup(lambda: cleaned.append("done"))
            return 1

        Task(work).run()
        self.assertEqual(cleaned, ["done"])

    def test_task_cleanup_finishes_before_result_is_published(self):
        events = []

        def work(task):
            task.register_cleanup(lambda: events.append("cleanup"))
            return "value"

        task = Task(work)
        task.result.connect(lambda value: events.append(f"result:{value}"))
        task.run()
        self.assertEqual(events, ["cleanup", "result:value"])

    def test_task_runs_registered_cleanup_after_failure(self):
        cleaned = []

        def work(task):
            task.register_cleanup(lambda: cleaned.append("done"))
            raise RuntimeError("expected test failure")

        Task(work).run()
        self.assertEqual(cleaned, ["done"])

    def test_resistant_child_stays_registered_for_shutdown_retry(self):
        class ResistantProcess:
            def poll(self):
                return None

            def terminate(self):
                raise OSError("still running")

            def kill(self):
                raise OSError("still running")

            def wait(self, timeout=None):
                raise TimeoutError(timeout)

        process = ResistantProcess()
        shared.register_child_process(process)
        try:
            shared.terminate_all_child_processes(grace_seconds=0)
            self.assertIn(process, shared._CHILD_PROCS)
        finally:
            shared.unregister_child_process(process)


if __name__ == "__main__":
    unittest.main()
