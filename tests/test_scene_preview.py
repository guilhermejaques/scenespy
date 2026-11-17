import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scenespy.scene_engine import SceneEngine


class ScenePreviewTests(unittest.TestCase):
    def test_cut_progress_labels_distinguish_scenes_and_intervals(self):
        engine = SceneEngine("video.mp4", "output", {})
        engine.scene_mode = True
        self.assertEqual(engine._cut_progress_label(3, 12),
                         "Scenes cut: 3 / 12")
        engine.scene_mode = False
        self.assertEqual(engine._cut_progress_label(7, 20),
                         "Segments cut: 7 / 20")

    def test_mkv_video_start_offset_is_removed_from_tagged_duration(self):
        engine = SceneEngine("video.mkv", "output", {})
        payload = json.dumps({"streams": [{
            "start_time": "0.083",
            "tags": {"DURATION": "00:00:06.082000000"},
        }]}).encode()

        with patch(
                "scenespy.scene_engine.check_output_hidden_cancelable",
                return_value=payload):
            frames = engine._get_video_total_frames(24.0, 6.12)

        self.assertEqual(frames, 144)

    def test_scene_backend_without_close_releases_container_and_file(self):
        released = []

        class Resource:
            def __init__(self, name):
                self.name = name

            def close(self):
                released.append(self.name)

        backend = type("Backend", (), {
            "_container": Resource("container"),
            "_io": Resource("file"),
        })()

        SceneEngine._close_video_stream(backend)

        self.assertEqual(released, ["container", "file"])

    def test_scene_analysis_cache_requires_exact_source_and_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder, "video.mp4")
            source.write_bytes(b"source")
            cache = Path(folder, "scene-analysis.json")
            cfg = {
                "label": "Normal", "base_threshold": 27,
                "min_dur": 2.0, "dur_max_boost": 4.0,
                "ANALYSIS_CACHE_FILE": str(cache),
            }
            engine = SceneEngine(str(source), folder, cfg)
            scenes = [(0, 24), (24, 48)]
            engine._save_analysis_cache(scenes)

            self.assertEqual(engine._load_analysis_cache(), scenes)

            changed_profile = SceneEngine(
                str(source), folder, {**cfg, "label": "High"})
            self.assertIsNone(changed_profile._load_analysis_cache())

            source.write_bytes(b"changed source")
            self.assertIsNone(engine._load_analysis_cache())

    def test_export_worker_count_is_bounded_per_backend(self):
        cpu = SceneEngine("video.mp4", "output", {"ENCODER": "cpu"})
        nvidia = SceneEngine(
            "video.mp4", "output", {"ENCODER": "nvidia"})
        intel = SceneEngine(
            "video.mp4", "output", {"ENCODER": "intel"})
        for engine in (cpu, nvidia, intel):
            engine.scene_mode = True

        self.assertLessEqual(cpu._export_worker_count(20), 3)
        self.assertEqual(nvidia._export_worker_count(20), 2)
        self.assertEqual(intel._export_worker_count(20), 1)
        nvidia.scene_mode = False
        self.assertEqual(nvidia._export_worker_count(20), 2)

    def test_grouped_interval_requires_regular_contiguous_cfr_segments(self):
        engine = SceneEngine(
            "video.mp4", "output", {
                "FIXED_INTERVAL": 1,
                "ENABLE_GROUPED_INTERVAL_EXPORT": True,
            })
        engine.scene_mode = False
        tasks = [
            (index, f"scene_{index}.mp4", index - 1.0, float(index),
             index - 1.0, float(index), (index - 1) * 24, index * 24, True)
            for index in range(1, 21)]

        self.assertTrue(engine._grouped_interval_eligible(tasks, 24.0))

        with_gap = list(tasks)
        with_gap[10] = (*with_gap[10][:6], 250, 264, True)
        self.assertFalse(engine._grouped_interval_eligible(with_gap, 24.0))
        engine.cfg["FRAME_TIMES"] = [0.0, 0.04]
        self.assertFalse(engine._grouped_interval_eligible(tasks, 24.0))

    def test_process_summary_separates_operation_and_pipeline_timings(self):
        engine = SceneEngine(
            "video.mp4", "output",
            {"PIPELINE_TIMINGS": {
                "analysis_proxy_seconds": 1.25,
                "analysis_seconds": 2.5,
            }})
        engine.scene_mode = True
        engine._start_time = 10.0
        engine._end_time = 14.0
        engine._stage_timings["export_seconds"] = 3.0
        engine._get_video_duration = lambda: 1.0
        engine._get_video_fps = lambda: 24.0
        engine._get_video_total_frames = lambda _fps, _duration: 24

        timings = engine._process_summary("output")["process"]["timings"]

        self.assertEqual(timings["operation_seconds"], 4.0)
        self.assertEqual(timings["measured_pipeline_seconds"], 6.75)

    def test_scene_export_records_match_timeline_ranges_and_names(self):
        engine = SceneEngine("video.mp4", "output", {})
        records = engine._scene_export_records(
            [(0, 24), (24, 60)], [1, 3], 24.0)

        self.assertEqual([record["scene_number"] for record in records],
                         [1, 3])
        self.assertEqual([record["file"] for record in records],
                         ["scene_1.mp4", "scene_3.mp4"])
        self.assertEqual(records[0]["start_frame"], 0)
        self.assertEqual(records[0]["end_frame"], 24)
        self.assertEqual(records[0]["last_frame"], 23)
        self.assertEqual(records[0]["frame_count"], 24)
        self.assertEqual(records[0]["start_time"], 0.0)
        self.assertEqual(records[0]["end_time"], 1.0)
        self.assertEqual(records[1]["duration"], 1.5)

    def test_scene_json_contains_auditable_clip_records(self):
        engine = SceneEngine("video.mp4", "output", {})
        engine.done = 1
        engine._cut_scene_records = engine._scene_export_records(
            [(0, 24)], [1], 24.0)
        engine._cut_scene_records[0].update(
            actual_duration=1.0, status="exported", exported=True)
        engine._process_summary = lambda _outdir: {
            "video": {"file": "video.mp4", "duration": 1.0},
            "process": {"mode": "scene"},
            "output": {"folder": _outdir},
        }

        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "scene_1.mp4").write_bytes(b"exported")
            engine._write_scene_metadata(folder, [(0, 24)], 24.0)
            payload = json.loads(Path(folder, "scenes.json").read_text(
                encoding="utf-8"))

        self.assertEqual(payload["results"]["total_cuts"], 1)
        self.assertEqual(next(iter(payload)), "summary")
        self.assertTrue(payload["summary"]["audit_passed"])
        self.assertEqual(payload["scenes"], engine._cut_scene_records)
        self.assertEqual(payload["scenes"][0]["file"], "scene_1.mp4")
        self.assertTrue(payload["scenes"][0]["exported"])

    def test_scene_json_reports_hardware_fallbacks(self):
        engine = SceneEngine(
            "video.mp4", "output", {"ENCODER": "nvidia"})
        engine.done = 1
        engine._cut_scene_records = engine._scene_export_records(
            [(0, 24)], [1], 24.0)
        engine._cut_scene_records[0].update(
            actual_duration=1.0, status="exported", exported=True,
            acceleration={
                "requested": "nvidia", "used": "cpu", "fallback": True,
                "fallback_reason": "NVENC unavailable",
            })
        engine._process_summary = lambda _outdir: {
            "video": {"file": "video.mp4", "duration": 1.0},
            "process": {"mode": "scene"},
            "output": {"folder": _outdir},
        }

        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "scene_1.mp4").write_bytes(b"exported")
            engine._write_scene_metadata(folder, [(0, 24)], 24.0)
            payload = json.loads(Path(folder, "scenes.json").read_text(
                encoding="utf-8"))

        self.assertEqual(payload["summary"]["acceleration_requested"], "nvidia")
        self.assertEqual(payload["summary"]["acceleration_fallbacks"], 1)
        self.assertEqual(payload["scenes"][0]["acceleration"]["used"], "cpu")

    def test_failed_cut_updates_matching_scene_record(self):
        engine = SceneEngine("video.mp4", "output", {})
        engine._cut_scene_records = engine._scene_export_records(
            [(24, 48)], [2], 24.0)
        engine._cut_scene_records_by_number = {2: engine._cut_scene_records[0]}
        task = (2, "scene_2.mp4", 1.0, 2.0, 0.0, 2.0, 24, 48, True)

        failure = engine._record_cut_failure(task, RuntimeError("cut failed"))

        self.assertEqual(failure["scene_number"], 2)
        self.assertEqual(engine._cut_scene_records[0]["status"], "failed")
        self.assertFalse(engine._cut_scene_records[0]["exported"])
        self.assertEqual(engine._cut_scene_records[0]["error"], "cut failed")

    def test_interval_pacing_uses_shared_decoder_for_final_frame(self):
        frames = []

        class Previewer:
            enabled = True

            def after(self, _delay, callback):
                callback()

            def update_image_at(self, _image, seconds):
                frames.append(seconds)

        engine = SceneEngine(
            "video.mp4", "output",
            {"INTERVAL_PREVIEW_SECONDS": .01},
            previewer=Previewer())
        engine._analysis_ratio = 0.0
        engine._get_video_duration = lambda: 1.0
        engine._get_video_fps = lambda: 10.0
        engine._total_frames = 10
        engine._get_preview_frame_at = lambda _seconds: object()

        engine._pace_fixed_interval_preview()

        self.assertEqual(engine._analysis_ratio, 1.0)
        self.assertEqual(frames, [.9])

    def test_full_progress_preview_uses_last_frame_not_eof(self):
        engine = SceneEngine("video.mp4", "output", {})
        engine._get_video_duration = lambda: 2.0
        engine._get_video_fps = lambda: 10.0

        self.assertEqual(engine._preview_time_for_ratio(0.0), 0.0)
        self.assertAlmostEqual(engine._preview_time_for_ratio(1.0), 1.9)

        engine._total_frames = 19
        self.assertAlmostEqual(engine._preview_time_for_ratio(1.0), 1.8)

    def test_export_frame_time_prefers_vfr_timestamp_map(self):
        engine = SceneEngine.__new__(SceneEngine)
        engine.cfg = {"FRAME_TIMES": [0.0, .031, .071, .106]}
        engine._get_video_duration = lambda: .15

        self.assertEqual(engine._time_for_frame(2, 30.0), .071)
        self.assertEqual(engine._time_for_frame(4, 30.0), .15)

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
