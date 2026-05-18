from .shared import *
from .scene_analysis import *

class SceneEngine:
    """Detects scene boundaries and exports video segments."""
    def __init__(self, video, output, cfg, logbox=None, progressbar=None,
                 previewer=None, preview_enabled=True, preview_pool=None):
        self.video = video
        self.output = output
        self.cfg = cfg
        self.log = logbox
        self.progress = progressbar
        self.previewer = previewer
        self.preview_enabled = preview_enabled
        self.preview_pool = preview_pool or [video]
        self._stop = False
        self.detected = 0
        self.total = 0
        self.done = 0
        self.failed = 0
        self.completed_attempts = 0
        self._start_time = None
        self._end_time = None
        self._video_info_shown = False
        self._fps = None
        self._total_frames = None
        self._ffmpeg_proc = None
        self._ffmpeg_procs = set()
        self._ffmpeg_proc_lock = threading.Lock()
        self._ui_alive = True
        self._duration = None
        self._last_preview_ratio = -1
        self._keyframes_cache = None
        self._has_audio_cache = None
        self._preview_cap = None
        self._preview_lock = threading.Lock()
        self._preview_thread = None
        self._preview_stop = False
        self._analysis_mosaic = None
        self._analysis_status_step = 0
        self._video_obj = None
        self._scene_candidates = []
        self._rejected_scene_candidates = []
        self._cut_failures = []
        self._cut_output_dir = None
        self._temp_files = []

    def add_temp_file(self, path):
        if path and path not in self._temp_files:
            self._temp_files.append(path)

    def cleanup_temp_files(self):
        for path in list(self._temp_files):
            remove_temp_file(path)
        self._temp_files = []

    def stop(self):
        self._stop = True
        self._ui_alive = False
        self._preview_stop = True
        try:
            procs = []
            if self._ffmpeg_proc:
                procs.append(self._ffmpeg_proc)
            try:
                with self._ffmpeg_proc_lock:
                    procs.extend(list(self._ffmpeg_procs))
            except Exception:
                pass
            for proc in procs:
                terminate_process(proc, grace_seconds=0.3)
        except Exception:
            pass
        try:
            if self._video_obj:
                self._video_obj.close()
        except Exception:
            pass
        self._video_obj = None
        try:
            if self._preview_cap:
                with self._preview_lock:
                    if self._preview_cap:
                        self._preview_cap.release()
        except Exception:
            pass
        self._preview_cap = None
        self._keyframes_cache = None
        self._duration = None
        self._total_frames = None
        self._has_audio_cache = None
        self._ffmpeg_proc = None
        try:
            with self._ffmpeg_proc_lock:
                self._ffmpeg_procs.clear()
        except Exception:
            pass
        self._scene_candidates = []
        self._rejected_scene_candidates = []
        self._cut_failures = []
        self._cut_output_dir = None
        if self._analysis_mosaic:
            self._analysis_mosaic.stop()
            self._analysis_mosaic = None
        self.cleanup_temp_files()

    def disable_preview(self):
        self.preview_enabled = False
        if self._analysis_mosaic:
            self._analysis_mosaic.stop()
            self._analysis_mosaic = None
        self._preview_stop = True
        if self._preview_thread and self._preview_thread.is_alive() and threading.current_thread() is not self._preview_thread:
            try:
                self._preview_thread.join(timeout=1.0)
            except Exception:
                pass
        self._preview_thread = None
        try:
            with self._preview_lock:
                if self._preview_cap:
                    self._preview_cap.release()
                self._preview_cap = None
        except Exception:
            pass

    def total_time(self):
        if not self._start_time:
            return "--:--"
        end = self._end_time or time.time()
        elapsed = max(1, int(end - self._start_time))
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _format_process_timestamp(self, ts):
        if not ts:
            return None
        try:
            return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")
        except Exception:
            return None

    def _format_elapsed(self, seconds):
        seconds = max(0, int(round(float(seconds or 0))))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _process_summary(self, outdir):
        end_time = self._end_time or time.time()
        elapsed = max(0.0, end_time - self._start_time) if self._start_time else 0.0
        try:
            duration = round(float(self._get_video_duration()), 4)
        except Exception:
            duration = None
        return {
            "video": {
                "file": os.path.basename(self.video),
                "path": self.video,
                "duration": duration,
            },
            "process": {
                "mode": "scene" if getattr(self, "scene_mode", True) else "interval",
                "profile": self.cfg.get("label", "NA"),
                "encoder": self.cfg.get("ENCODER", "cpu"),
                "started_at": self._format_process_timestamp(self._start_time),
                "finished_at": self._format_process_timestamp(end_time),
                "elapsed": self._format_elapsed(elapsed),
                "elapsed_seconds": round(elapsed, 2),
            },
            "output": {
                "folder": outdir,
            },
        }

    def _reset_detection_timing(self):
        self._detection_timing_last = time.time()
        self._detection_timing_start = self._detection_timing_last

    def _log_detection_stage(self, stage):
        if not self.cfg.get("DEBUG", False):
            return
        now = time.time()
        last = getattr(self, "_detection_timing_last", now)
        start = getattr(self, "_detection_timing_start", last)
        print(f"[TIMING] {stage}: +{now - last:.2f}s total={now - start:.2f}s")
        self._detection_timing_last = now

    def _analysis_eta_text(self, ratio=None):
        step = getattr(self, "_analysis_status_step", 0)
        dots = "." * ((step % 3) + 1)
        self._analysis_status_step = step + 1
        if ratio is not None:
            pct = max(0, min(99, int(ratio * 100)))
            return f"analyzing {pct}%{dots}"
        return f"analyzing{dots}"

    def _write_analysis_status(self, label, progress=None):
        if self.progress and progress is not None:
            self.progress.after(0, lambda v=progress: self.progress.update(v))
        if self.log:
            self.log.after(0, lambda text=label: self.log.write_status(
                detected=text, cut=0, eta=self._analysis_eta_text()))

    def run(self, scene_mode=True):
        self.scene_mode = scene_mode
        self._stop = False
        self._preview_stop = False
        self._ui_alive = True
        self._analysis_ratio = 0.0
        self._start_time = time.time()
        self._end_time = None
        self.done = 0
        self.failed = 0
        self.completed_attempts = 0
        self.detected = 0
        self.last_preview = 0
        self._scene_candidates = []
        self._rejected_scene_candidates = []
        self._cut_failures = []
        self._cut_output_dir = None
        self._analysis_mosaic = None
        self._analysis_status_step = 0

        if self.previewer and self.preview_enabled:
            try:
                img = self._get_preview_frame_at(0)
                if img and self._ui_alive:
                    self.previewer.after(0, lambda img=img: self.previewer.update_image(img))
            except Exception:
                pass

        if self.preview_enabled:
            try:
                with self._preview_lock:
                    self._preview_cap = cv2.VideoCapture(self.video)
            except Exception:
                self._preview_cap = None

        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=False)
        self._preview_thread.start()

        if self.previewer and not self._video_info_shown:
            info_text = self._get_video_info_text()
            self.previewer.after(0, lambda t=info_text: self.previewer.update_info(t))
            self._video_info_shown = True

        try:
            if self.preview_enabled and self.previewer:
                self._analysis_mosaic = AnalysisMosaicPreview(
                    self.preview_pool, self.previewer, stop_cb=lambda: self._stop)
                self._analysis_mosaic.start()
            scenes = self._detect_scenes_progressive() if scene_mode else self._fixed_interval()
            if self._analysis_mosaic:
                self._analysis_mosaic.stop()
                self._analysis_mosaic = None
            if not scenes or self._stop:
                return False

            if self.log:
                self.log.after(0, lambda: self.log.write_status(detected=self.detected, cut=0, eta="--:--"))

            self._cut_scenes(scenes)
            if self._stop:
                return False
        finally:
            if self._analysis_mosaic:
                self._analysis_mosaic.stop()
                self._analysis_mosaic = None
            self._ui_alive = False
            self._preview_stop = True
            if self._preview_thread and self._preview_thread.is_alive():
                try:
                    self._preview_thread.join(timeout=1.0)
                except Exception:
                    pass
            self._preview_thread = None

            try:
                with self._preview_lock:
                    if self._preview_cap:
                        self._preview_cap.release()
                    self._preview_cap = None
            except Exception:
                pass
            self.cleanup_temp_files()

        self._end_time = time.time()

        return True

    def _get_video_info_text(self):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries",
               "stream=width,height,avg_frame_rate,r_frame_rate,bit_rate:stream_tags=rotate:stream_side_data=rotation:format=bit_rate,duration",
               "-of", "json", self.video]
        try:
            data = json.loads(check_output_hidden(cmd).decode(errors="ignore") or "{}")
            stream = (data.get("streams") or [{}])[0]
            fmt = data.get("format") or {}

            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            rotation = self._get_video_rotation(stream)
            if width > 0 and height > 0 and abs(rotation) % 180 == 90:
                width, height = height, width

            fps = (
                self._parse_frame_rate(stream.get("avg_frame_rate")) or
                self._parse_frame_rate(stream.get("r_frame_rate"))
            )
            bitrate = (
                self._parse_int(stream.get("bit_rate")) or
                self._parse_int(fmt.get("bit_rate")) or
                self._estimate_bitrate_from_size(fmt.get("duration"))
            )

            resolution_text = f"{width}x{height}" if width > 0 and height > 0 else "Resolution: unknown"
            fps_text = f"FPS: {fps:.2f}" if fps else "FPS: unknown"
            bitrate_text = f"Bitrate: {bitrate / 1000:.0f} kbps" if bitrate else "Bitrate: unknown"
            return f"{resolution_text} | {fps_text} | {bitrate_text}"
        except Exception:
            return "Video info unavailable"

    def _parse_frame_rate(self, value):
        try:
            text = str(value or "").strip()
            if not text or text in {"0/0", "N/A"}:
                return None
            if "/" in text:
                num, den = text.split("/", 1)
                den = float(den)
                if den == 0:
                    return None
                fps = float(num) / den
            else:
                fps = float(text)
            return fps if fps > 0 else None
        except Exception:
            return None

    def _parse_int(self, value):
        try:
            text = str(value or "").strip()
            if not text or text == "N/A":
                return None
            parsed = int(float(text))
            return parsed if parsed > 0 else None
        except Exception:
            return None

    def _get_video_rotation(self, stream):
        try:
            rotate = self._parse_int((stream.get("tags") or {}).get("rotate"))
            if rotate is not None:
                return rotate
            for item in stream.get("side_data_list") or []:
                rotation = self._parse_int(item.get("rotation"))
                if rotation is not None:
                    return rotation
        except Exception:
            pass
        return 0

    def _estimate_bitrate_from_size(self, duration):
        try:
            seconds = float(duration)
            if seconds <= 0:
                return None
            return int((os.path.getsize(self.video) * 8) / seconds)
        except Exception:
            return None

    def _detect_scenes_progressive(self):
        self._reset_detection_timing()
        fps = self._get_video_fps()

        if self.log:
            self.log.after(0, lambda: self.log.write_status(
                detected=self._analysis_eta_text(), cut=0, eta="analyzing"))

        threshold, min_dur = self._map_threshold()
        self._log_detection_stage("adaptive_threshold")

        result = _ensure_scenedetect()
        if result is None:
            raise RuntimeError("scenedetect not installed")
        open_video, SceneManager, ContentDetector, StatsManager, CompWeights = result

        backend = "pyav"
        if "_fixed" in os.path.splitext(os.path.basename(self.video))[0].lower():
            backend = "opencv"

        try_backends = ["pyav", "opencv"] if backend == "pyav" else [backend]
        video = None
        last_error = None
        for be in try_backends:
            try:
                if be == "pyav":
                    video = open_video(self.video, backend=be, suppress_output=True)
                else:
                    video = open_video(self.video, backend=be)
                self._video_obj = video
                break
            except Exception as e:
                last_error = e
                if be == "pyav":
                    continue
        if video is None:
            raise RuntimeError(f"Failed to open video: {last_error}")

        if backend == "opencv":
            try:
                _ = video.frame_rate
            except Exception:
                video.close()
                self._video_obj = None
                raise RuntimeError("Failed to read video stream")

        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.auto_downscale = False
        scene_manager.downscale = 2
        scene_manager.add_detector(
            ContentDetector(threshold=threshold,
                            min_scene_len=int(min_dur * fps),
                            luma_only=False,
                            weights=CompWeights(
                                delta_hue=0.85, delta_sat=0.85, delta_lum=1.0, delta_edges=1.15))
        )
        video_duration = None
        try:
            video_duration = self._get_video_duration()
        except Exception:
            pass

        if video_duration and video_duration > 0:
            self._total_frames = self._get_video_total_frames(fps, video_duration)

        if not video_duration or video_duration <= 0:
            try:
                cmd = ["ffprobe", "-v", "error",
                       "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", self.video]
                video_duration = float(check_output_hidden(cmd).decode().strip())
                self._total_frames = int(video_duration * fps)
            except Exception:
                if self._total_frames is None:
                    self._total_frames = int(video.frame_rate * 1)
                video_duration = max((self._total_frames or 1) / fps, 1.0)

        def _progress_cb(frame_num, _):
            if self._stop:
                return False

            try:
                frame_idx = _safe_frame_index(frame_num)
                if frame_idx <= 0:
                    return True
            except Exception:
                return True

            current_time = frame_idx / fps
            if time.time() - getattr(self, "_last_ui_update", 0) < 0.3:
                return True
            self._last_ui_update = time.time()

            if video_duration and video_duration > 0:
                ratio = min(current_time / video_duration, 1.0)
            else:
                ratio = self.done / max(self.total, 1)
            ratio = max(self._analysis_ratio, ratio)
            self._analysis_ratio = ratio

            if self.progress:
                self.progress.after(0, lambda v=ratio * 0.4: self.progress.update(v))

            if self.log:
                eta = self._analysis_eta_text(ratio)
                self.log.after(0, lambda d=self.detected, c=self.done, e=eta:
                self.log.write_status(detected=d, cut=c, eta=e))
            return True

        scene_list = []
        try:
            scene_manager.detect_scenes(video=video, callback=_progress_cb)
            if self._stop:
                return []

            scene_list = scene_manager.get_scene_list()
            self.detected = len(scene_list)
            self._log_detection_stage("pyscenedetect")
        except Exception as e:
            err_str = str(e).lower()
            needs_retry = (
                    "avcodec_send_packet" in err_str or
                    "avcodec" in err_str or
                    "decode" in err_str or
                    "invalid data" in err_str or
                    "no start code" in err_str or
                    "avcodec_receive_frame" in err_str or
                    "packet" in err_str or
                    "av.error" in err_str or
                    "fatal" in err_str or
                    "thread" in err_str
            )
            if needs_retry and backend == "pyav":
                try:
                    if video:
                        video.close()
                        self._video_obj = None
                except Exception:
                    pass
                fixed = self.video.rsplit(".", 1)[0] + "_fixed.mp4"
                remove_temp_file(fixed)
                self.add_temp_file(fixed)
                print(f"Video has corrupted frames - re-encoding with ffmpeg...")
                if self.log:
                    self.log.after(0, lambda: self.log.write_status(
                        detected="Fixing corrupted video...", cut=0, eta="--:--"))
                self._run_ffmpeg_tracked(
                    ["ffmpeg", "-y", "-err_detect", "ignore_err",
                     "-i", self.video, "-c:v", "libx264", "-crf", "22",
                     "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                     "-c:a", "copy", fixed],
                    timeout=300
                )
                if self.log:
                    self.log.after(0, lambda: self.log.write_status(
                        detected="Re-encoding complete", cut=0, eta="--:--"))

                if os.path.exists(fixed):
                    print(f"Using fixed video: {fixed}")
                    if not is_valid_video_file(fixed):
                        print(f"Fixed file is corrupted, re-encoding again...")
                        remove_temp_file(fixed)
                        self._run_ffmpeg_tracked(
                            ["ffmpeg", "-y", "-err_detect", "ignore_err",
                             "-i", self.video, "-c:v", "libx264", "-crf", "22",
                             "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                             "-c:a", "copy", fixed],
                            timeout=300
                        )
                    if not os.path.exists(fixed) or not is_valid_video_file(fixed):
                        print("ffmpeg re-encode failed, no valid fixed file")
                        if self.log:
                            self.log.after(0, lambda: self.log.write_status(
                                detected="Failed to fix corrupted video", cut=0, eta="--:--"))
                        return []
                    self.video = fixed

                    self._fps = None
                    self._duration = None
                    self._keyframes_cache = None
                    self._total_frames = None
                    fps = self._get_video_fps()
                    try:
                        video_duration = self._get_video_duration()
                        self._total_frames = self._get_video_total_frames(fps, video_duration)
                    except Exception:
                        video_duration = None
                    try:
                        video = open_video(self.video, backend="opencv")
                        self._video_obj = video
                    except Exception:
                        try:
                            video = open_video(self.video, backend="pyav", suppress_output=True)
                            self._video_obj = video
                        except Exception:
                            print("Both backends failed on fixed video!")
                            if self.log:
                                self.log.after(0, lambda: self.log.write_status(
                                    detected="Failed to open fixed video", cut=0, eta="--:--"))
                            return []
                    print(f"Video opened successfully, starting detection...")
                    scene_manager = SceneManager(StatsManager())
                    scene_manager.auto_downscale = False
                    scene_manager.downscale = 2
                    scene_manager.add_detector(
                        ContentDetector(threshold=threshold,
                                        min_scene_len=int(min_dur * fps),
                                        luma_only=False,
                                        weights=CompWeights(
                                            delta_hue=0.85, delta_sat=0.85, delta_lum=1.0, delta_edges=1.15))
                    )
                    scene_manager.detect_scenes(video=video, callback=_progress_cb)
                    scene_list = scene_manager.get_scene_list()
                    self.detected = len(scene_list)
                    self._log_detection_stage("pyscenedetect_fixed")
                    print(f"Detection complete: {self.detected} scenes found")
                    if self.log:
                        self.log.after(0, lambda d=self.detected: self.log.write_status(
                            detected=f"{d} scenes detected", cut=0, eta="--:--"))
                    try:
                        video.close()
                        self._video_obj = None
                    except Exception:
                        pass
                    scenes = self._compose_scene_list(scene_list, fps, min_dur)
                    return scenes
                else:
                    print("ffmpeg re-encode failed, no fixed file created")
                    if self.log:
                        self.log.after(0, lambda: self.log.write_status(
                            detected="Failed to create fixed video", cut=0, eta="--:--"))
                    return []
            else:
                return []
        finally:
            try:
                if video:
                    video.close()
                    self._video_obj = None
            except Exception:
                pass

        scenes = self._compose_scene_list(scene_list, fps, min_dur)
        self.detected = len(scenes)
        return scenes

    def _compose_scene_list(self, scene_list, fps, min_dur):
        total_frames = self._total_frames
        if not total_frames:
            try:
                duration = self._get_video_duration()
                total_frames = self._get_video_total_frames(fps, duration)
            except Exception:
                total_frames = int(fps)

        self._write_analysis_status("building candidates", 0.45)
        hard_boundaries = set()
        for start, end in scene_list or []:
            s = start.get_frames()
            e = end.get_frames()
            if 0 < s < total_frames:
                hard_boundaries.add(int(s))
            if 0 < e < total_frames:
                hard_boundaries.add(int(e))
        hard_candidates = [
            _make_candidate(frame, 1.0, "hard_cut", confidence=1.0, source="pyscenedetect")
            for frame in sorted(hard_boundaries)
        ]

        profile = _profile_name(self.cfg)
        feature_cache = _build_transition_feature_cache(
            self.video, fps, total_frames, profile=profile, stop_cb=lambda: self._stop)
        self._log_detection_stage("transition_feature_cache")
        self._write_analysis_status("finding transitions", 0.58)
        gradual_candidates = _detect_gradual_transitions(
            self.video, fps, total_frames, profile=profile,
            stop_cb=lambda: self._stop, features=feature_cache)
        semantic_candidates = _detect_semantic_transitions(
            self.video, fps, total_frames, profile=profile,
            stop_cb=lambda: self._stop, features=feature_cache)
        self._log_detection_stage("transition_candidates")

        min_gap = self._boundary_min_gap_frames(fps, min_dur, profile)
        candidates = self._merge_candidates(
            hard_candidates + gradual_candidates + semantic_candidates,
            min_gap, total_frames)
        candidates = _refine_scene_candidates(
            self.video, candidates, fps, total_frames, profile=profile,
            stop_cb=lambda: self._stop)
        self._log_detection_stage("candidate_refine")
        self._write_analysis_status("refining cuts", 0.70)
        candidates = self._merge_candidates(candidates, min_gap, total_frames)
        candidates = _add_candidate_context_scores(
            self.video, candidates, fps, total_frames, stop_cb=lambda: self._stop)
        self._log_detection_stage("candidate_context")
        self._write_analysis_status("checking context", 0.82)
        candidates, rejected = self._classify_candidates(candidates, fps, total_frames, profile, min_dur)
        self._log_detection_stage("candidate_classify")
        self._write_analysis_status("preparing cuts", 0.90)
        self._scene_candidates = candidates
        self._rejected_scene_candidates = rejected
        boundaries = [int(c["frame"]) for c in candidates]
        if not boundaries:
            self.detected = 1
            return [(0, total_frames)]
        scenes = self._scenes_from_boundaries(boundaries, total_frames, fps=fps)
        self.detected = len(scenes)
        return scenes

    def _boundary_min_gap_frames(self, fps, min_dur, profile):
        profile_floor = {
            "Low": 3.5,
            "Normal": 1.2,
            "High": 0.45,
            "Auto": 0.8,
        }.get(profile, 1.2)
        seconds = min(max(profile_floor, min_dur * 0.35), max(min_dur, profile_floor))
        return max(1, int(seconds * fps))

    def _classifier_min_scene_seconds(self, profile, adaptive_min_dur):
        params = _classifier_profile_params(profile)
        base = float(params["min_scene_s"])
        if adaptive_min_dur is None:
            return base
        ratios = {"Low": 0.55, "Normal": 0.38, "High": 0.22, "Auto": 0.30}
        caps = {"Low": 5.5, "Normal": 2.2, "High": 0.9, "Auto": 1.3}
        adaptive = float(adaptive_min_dur) * ratios.get(profile, 0.35)
        return min(caps.get(profile, base), max(base, adaptive))

    def _classify_candidates(self, candidates, fps, total_frames, profile, adaptive_min_dur=None):
        if not candidates:
            return [], []

        params = _classifier_profile_params(profile)
        accepted = []
        rejected = []
        sorted_candidates = sorted(candidates, key=lambda c: int(c["frame"]))
        min_scene_s = self._classifier_min_scene_seconds(profile, adaptive_min_dur)
        min_scene_frames = max(1, int(min_scene_s * fps))

        for idx, cand in enumerate(sorted_candidates):
            frame = int(cand["frame"])
            cand_type = cand.get("type", "unknown")
            merged_types = set(cand.get("merged_types", [cand_type]))
            confidence = float(cand.get("confidence", 0.0))
            refine_score = float(cand.get("refine_score", 0.0))
            context_ratio = float(cand.get("context_ratio", 1.0))
            context_cross = float(cand.get("context_cross", 0.0))
            camera_explained = float(cand.get("camera_motion_explained", 0.0))
            camera_aligned = float(cand.get("camera_motion_aligned", 0.0))
            score = confidence

            if cand_type == "hard_cut" or "hard_cut" in merged_types:
                score += 0.30
            if len(merged_types) >= 2:
                score += 0.16
            if refine_score >= params["refine_floor"]:
                score += min(0.18, refine_score * 0.35)
            context_veto = "hard_cut" not in merged_types and context_ratio < 0.75 and context_cross < 0.08
            camera_veto = (
                    "hard_cut" not in merged_types and
                    camera_explained > 0.42 and
                    camera_aligned < 0.075
            )

            if context_ratio >= 1.8 and context_cross >= 0.05:
                score += min(0.16, (context_ratio - 1.0) * 0.06)
            elif "hard_cut" not in merged_types and context_ratio < 1.20:
                score -= 0.45
            if "hard_cut" not in merged_types and camera_explained > 0.25:
                score -= min(0.38, camera_explained * 0.45)
            if cand.get("source") == "fade":
                score += 0.08

            prev_frame = int(sorted_candidates[idx - 1]["frame"]) if idx > 0 else 0
            next_frame = int(sorted_candidates[idx + 1]["frame"]) if idx + 1 < len(sorted_candidates) else total_frames
            left_len = frame - prev_frame
            right_len = next_frame - frame
            if left_len < min_scene_frames or right_len < min_scene_frames:
                short_ratio = min(left_len, right_len) / max(1, min_scene_frames)
                score -= min(0.34, 0.12 + (1.0 - short_ratio) * 0.24)

            threshold = params["accept"]
            if cand_type == "semantic" and len(merged_types) == 1:
                threshold = params["semantic"]
            elif cand_type == "gradual" and "hard_cut" not in merged_types:
                threshold = params["gradual"]
            if context_veto or camera_veto:
                threshold = max(threshold, score + 0.01)

            classified = dict(cand)
            classified["classifier_score"] = round(float(score), 4)
            classified["classifier_threshold"] = round(float(threshold), 4)
            classified["decision"] = "accepted" if score >= threshold else "rejected"

            if score >= threshold:
                accepted.append(classified)
            else:
                rejected.append(classified)

        return accepted, rejected

    def _merge_candidates(self, candidates, min_gap, total_frames):
        valid = [
            dict(c) for c in candidates
            if 0 < int(c.get("frame", 0)) < total_frames
        ]
        if not valid:
            return []

        priority = {"hard_cut": 3, "gradual": 2, "semantic": 1}
        valid.sort(key=lambda c: (
            -priority.get(c.get("type"), 0),
            -float(c.get("confidence", 0.0)),
            int(c["frame"])))

        merged = []
        for cand in valid:
            frame = int(cand["frame"])
            match_idx = None
            match_dist = None
            for idx, existing in enumerate(merged):
                dist = abs(frame - int(existing["frame"]))
                if dist <= min_gap and (match_dist is None or dist < match_dist):
                    match_idx = idx
                    match_dist = dist
            if match_idx is None:
                merged.append(cand)
                continue

            existing = merged[match_idx]
            cand_priority = priority.get(cand.get("type"), 0)
            existing_priority = priority.get(existing.get("type"), 0)
            if cand_priority > existing_priority:
                keeper, other = cand, existing
            elif cand_priority < existing_priority:
                keeper, other = existing, cand
            else:
                if cand.get("confidence", 0.0) > existing.get("confidence", 0.0):
                    keeper, other = cand, existing
                else:
                    keeper, other = existing, cand

            if keeper.get("type") == other.get("type") and abs(int(keeper["frame"]) - int(other["frame"])) <= min_gap:
                total_conf = max(1e-6, keeper.get("confidence", 0.0) + other.get("confidence", 0.0))
                keeper = dict(keeper)
                keeper["frame"] = int(round(
                    (int(keeper["frame"]) * keeper.get("confidence", 0.0) +
                     int(other["frame"]) * other.get("confidence", 0.0)) / total_conf))
            keeper = dict(keeper)
            keeper["score"] = max(float(keeper.get("score", 0.0)), float(other.get("score", 0.0)))
            keeper["confidence"] = max(float(keeper.get("confidence", 0.0)),
                                       float(other.get("confidence", 0.0)))
            keeper["merged_types"] = sorted(set(
                keeper.get("merged_types", [keeper.get("type")]) +
                other.get("merged_types", [other.get("type")])))
            merged[match_idx] = keeper

        return sorted(merged, key=lambda c: int(c["frame"]))

    def _scenes_from_boundaries(self, boundaries, total_frames, fps=None):
        points = [0] + [b for b in boundaries if 0 < b < total_frames] + [total_frames]
        scenes = []
        for i in range(len(points) - 1):
            if points[i + 1] > points[i]:
                scenes.append((points[i], points[i + 1]))
        if fps and len(scenes) > 1:
            min_tail_frames = max(2, int(round(float(fps) * 0.10)))
            tail_start, tail_end = scenes[-1]
            if tail_end - tail_start <= min_tail_frames:
                prev_start, _prev_end = scenes[-2]
                scenes[-2] = (prev_start, tail_end)
                scenes.pop()
        return scenes or [(0, total_frames)]

    def _fixed_interval(self):
        interval = self.cfg.get("FIXED_INTERVAL", 10)
        duration = self._get_video_duration()
        fps = self._get_video_fps()
        self._total_frames = self._get_video_total_frames(fps, duration)
        duration = max(self._total_frames / fps, 1.0)

        scenes, t = [], 0.0
        while t < duration:
            start_frame = int(round(t * fps))
            end_frame = min(self._total_frames, int(round(min(t + interval, duration) * fps)))
            if end_frame > start_frame:
                scenes.append((start_frame, end_frame))
            t += interval
        self.detected = len(scenes)
        return scenes

    def _record_cut_failure(self, task, error):
        idx, output_file, start_time, end_time, _aligned_start, _aligned_end, _start_frame, _end_frame, _precise = task
        message = str(error) or "Unknown error"
        failure = {
            "scene_number": int(idx),
            "file": os.path.basename(output_file),
            "start_time": round(float(start_time), 4),
            "end_time": round(float(end_time), 4),
            "duration": round(max(0.0, float(end_time) - float(start_time)), 4),
            "error_type": type(error).__name__,
            "error": message[:1000],
        }
        self._cut_failures.append(failure)
        self.failed = len(self._cut_failures)
        return failure

    def _write_cut_failure_report(self, outdir):
        if not self._cut_failures:
            return None
        path = os.path.join(outdir, "cut_errors.json")
        report = self._process_summary(outdir)
        report["results"] = {
            "failed_cuts": len(self._cut_failures),
            "failed_duration": round(sum(float(f.get("duration", 0.0)) for f in self._cut_failures), 4),
        }
        report["failed_scenes"] = self._cut_failures
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return path

    def _validate_cut_output(self, output, expected_duration):
        if not os.path.exists(output):
            raise RuntimeError("Output file was not created")
        if not is_valid_video_file(output):
            raise RuntimeError("Output file has no valid video stream")
        cmd = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", output]
        try:
            data = json.loads(check_output_hidden(cmd).decode(errors="ignore") or "{}")
            actual_duration = float((data.get("format") or {}).get("duration", 0.0))
        except Exception as e:
            raise RuntimeError(f"Could not validate output duration: {e}") from e
        bad_streams = [
            s.get("codec_type", "unknown")
            for s in data.get("streams", [])
            if s.get("codec_type") not in {"video", "audio"}
        ]
        if bad_streams:
            remove_temp_file(output)
            raise RuntimeError(f"Output contains unsupported stream(s): {', '.join(sorted(set(bad_streams)))}")
        if expected_duration > 0.2 and actual_duration < expected_duration * 0.5:
            remove_temp_file(output)
            raise RuntimeError(
                f"Output duration is too short ({actual_duration:.2f}s of {expected_duration:.2f}s)")
        if expected_duration > 0.2:
            max_duration = max(expected_duration + 3.0, expected_duration * 2.5)
            if actual_duration > max_duration:
                remove_temp_file(output)
                raise RuntimeError(
                    f"Output duration is too long ({actual_duration:.2f}s of {expected_duration:.2f}s)")
        return actual_duration

    def _cut_scenes(self, scenes):
        outdir = build_output_dir(
            self.output,
            mode="scene" if self.scene_mode else "interval",
            profile=self.cfg.get("label", "NA"),
            accel=self.cfg.get("ENCODER", "cpu")
        )
        self._cut_output_dir = outdir
        self._cut_failures = []
        self.failed = 0
        fps = self._get_video_fps()
        self.total = len(scenes)
        self.done = 0
        self.completed_attempts = 0
        self._cut_lock = threading.Lock()

        keyframes = self._get_keyframes() or [0.0]
        tasks = []
        for i, (start_frame, end_frame) in enumerate(scenes):
            start_time = start_frame / fps
            end_time = end_frame / fps
            output_file = os.path.join(outdir, f"scene_{i:03d}.mp4")
            aligned_start = self._nearest_keyframe_before(start_time, keyframes)
            aligned_end = self._nearest_keyframe_after(end_time, keyframes)
            start_delta = abs(start_time - aligned_start)
            end_delta = abs(end_time - aligned_end)
            needs_precise_cut = (
                    self.scene_mode or
                    start_delta > (1 / fps) or
                    end_delta > (1 / fps)
            )
            tasks.append((
                i, output_file, start_time, end_time,
                aligned_start, aligned_end, int(start_frame), int(end_frame),
                needs_precise_cut
            ))

        def _cut_one(task):
            idx, output_file, start_time, end_time, aligned_start, aligned_end, start_frame, end_frame, precise = task
            if precise:
                self._run_ffmpeg_precise_cut(
                    self.video, start_frame, end_frame, output_file, aligned_start)
            else:
                self._run_ffmpeg_copy(start_time, end_time, output_file)
            self._validate_cut_output(output_file, end_time - start_time)

        if self.total <= 1:
            try:
                _cut_one(tasks[0])
                self.done = 1
            except Exception as cut_error:
                if self._stop or str(cut_error) == "Stopped":
                    return
                failure = self._record_cut_failure(tasks[0], cut_error)
                if self.log:
                    err_msg = str(cut_error)[:80] or "Unknown error"
                    err_type = failure["error_type"]
                    self.log.after(0, lambda e=err_msg, t=err_type: self.log.append_message(
                        f"Error: scene_000 could not be exported [{t}] ({e})", kind="error"))
            self.completed_attempts = 1
            if self.log:
                self.log.after(0, lambda: self.log.write_status(detected=self.detected, cut=self.done, eta="00:00"))
            if self.progress:
                self.progress.after(0, lambda v=1.0: self.progress.update(v))
        else:
            with ThreadPoolExecutor(max_workers=min(MAX_CUT_WORKERS, self.total)) as pool:
                futures = {pool.submit(_cut_one, task): task for task in tasks}
                for future in as_completed(futures):
                    if self._stop:
                        pool.shutdown(wait=False)
                        return
                    try:
                        future.result()
                        with self._cut_lock:
                            self.done += 1
                    except Exception as cut_error:
                        if self._stop or str(cut_error) == "Stopped":
                            pool.shutdown(wait=False)
                            return
                        task = futures[future]
                        failure = self._record_cut_failure(task, cut_error)
                        print(f"[CUT ERROR] {cut_error}")
                        if self.log:
                            err_msg = str(cut_error)[:60]
                            scene_name = failure["file"]
                            err_type = failure["error_type"]
                            self.log.after(0, lambda n=scene_name, e=err_msg, t=err_type: self.log.append_message(
                                f"Warning: {n} could not be exported [{t}] ({e})", kind="warning"))

                    with self._cut_lock:
                        self.completed_attempts += 1
                        if self.progress:
                            ratio = self.completed_attempts / self.total
                            self.progress.after(0, lambda v=ratio: self.progress.update(v))
                        if self.log:
                            eta = self._calculate_eta()
                            self.log.after(0, lambda d=self.detected, c=self.done, e=eta:
                            self.log.write_status(d, c, e))

        self._end_time = time.time()
        self._write_scene_metadata(outdir, scenes, fps)
        self._write_cut_failure_report(outdir)

        if self.done <= 0 and self._cut_failures and not self._stop:
            raise RuntimeError("No scenes could be exported. See cut_errors.json for details.")

    def _write_scene_metadata(self, outdir, scenes, fps):
        try:
            metadata = self._process_summary(outdir)
            failed_cuts = len(getattr(self, "_cut_failures", []))
            metadata["results"] = {
                "status": "completed_with_errors" if failed_cuts else "completed",
                "total_cuts": len(scenes),
                "successful_cuts": int(self.done),
                "failed_cuts": failed_cuts,
            }
            if failed_cuts:
                metadata["results"]["error_report"] = "cut_errors.json"
            with open(os.path.join(outdir, "scenes.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Failed to write scene metadata: {e}")

    def _get_progress_ratio(self):
        if self.total > 0:
            return self.done / self.total
        return self._analysis_ratio

    def _calculate_eta(self):
        """Calculate ETA - only during cut phase, not during detection.

        Returns "--:--" during detection phase to avoid showing stale/misleading estimates.
        Shows real ETA based on actual cut rate during cutting phase.
        """
        elapsed = time.time() - self._start_time
        if elapsed < 1:
            return "--:--"

        completed = getattr(self, "completed_attempts", self.done)
        if self.total > 0 and completed > 0:
            rate = completed / elapsed
            remaining = self.total - completed
            eta_seconds = int(remaining / rate) if rate > 0 else 0
            if 0 < eta_seconds < 86400:
                m, s = divmod(eta_seconds, 60)
                h, m = divmod(m, 60)
                return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        return "--:--"

    def _get_video_fps(self):
        if self._fps is not None:
            return self._fps
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=avg_frame_rate",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        num, den = check_output_hidden(cmd).decode().strip().split("/")
        self._fps = float(num) / float(den)
        return self._fps

    def _get_video_total_frames(self, fps=None, duration=None):
        fps = float(fps or self._get_video_fps() or 0.0)
        explicit_counts = []
        duration_counts = []
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=nb_frames:stream_tags=NUMBER_OF_FRAMES,DURATION",
               "-of", "json", self.video]
        try:
            data = json.loads(check_output_hidden(cmd).decode(errors="ignore") or "{}")
            stream = (data.get("streams") or [{}])[0]
            for value in (stream.get("nb_frames"), stream.get("tags", {}).get("NUMBER_OF_FRAMES")):
                try:
                    count = int(str(value).strip())
                    if count > 0:
                        explicit_counts.append(count)
                except Exception:
                    pass
            tag_duration = stream.get("tags", {}).get("DURATION")
            if tag_duration and fps > 0:
                seconds = self._parse_duration_seconds(tag_duration)
                if seconds and seconds > 0:
                    duration_counts.append(int(round(seconds * fps)))
        except Exception:
            pass

        if duration and fps > 0:
            duration_counts.append(int(round(float(duration) * fps)))

        valid = [c for c in explicit_counts if c > 0]
        if valid:
            return min(valid)
        valid = [c for c in duration_counts if c > 0]
        return min(valid) if valid else max(1, int(fps or 1))

    def _parse_duration_seconds(self, value):
        try:
            text = str(value).strip()
            if not text:
                return None
            if ":" not in text:
                return float(text)
            parts = text.split(":")
            if len(parts) != 3:
                return None
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
        except Exception:
            return None

    def _map_threshold(self):
        """Returns (threshold, min_scene_len) for the ContentDetector.

        Hybrid approach: FIXED threshold per profile + ADAPTIVE min_dur.

        Pipeline:
          1. _adaptive_threshold() scans the video -> returns (raw_t, raw_d)
             - raw_t: pacing score (15-55, higher = more motion) - used only for Auto
             - raw_d: adaptive minimum scene duration (1-10s)

          2. Profile selects a FIXED threshold (PySceneDetect content_val scale):
             - Low    (42) -> conservative, clear scene/location changes
             - Normal (27) -> balanced, standard scene breaks
             - High   (18) -> sensitive, trailers/shorts/action

          3. Profile scales the adaptive min_dur:
             - Low    -> longer minimum scenes -> fewer false positives
             - Normal -> baseline
             - High   -> shorter minimum scenes -> catches fast cuts

        PySceneDetect ContentDetector constraints:
          - threshold: content_val scale (0-255), default=27.0
          - min_scene_len: in frames, default=15, minimum=1
        """
        raw_t, raw_d = _adaptive_threshold(self.video)

        if self.cfg.get("ADAPTIVE"):
            threshold = 10.0 + (raw_t - 15.0) * (35.0 / 40.0)
            threshold = max(10.0, min(50.0, threshold))
            threshold = round(threshold, 1)
            return threshold, raw_d

        profiles = {
            "Low": {"base_threshold": 42, "min_dur": 5.0, "dur_max_boost": 5.0, "label": "Low"},
            "Normal": {"base_threshold": 27, "min_dur": 2.0, "dur_max_boost": 4.0, "label": "Normal"},
            "High": {"base_threshold": 18, "min_dur": 0.7, "dur_max_boost": 2.5, "label": "High"},
        }

        profile = "Normal"
        for key, val in profiles.items():
            if self.cfg.get("label") == key:
                profile = key
                break

        cfg = profiles[profile]
        threshold = float(cfg["base_threshold"])

        base_dur = cfg["min_dur"]
        max_boost = cfg["dur_max_boost"]

        boost_ratio = (raw_d - 1.0) / 9.0
        min_dur = base_dur + boost_ratio * max_boost

        min_dur = max(0.5, min(15.0, min_dur))
        min_dur = round(min_dur, 1)

        if cfg.get("DEBUG", False):
            try:
                fps_est = self._get_video_fps()
            except Exception:
                fps_est = 30.0
            print(f"[DEBUG] Adaptive scan: raw_t={raw_t}, raw_d={raw_d}")
            print(
                f"[DEBUG] Profile={profile}, threshold={threshold}, min_dur_floor={cfg['min_dur']}, max_boost={cfg['dur_max_boost']}")
            print(f"[DEBUG] min_dur={min_dur}, min_scene_len={int(min_dur * fps_est)} frames (at ~{fps_est}fps)")

        return float(threshold), min_dur

    def _get_video_duration(self):
        if self._duration is not None:
            return self._duration
        cmd = ["ffprobe", "-v", "error",
               "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        self._duration = float(check_output_hidden(cmd).decode().strip())
        return self._duration

    def _get_keyframes(self):
        if self._keyframes_cache is not None:
            return self._keyframes_cache
        cmd = ["ffprobe", "-select_streams", "v", "-skip_frame", "nokey",
               "-show_entries", "frame=pkt_pts_time,best_effort_timestamp_time",
               "-loglevel", "error", "-of", "csv=p=0", self.video]
        result = run_hidden(cmd, capture_output=True, text=True)
        keyframes = []
        for line in result.stdout.splitlines():
            for part in line.replace(",", " ").split():
                try:
                    keyframes.append(float(part.strip()))
                    break
                except Exception:
                    continue
        keyframes = sorted(set(keyframes))
        self._keyframes_cache = keyframes
        return keyframes

    def _run_ffmpeg_tracked(self, cmd, timeout=300):
        if self._stop:
            return subprocess.CompletedProcess(cmd, -9, b"", b"Stopped")
        cmd = normalize_command(cmd)
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = CREATE_NO_WINDOW
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs
        )
        register_child_process(proc)
        self._ffmpeg_proc = proc
        with self._ffmpeg_proc_lock:
            self._ffmpeg_procs.add(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            terminate_process(proc)
            stdout, stderr = proc.communicate()
            return subprocess.CompletedProcess(cmd, -9, stdout, stderr)
        finally:
            unregister_child_process(proc)
            with self._ffmpeg_proc_lock:
                self._ffmpeg_procs.discard(proc)
            if self._ffmpeg_proc is proc:
                self._ffmpeg_proc = None

    def _ffmpeg_was_stopped(self, result):
        if self._stop:
            return True
        if not result:
            return False
        try:
            stderr = result.stderr.decode(errors="ignore") if isinstance(result.stderr, bytes) else str(result.stderr or "")
        except Exception:
            stderr = ""
        return result.returncode == -9 and "Stopped" in stderr

    def _run_ffmpeg_copy(self, start, end, output):
        duration = end - start
        cmd = ["ffmpeg", "-y", "-ss", f"{start:.6f}",
               "-i", self.video, "-t", f"{duration:.6f}",
               "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
               "-map_metadata", "-1", "-map_chapters", "-1",
               "-c", "copy", "-avoid_negative_ts", "make_zero",
               "-muxpreload", "0", "-muxdelay", "0", output]
        result = self._run_ffmpeg_tracked(cmd, timeout=300)
        if result.returncode != 0:
            if self._ffmpeg_was_stopped(result):
                raise RuntimeError("Stopped")
            raise RuntimeError(result.stderr.decode(errors="ignore")[:300] or "FFmpeg copy failed")

    def _has_audio_stream(self):
        if self._has_audio_cache is not None:
            return self._has_audio_cache
        cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
               "-show_entries", "stream=index",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        try:
            self._has_audio_cache = bool(check_output_hidden(cmd).decode().strip())
        except Exception:
            self._has_audio_cache = False
        return self._has_audio_cache

    def _run_ffmpeg_precise_cut(self, input_file, start_frame, end_frame, output, aligned_start):
        encoder = self.cfg.get("ENCODER", "cpu")
        bitrate = self._get_video_bitrate()
        fps = self._get_video_fps()
        start_time = start_frame / fps
        end_time = end_frame / fps
        local_start_frame = max(0, int(round((start_time - aligned_start) * fps)))
        local_end_frame = max(local_start_frame + 1, int(round((end_time - aligned_start) * fps)))
        local_audio_start = max(0.0, start_time - aligned_start)
        local_audio_end = max(local_audio_start + (1.0 / fps), end_time - aligned_start)

        if encoder == "cpu":
            codec = ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-tune", "film"]
        elif encoder == "nvidia":
            codec = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19",
                     "-rc", "vbr", "-b:v", "0", "-spatial_aq", "1",
                     "-temporal_aq", "1", "-rc-lookahead", "20"]
        elif encoder == "amd":
            codec = ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "vbr_peak"]
        elif encoder == "intel":
            codec = ["-c:v", "h264_qsv", "-preset", "medium", "-global_quality", "20"]
        elif encoder == "apple":
            codec = ["-c:v", "h264_videotoolbox", "-q:v", "65"]
        else:
            codec = ["-c:v", "libx264", "-crf", "16"]

        rate_opts = []
        if bitrate and encoder == "cpu":
            rate_opts = ["-maxrate", str(bitrate), "-bufsize", str(bitrate * 2)]

        vf = f"trim=start_frame={local_start_frame}:end_frame={local_end_frame},setpts=PTS-STARTPTS"
        if self._has_audio_stream():
            filter_complex = (
                f"[0:v]{vf}[v];"
                f"[0:a]atrim=start={local_audio_start:.6f}:end={local_audio_end:.6f},"
                f"asetpts=PTS-STARTPTS[a]"
            )
            maps = ["-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]"]
            audio_opts = ["-c:a", "aac", "-b:a", "192k"]
        else:
            maps = ["-vf", vf]
            audio_opts = ["-an"]

        cmd = ["ffmpeg", "-y", "-ss", f"{aligned_start:.6f}", "-i", input_file,
               *maps, *codec, *rate_opts,
               "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
               "-pix_fmt", "yuv420p", *audio_opts,
               "-movflags", "+faststart", output]

        result = self._run_ffmpeg_tracked(cmd, timeout=300)

        if self._ffmpeg_was_stopped(result):
            raise RuntimeError("Stopped")

        if result.returncode != 0 and encoder != "cpu":
            codec_fallback = ["-c:v", "libx264", "-crf", "16", "-preset", "medium"]
            cmd = ["ffmpeg", "-y", "-ss", f"{aligned_start:.6f}", "-i", input_file,
                   *maps, *codec_fallback,
                   "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
                   "-pix_fmt", "yuv420p", *audio_opts,
                   "-movflags", "+faststart", output]
            result2 = self._run_ffmpeg_tracked(cmd, timeout=300)
            if self._ffmpeg_was_stopped(result2):
                raise RuntimeError("Stopped")
            if result2.returncode == 0:
                return
            print("FFMPEG ERROR:\n", result.stderr.decode(errors="ignore"))
            if self.log:
                err_msg = result.stderr.decode(errors="ignore")[:80]
                self.log.after(0, lambda e=err_msg: self.log.append_message(
                    f"FFmpeg error: {e}", kind="error"))
            raise RuntimeError(result2.stderr.decode(errors="ignore")[:300] or "FFmpeg fallback failed")
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="ignore")[:300] or "FFmpeg precise cut failed")

    def _nearest_keyframe_before(self, t, keyframes):
        idx = bisect.bisect_right(keyframes, t)
        return keyframes[idx - 1] if idx > 0 else 0.0

    def _nearest_keyframe_after(self, t, keyframes):
        if not keyframes:
            return t
        idx = bisect.bisect_left(keyframes, t)
        if idx < len(keyframes):
            return keyframes[idx]
        return min(keyframes[-1], t)

    def _get_video_bitrate(self):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=bit_rate",
               "-of", "default=noprint_wrappers=1:nokey=1", self.video]
        try:
            return int(check_output_hidden(cmd).decode().strip())
        except Exception:
            return None

    def _preview_loop(self):
        while not self._preview_stop:
            if self.previewer and self.preview_enabled:
                now = time.time()
                if now - self.last_preview >= PREVIEW_INTERVAL:
                    try:
                        if self._analysis_mosaic:
                            time.sleep(0.05)
                            continue
                        if self.total <= 0:
                            time.sleep(0.05)
                            continue
                        ratio = self._get_progress_ratio()
                        ratio = max(0.0, min(1.0, ratio))
                        if abs(ratio - self._last_preview_ratio) < 0.02:
                            time.sleep(0.02)
                            continue
                        self._last_preview_ratio = ratio
                        duration = self._get_video_duration()
                        t = ratio * duration
                        img = self._get_preview_frame_at(t)
                        if img and self._ui_alive:
                            self.previewer.after(
                                0, lambda img=img: (
                                    self.previewer.update_image(img)
                                    if self._ui_alive else None))
                    except Exception:
                        pass
                    self.last_preview = now
            time.sleep(0.05)

    def _get_preview_frame_at(self, t):
        try:
            with self._preview_lock:
                if not self._preview_cap:
                    return None
                self._preview_cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = self._preview_cap.read()
            if not ret or frame is None:
                return None
            return resize_for_preview(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        except Exception:
            return None
