from .shared import *
from .widgets import *
from .scene_engine import SceneEngine
from .face_engine import FaceDetectionEngine

class ScenespyApp(ctk.CTk):
    """Main window that coordinates input selection, processing, and progress updates."""
    def __init__(self):
        super().__init__()
        self.title("Scenespy")
        self.geometry("1000x650")
        self.engine = None
        self.running = False
        self.stop_pending = False
        self.batch_stop = False
        self.resizable(False, False)
        self.preview_enabled = ENABLE_PREVIEW_DEFAULT
        self.available_encoder_accel = detect_available_encoder_accel()
        self.available_inference_accel = detect_available_inference_accel()
        self.available_accel = self.available_encoder_accel | self.available_inference_accel
        self.mode_requirements = self._check_mode_requirements()

        self.saved_settings = load_settings()

        self._build_ui()

    def _check_mode_requirements(self):
        model_path = os.path.join(APP_DIR, "models", "yolov8n-face.pt")
        requirements = {
            "scene": _missing_modules("scenedetect", "av") + _missing_executables("ffmpeg", "ffprobe"),
            "interval": _missing_executables("ffmpeg", "ffprobe"),
            "faces": _missing_modules("torch", "ultralytics", "mediapipe") + _missing_executables("ffmpeg", "ffprobe"),
        }
        if not os.path.isfile(model_path):
            requirements["faces"].append("models/yolov8n-face.pt")
        return requirements

    def _mode_label(self, mode):
        return {
            "scene": "Scene detection",
            "interval": "Every seconds",
            "faces": "Detect faces",
        }.get(mode, mode)

    def _missing_for_mode(self, mode):
        return list(self.mode_requirements.get(mode, []))

    def _mode_requirement_message(self, mode):
        missing = self._missing_for_mode(mode)
        if not missing:
            return None
        return f"{self._mode_label(mode)}: install missing requirements: {', '.join(missing)}."

    def _show_mode_requirement_message(self, mode):
        message = self._mode_requirement_message(mode)
        if message and hasattr(self, "log"):
            self.log.show_message(message)
        return message

    def _build_ui(self):
        self.left = ctk.CTkFrame(
            self, width=420, fg_color=BG_PANEL, border_width=1,
            border_color=BORDER_SOFT2, corner_radius=0)
        self.left.pack(side="left", fill="y", padx=10, pady=10)

        files = Section(self.left, "Files")
        files.pack(fill="x", padx=12, pady=12)
        self.video_selector = FileSelector(files, "Source video(s)")
        self.video_selector.pack(fill="x", padx=12)
        self.output_selector = DirectorySelector(files, "Output folder")
        self.output_selector.pack(fill="x", padx=12)

        last_video = self.saved_settings.get("last_video", "")
        last_output = self.saved_settings.get("last_output", "")
        if last_video:
            self.video_selector.entry.insert(0, last_video)
            self.video_selector.paths = [last_video]
        if last_output:
            self.output_selector.entry.insert(0, last_output)

        mode = Section(self.left, "Cut Mode")
        mode.pack(fill="x", padx=12)

        self.cut_mode = ctk.StringVar(value="scene")
        self.cut_mode.trace_add("write", self._on_cut_mode_change)

        group = RadioGroup(mode, self.cut_mode, [
            ("Scene detection", "scene"),
            ("Every seconds", "interval"),
            ("Detect faces", "faces"),
        ], radio_width=150)
        group.pack(fill="x", padx=12, pady=(0, 6))
        self.mode_radios = group.radios

        vcmd = (self.register(self._validate_interval), "%P")
        self.interval_entry = ctk.CTkEntry(
            mode, height=25, width=90, fg_color=BG_MAIN,
            border_color=BORDER_SOFT, border_width=1, corner_radius=15,
            text_color="#ededed", font=ui_font(11),
            placeholder_text_color=TEXT_MUTED, placeholder_text="Seconds",
            validate="key", validatecommand=vcmd)
        self.interval_entry.pack(anchor="w", padx=(186, 0), pady=(0, 8))

        profile_sec = Section(self.left, "Detection Sensitivity")
        profile_sec.pack(fill="x", padx=12, pady=5)

        self.profile = ctk.StringVar(value="Normal")
        options = [(cfg["label"], key) for key, cfg in PROFILES.items()]
        group2 = RadioGroup(profile_sec, self.profile, options, radio_width=110)
        group2.pack(fill="x", padx=12)
        self.profile_radios = group2.radios

        accel_sec = Section(self.left, "Hardware Acceleration")
        accel_sec.pack(fill="x", padx=12)
        self.accel = ctk.StringVar(value="cpu")
        group3 = RadioGroup(accel_sec, self.accel,
                            [(val.upper(), val) for val in ACCEL_OPTIONS],
                            radio_width=110)
        group3.pack(fill="x", padx=12)
        self.accel_radios = group3.radios
        self.update_accel_radios()

        self.start_btn = ctk.CTkButton(
            self.left, text="Start", height=13, corner_radius=420,
            fg_color=ACCENT, hover_color="#4f46e5", text_color="white",
            command=self.toggle_start)
        self.start_btn.pack(pady=(20, 10))

        self.log = LogBox(self.left, height=220)
        self.log.pack(fill="x", padx=10, pady=10)

        self.right = ctk.CTkFrame(
            self, width=250, fg_color=BG_PANEL, border_width=1,
            border_color=BORDER_SOFT2, corner_radius=15)
        self.right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        section = Section(self.right, "Preview")
        section.pack(padx=12, pady=12, fill="x")
        self.preview_switch = ctk.CTkSwitch(
            section, text="Show Thumbnail", command=self.toggle_preview)
        self.preview_switch.pack(anchor="e", padx=10, pady=8)

        if self.preview_enabled:
            self.preview_switch.select()
            self.toggle_preview()

        self.preview_frame = PreviewFrame(self.right)
        self.preview_frame.set_enabled(self.preview_enabled)
        self.preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.progress = ProgressBar(self.right)
        self.progress.pack(fill="x", padx=20, pady=10)

        self.left.pack_propagate(False)
        self.right.pack_propagate(False)
        self._on_cut_mode_change()

    def toggle_preview(self):
        self.preview_enabled = self.preview_switch.get()
        if hasattr(self, "preview_frame"):
            self.preview_frame.set_enabled(self.preview_enabled)
            if self.running and not self.preview_enabled:
                self.preview_frame.show_loading()
        if not self.preview_enabled:
            if self.engine and hasattr(self.engine, "disable_preview"):
                self.engine.disable_preview()

    def toggle_start(self):
        if self.running:
            self.confirm_stop()
        else:
            if self.engine is not None:
                return
            self.start_process()

    def start_process(self):
        videos = self.video_selector.get_paths()
        output = self.output_selector.get().strip()
        first_video = videos[0] if videos else ""

        save_settings(video=first_video, output=output)

        if not videos and not output:
            return
        if not videos or not output:
            self.log.show_message("Select input video and output folder")
            return
        if not os.path.isdir(output):
            self.log.show_message("Invalid output folder")
            return

        valid_videos = []
        skipped = []
        for video in videos:
            name_no_ext = os.path.splitext(os.path.basename(video))[0].lower()
            if name_no_ext.endswith("_fixed") or "_fixed_fixed" in name_no_ext:
                skipped.append((video, "Temporary repaired file"))
                continue
            ext = os.path.splitext(video)[1].lower()
            if ext not in ALLOWED_VIDEO_EXTENSIONS:
                skipped.append((video, "Unsupported file type"))
                continue
            if not os.path.isfile(video):
                skipped.append((video, "File does not exist"))
                continue
            if not is_valid_video_file(video):
                skipped.append((video, "Invalid or unsupported video file"))
                continue
            valid_videos.append(video)

        self.cleanup_process(reason="reset")
        if self.log:
            self.log.clear_status()
            if len(videos) > 1:
                self.log.append_message(
                    f"Queue: {len(valid_videos)} videos ready, {len(skipped)} skipped",
                    kind="info")
            for video, reason in skipped[:8]:
                self.log.append_message(
                    f"Skipped: {os.path.basename(video)} ({reason})", kind="warning")
            if len(skipped) > 8:
                self.log.append_message(
                    f"Skipped: {len(skipped) - 8} more invalid file(s).", kind="warning")

        if not valid_videos:
            if self.log:
                self.log.append_message("Error: no valid videos to process.", kind="error")
            return

        cfg = PROFILES[self.profile.get()].copy()
        requested = self.accel.get()
        mode = self.cut_mode.get()
        compat = MODE_ACCEL_COMPAT.get(mode, {})

        encoder_allowed = compat.get("encoder", {"cpu"}) & self.available_encoder_accel
        inference_allowed = compat.get("inference", {"cpu"}) & self.available_inference_accel

        encoder = requested if requested in encoder_allowed else "cpu"
        inference = requested if requested in inference_allowed else "cpu"

        cfg["ENCODER"] = encoder
        cfg["INFERENCE"] = inference
        cfg["DEBUG"] = DEBUG

        scene_mode = mode == "scene"
        missing_message = self._mode_requirement_message(mode)
        if missing_message:
            self.log.show_message(missing_message)
            return

        if mode == "interval":
            value = self.interval_entry.get()
            if not value:
                self.log.show_message("Interval cannot be empty!")
                self.reset_ui()
                return
            cfg["FIXED_INTERVAL"] = int(value)

        self.running = True
        self.batch_stop = False
        self.log.set_mode(mode)
        self.set_ui_state(True)
        self.after(0, self._finalize_start_ui)

        if self.preview_frame:
            self.after(0, self.preview_frame.show_loading)

        threading.Thread(
            target=self.run_batch,
            args=(valid_videos, output, cfg, mode, scene_mode, inference),
            daemon=True
        ).start()

    def stop_process(self):
        self.batch_stop = True
        self.stop_pending = True
        if not self.engine:
            return
        self.engine.stop()

    def _cooldown_between_videos(self):
        gc.collect()
        time.sleep(PROCESS_COOLDOWN_SECONDS)
        gc.collect()

    def run_batch(self, videos, output, cfg, mode, scene_mode, inference):
        started = time.time()
        completed = 0
        failed = 0
        partial = 0
        stopped = False

        try:
            for index, video in enumerate(videos, start=1):
                if self.batch_stop or self.stop_pending:
                    stopped = True
                    break

                basename = os.path.basename(video)
                if self.log and len(videos) > 1:
                    self.after(0, lambda i=index, n=len(videos), b=basename: self.log.append_message(
                        f"Processing {i}/{n}: {b}", kind="info"))
                if self.progress:
                    self.after(0, self.progress.reset)

                result = False
                error_msg = None
                error_type = None
                warning_msg = None
                temp_files = []
                engine = None
                try:
                    current_video = prepare_video_for_processing(video, temp_files=temp_files)
                    if mode == "faces":
                        engine = FaceDetectionEngine(
                            current_video, output, logbox=self.log, progressbar=self.progress,
                            previewer=self.preview_frame, profile=self.profile.get(),
                            accel=inference, preview_enabled=self.preview_enabled)
                        self.engine = engine
                        result = engine.run()
                    else:
                        engine = SceneEngine(
                            current_video, output, cfg.copy(), logbox=self.log, progressbar=self.progress,
                            previewer=self.preview_frame, preview_enabled=self.preview_enabled,
                            preview_pool=[current_video])
                        for temp_file in temp_files:
                            engine.add_temp_file(temp_file)
                        self.engine = engine
                        result = engine.run(scene_mode=scene_mode)

                    failed_cuts = int(getattr(engine, "failed", 0))
                    if result and failed_cuts:
                        partial += 1
                        warning_msg = (
                            f"Warning: {basename} finished with {failed_cuts} failed scene(s). "
                            "See cut_errors.json in its output folder."
                        )
                    if result:
                        completed += 1
                    elif not self.batch_stop:
                        failed += 1
                        error_msg = "the video could not be processed"
                        error_type = "ProcessingError"
                    else:
                        stopped = True
                        break
                except Exception as e:
                    failed += 1
                    error_msg = str(e) or "Unknown error"
                    error_type = type(e).__name__
                    print("Batch item error:", e)
                finally:
                    if warning_msg and self.log:
                        self.after(0, lambda m=warning_msg: self.log.append_message(m, kind="warning"))
                    if error_msg and self.log:
                        self.after(0, lambda b=basename, t=error_type, e=error_msg: self.log.append_message(
                            f"Error processing {b} [{t}]: {e}", kind="error"))
                    self.engine = None
                    engine = None
                    gc.collect()
                    for temp_file in temp_files:
                        remove_temp_file(temp_file)
                    if not self.batch_stop and not self.stop_pending and index < len(videos):
                        if self.progress:
                            self.after(0, self.progress.reset)
                        self._cooldown_between_videos()

            elapsed = max(1, int(time.time() - started))
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            total_time = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
            summary = (
                f"Queue finished: {completed}/{len(videos)} video(s) processed, "
                f"{failed} failed"
            )
            if partial:
                summary += f", {partial} with failed scene(s)"
            summary += "."
            final_warning = summary if (failed or partial) else None

            self.after(0, lambda: self.reset_ui(
                finished=completed > 0 and not stopped,
                total_time=total_time,
                stopped=stopped,
                warning_message=final_warning if not stopped else None))
            if completed <= 0 and not stopped and self.log:
                self.after(0, lambda: self.log.append_message(summary, kind="error"))
        finally:
            self.engine = None
            self.batch_stop = False
            self.stop_pending = False

    def reset_ui(self, finished=False, total_time=None, stopped=False, warning_message=None):
        self.stop_pending = False
        self.running = False
        if self.preview_frame:
            self.preview_frame.hide_loading()
        self.start_btn.configure(text="Start", fg_color=ACCENT,
                                 hover_color="#4f46e5", state="normal")
        self.set_ui_state(False)
        if stopped:
            self.cleanup_process(reason="stop")
        elif finished:
            self.cleanup_process(reason="finish", total_time=total_time,
                                 warning_message=warning_message)

    def set_ui_state(self, disabled):
        state = "disabled" if disabled else "normal"
        for widget in [self.video_selector.button, self.output_selector.button,
                       *self.mode_radios, *self.accel_radios]:
            widget.configure(state=state)
        self.video_selector.entry.configure(state=state)
        self.output_selector.entry.configure(state=state)

        if disabled:
            for rb in self.profile_radios:
                rb.configure(state="disabled")
        else:
            for rb in self.profile_radios:
                if self.cut_mode.get() == "faces" and rb.cget("value") == "Auto":
                    rb.configure(state="disabled")
                else:
                    rb.configure(state="normal")

        if self.cut_mode.get() == "interval":
            self.interval_entry.configure(state=state)
        self.preview_switch.configure(state="disabled" if self.running else "normal")
        self.update_accel_radios()

    def _on_cut_mode_change(self, *args):
        mode = self.cut_mode.get()
        if hasattr(self, "log"):
            self.log.set_mode(mode)

        if mode == "interval":
            self.interval_entry.pack(anchor="w", padx=(186, 0), pady=(0, 8))
            self.interval_entry.configure(state="disabled" if self.running else "normal")
        else:
            self.interval_entry.pack_forget()

        if mode == "interval":
            for rb in self.profile_radios:
                rb.configure(state="disabled")
        elif mode == "faces":
            if self.profile.get() == "Auto":
                self.profile.set("Normal")
            state = "disabled" if self.running else "normal"
            for rb in self.profile_radios:
                if rb.cget("value") == "Auto":
                    rb.configure(state="disabled")
                else:
                    rb.configure(state=state)
        else:
            state = "disabled" if self.running else "normal"
            for rb in self.profile_radios:
                rb.configure(state=state)

        self.update_accel_radios()
        if not self.running:
            self._show_mode_requirement_message(mode)

    def update_accel_radios(self):
        mode = self.cut_mode.get()
        compat = MODE_ACCEL_COMPAT.get(mode, {})
        enabled = (
                (compat.get("encoder", set()) & self.available_encoder_accel) |
                (compat.get("inference", set()) & self.available_inference_accel) |
                {"cpu"}
        )
        for rb in self.accel_radios:
            value = rb.cget("value")
            state = "disabled" if self.running or value not in enabled else "normal"
            rb.configure(state=state)
        if self.accel.get() not in enabled:
            self.accel.set("cpu")

    def cleanup_process(self, reason="reset", total_time=None, warning_message=None):
        if self.preview_frame and reason in ("stop", "finish"):
            self.preview_frame.clear_all()
        if self.progress and reason in ("stop", "reset"):
            self.after(0, self.progress.reset)
        elif self.progress and reason == "finish":
            self.after(0, self.progress.mark_finished)
        if self.log:
            if reason == "stop":
                self.log.show_message("Process stopped")
            elif reason == "finish":
                msg = "Process finished"
                if total_time:
                    msg += f" {total_time}"
                self.log.write_finished(msg)
                if warning_message:
                    self.log.append_message(warning_message, kind="warning")
        self.after(1000, gc.collect)

    def confirm_stop(self):
        if not self.running or self.stop_pending:
            return
        answer = mb.askyesno("Confirm Stop",
                             "The process is still running.\nDo you really want to stop it?")
        if not answer:
            return
        self.stop_pending = True
        self.after(0, self.stop_process)

    def _validate_interval(self, value: str) -> bool:
        if value == "":
            return True
        if not value.isdigit():
            return False
        return 1 <= int(value) <= 18000

    def _finalize_start_ui(self):
        self.start_btn.configure(text="Stop ", fg_color=DANGER, hover_color="#dc2626")
