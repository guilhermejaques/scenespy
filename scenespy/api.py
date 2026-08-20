import os
from numbers import Integral
from pathlib import Path

from .face_engine import FaceDetectionEngine
from .scene_engine import SceneEngine
from .shared import (
    ALLOWED_VIDEO_EXTENSIONS,
    DEBUG,
    MODE_ACCEL_COMPAT,
    MODEL_FILE,
    PROFILES,
    detect_available_encoder_accel,
    detect_available_inference_accel,
    face_detection_required_modules,
    format_status_lines,
    is_valid_video_file,
    prepare_video_for_processing,
    remove_temp_file,
    validate_runtime_dependencies,
    _missing_modules,
)


_MODES = {"scene", "interval", "faces"}
_ACCELERATORS = {"cpu", "nvidia", "amd", "intel", "apple"}
_SENSITIVITIES = {name.lower(): name for name in PROFILES}


class _ProgressAdapter:
    def __init__(self, callback, video):
        self.callback = callback
        self.video = video
        self.value = 0.0
        self.status = ""

    def after(self, _delay, callback):
        callback()

    def _emit(self, **values):
        if self.callback:
            event = {
                "video": self.video,
                "progress": self.value,
                "status": self.status,
            }
            event.update(values)
            self.callback(event)

    def set_status(self, text):
        self.status = str(text)
        self._emit()

    def update(self, value):
        self.value = max(0.0, min(1.0, float(value)))
        self._emit()

    def reset(self):
        self.value = 0.0
        self._emit()


class _LogAdapter:
    def __init__(self, progress):
        self.progress = progress

    def after(self, _delay, callback):
        callback()

    def write_status(self, detected=0, cut=0, eta="--:--"):
        self.progress._emit(detected=int(detected), saved=int(cut), eta=str(eta))

    def append_message(self, message, kind="info"):
        self.progress._emit(message=str(message), kind=str(kind))


class _ConsoleReporter:
    def __init__(self, mode):
        self.mode = mode
        self.last_status = None
        self.last_lines = None
        self.last_percent = None

    def __call__(self, event):
        status = event.get("status")
        if status and status != self.last_status:
            print(status)
            self.last_status = status
        percent = int(float(event.get("progress", 0.0)) * 100)
        if percent != self.last_percent:
            print(f"{percent}%")
            self.last_percent = percent
        if "detected" in event or "saved" in event or "eta" in event:
            lines = format_status_lines(
                self.mode,
                detected=event.get("detected"),
                cut=event.get("saved"),
                eta=event.get("eta"),
            )
            if lines != self.last_lines:
                print("\n".join(lines))
                self.last_lines = lines
        if event.get("message"):
            print(event["message"])


def _normalize_path(value, name):
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be a path-like value")
    raw_path = os.fspath(value)
    if not isinstance(raw_path, str):
        raise TypeError(f"{name} must resolve to a text path")
    if not raw_path.strip():
        raise ValueError(f"{name} cannot be empty")
    path = os.path.abspath(os.path.expanduser(raw_path))
    return path


def _normalize_mode(mode):
    if not isinstance(mode, str):
        raise TypeError("mode must be text")
    value = mode.strip().lower()
    aliases = {
        "scenes": "scene",
        "detect_scenes": "scene",
        "split": "interval",
        "every_seconds": "interval",
        "face": "faces",
        "extract_faces": "faces",
    }
    value = aliases.get(value, value)
    if value not in _MODES:
        raise ValueError("mode must be 'scene', 'interval', or 'faces'")
    return value


def _normalize_sensitivity(sensitivity, mode):
    if not isinstance(sensitivity, str):
        raise TypeError("sensitivity must be text")
    value = sensitivity.strip().lower()
    if value not in _SENSITIVITIES:
        raise ValueError("sensitivity must be 'Low', 'Normal', 'High', or 'Auto'")
    normalized = _SENSITIVITIES[value]
    if mode == "faces" and normalized == "Auto":
        raise ValueError("Auto sensitivity is not available for face detection")
    return normalized


def _select_acceleration(mode, accelerator):
    if not isinstance(accelerator, str):
        raise TypeError("accelerator must be text")
    requested = accelerator.strip().lower()
    if requested not in _ACCELERATORS:
        raise ValueError("accelerator must be 'cpu', 'nvidia', 'amd', 'intel', or 'apple'")
    compat = MODE_ACCEL_COMPAT[mode]
    encoder_available = detect_available_encoder_accel()
    inference_available = detect_available_inference_accel()
    encoder_allowed = compat.get("encoder", {"cpu"}) & encoder_available
    inference_allowed = compat.get("inference", {"cpu"}) & inference_available
    encoder = requested if requested in encoder_allowed else "cpu"
    inference = requested if requested in inference_allowed else "cpu"
    return encoder, inference


def _validate_input(video, output, mode):
    video_path = _normalize_path(video, "video")
    output_path = _normalize_path(output, "output")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")
    name_without_extension = Path(video_path).stem.lower()
    if name_without_extension.endswith("_fixed") or "_fixed_fixed" in name_without_extension:
        raise ValueError(f"temporary repaired videos cannot be processed: {video_path}")
    if Path(video_path).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported video type: {Path(video_path).suffix or '<none>'}")
    if os.path.exists(output_path) and not os.path.isdir(output_path):
        raise NotADirectoryError(f"invalid output folder: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    if not os.path.isdir(output_path):
        raise NotADirectoryError(f"invalid output folder: {output_path}")
    runtime_ok, runtime_message = validate_runtime_dependencies()
    if not runtime_ok:
        raise RuntimeError(runtime_message)
    required = ["scenedetect", "av"] if mode == "scene" else []
    if mode == "faces":
        required = list(face_detection_required_modules())
        if not os.path.isfile(MODEL_FILE):
            raise RuntimeError(f"face model not found: {MODEL_FILE}")
    missing = _missing_modules(*required)
    if missing:
        raise RuntimeError(f"missing dependencies for {mode}: {', '.join(missing)}")
    if not is_valid_video_file(video_path):
        raise ValueError(f"invalid or unsupported video: {video_path}")
    return video_path, output_path


def process_video(video, output, mode="scene", sensitivity="Normal",
                  accelerator="cpu", interval=10, progress=None, verbose=True):
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be True or False")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    normalized_mode = _normalize_mode(mode)
    normalized_sensitivity = _normalize_sensitivity(sensitivity, normalized_mode)
    if normalized_mode == "interval":
        if isinstance(interval, bool) or not isinstance(interval, Integral):
            raise TypeError("interval must be an integer number of seconds")
        interval = int(interval)
        if not 1 <= interval <= 18000:
            raise ValueError("interval must be between 1 and 18000 seconds")
    video_path, output_path = _validate_input(video, output, normalized_mode)
    encoder, inference = _select_acceleration(normalized_mode, accelerator)
    reporter = progress or (_ConsoleReporter(normalized_mode) if verbose else None)
    progress_adapter = _ProgressAdapter(reporter, video_path) if reporter else None
    log_adapter = _LogAdapter(progress_adapter) if progress_adapter else None
    temp_files = []
    engine = None
    try:
        prepared_video = prepare_video_for_processing(video_path, temp_files=temp_files)
        if normalized_mode == "faces":
            engine = FaceDetectionEngine(
                prepared_video,
                output_path,
                logbox=log_adapter,
                progressbar=progress_adapter,
                previewer=None,
                profile=normalized_sensitivity,
                accel=inference,
                preview_enabled=False,
            )
            success = engine.run()
            result_output = engine._output_dir
        else:
            config = PROFILES[normalized_sensitivity].copy()
            config["ENCODER"] = encoder
            config["INFERENCE"] = inference
            config["DEBUG"] = DEBUG
            if normalized_mode == "interval":
                config["FIXED_INTERVAL"] = interval
            engine = SceneEngine(
                prepared_video,
                output_path,
                config,
                logbox=log_adapter,
                progressbar=progress_adapter,
                previewer=None,
                preview_enabled=False,
            )
            success = engine.run(scene_mode=normalized_mode == "scene")
            result_output = engine._cut_output_dir
        result = {
            "success": bool(success),
            "video": video_path,
            "output_dir": result_output,
            "mode": normalized_mode,
            "sensitivity": normalized_sensitivity,
            "accelerator": engine.accel if normalized_mode == "faces" else encoder,
            "detected": int(getattr(engine, "detected", 0)),
            "saved": int(getattr(engine, "done", 0)),
            "failed": int(getattr(engine, "failed", 0)),
            "elapsed": engine.total_time(),
        }
        if progress_adapter:
            progress_adapter.value = 1.0 if success else progress_adapter.value
            progress_adapter._emit(result=result)
        if verbose:
            print(f"Process finished {result['elapsed']}")
        return result
    finally:
        if engine is not None:
            engine.cleanup_temp_files() if hasattr(engine, "cleanup_temp_files") else None
        for temp_file in temp_files:
            remove_temp_file(temp_file)


def detect_scenes(video, output, sensitivity="Normal", accelerator="cpu",
                  progress=None, verbose=True):
    return process_video(
        video,
        output,
        mode="scene",
        sensitivity=sensitivity,
        accelerator=accelerator,
        progress=progress,
        verbose=verbose,
    )


def split_video(video, output, interval=10, accelerator="cpu",
                progress=None, verbose=True):
    return process_video(
        video,
        output,
        mode="interval",
        sensitivity="Normal",
        accelerator=accelerator,
        interval=interval,
        progress=progress,
        verbose=verbose,
    )


def extract_faces(video, output, sensitivity="Normal", accelerator="cpu",
                  progress=None, verbose=True):
    return process_video(
        video,
        output,
        mode="faces",
        sensitivity=sensitivity,
        accelerator=accelerator,
        progress=progress,
        verbose=verbose,
    )


def process_videos(videos, output, mode="scene", sensitivity="Normal",
                   accelerator="cpu", interval=10, progress=None,
                   continue_on_error=True, verbose=True):
    if not isinstance(continue_on_error, bool):
        raise TypeError("continue_on_error must be True or False")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be True or False")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    normalized_mode = _normalize_mode(mode)
    normalized_sensitivity = _normalize_sensitivity(sensitivity, normalized_mode)
    if not isinstance(accelerator, str):
        raise TypeError("accelerator must be text")
    if accelerator.strip().lower() not in _ACCELERATORS:
        raise ValueError("accelerator must be 'cpu', 'nvidia', 'amd', 'intel', or 'apple'")
    if normalized_mode == "interval":
        if isinstance(interval, bool) or not isinstance(interval, Integral):
            raise TypeError("interval must be an integer number of seconds")
        interval = int(interval)
        if not 1 <= interval <= 18000:
            raise ValueError("interval must be between 1 and 18000 seconds")
    if isinstance(videos, (str, os.PathLike)):
        video_list = [videos]
    else:
        try:
            video_list = list(videos)
        except TypeError as exc:
            raise TypeError("videos must be a path or an iterable of paths") from exc
    if not video_list:
        raise ValueError("videos cannot be empty")
    results = []
    if verbose and len(video_list) > 1:
        print(f"Queue: {len(video_list)} videos ready, 0 skipped")
    for index, video in enumerate(video_list, start=1):
        if verbose and len(video_list) > 1:
            print(f"Processing {index}/{len(video_list)}: {os.path.basename(os.fspath(video))}")
        try:
            results.append(process_video(
                video,
                output,
                mode=normalized_mode,
                sensitivity=normalized_sensitivity,
                accelerator=accelerator,
                interval=interval,
                progress=progress,
                verbose=verbose,
            ))
        except Exception as exc:
            if not continue_on_error:
                raise
            try:
                failed_video = _normalize_path(video, "video")
            except Exception:
                failed_video = str(video)
            results.append({
                "success": False,
                "video": failed_video,
                "output_dir": None,
                "mode": normalized_mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            if verbose:
                print(f"Error processing {os.path.basename(failed_video)} "
                      f"[{type(exc).__name__}]: {exc}")
    if verbose and len(video_list) > 1:
        completed = sum(1 for result in results if result.get("success"))
        failed = len(results) - completed
        print(f"Queue finished: {completed}/{len(video_list)} video(s) processed, {failed} failed.")
    return results
