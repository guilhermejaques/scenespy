from .shared import *
from . import shared as shared

# FaceDetectionEngine (heavy deps loaded lazily)
# ============================================================
class FaceDetectionEngine:
    def __init__(self, video, output, logbox=None, progressbar=None,
                 previewer=None, profile="Normal", accel="cpu", preview_enabled=True):
        self.video = video
        self.output = output
        self.log = logbox
        self.progress = progressbar
        self.previewer = previewer
        self.preview_enabled = preview_enabled
        self._ui_alive = True
        self._last_eta_update = 0  # FIX: Initialize for ETA throttling

        self.profile = profile
        self.accel = accel
        self._stop = False
        self._start_time = None
        self._end_time = None
        self.detected = 0
        self.done = 0
        self._face_ratio = 0.0

        # Lazy-loaded (initialized in run() on worker thread)
        self.device = None
        self.model = None
        self.mp_face = None
        self.torch = None
        self.YOLO = None
        self.mp = None

        self.profile_cfg = {
            "Low": {"conf": 0.45, "min_size": 64, "ttl": 0.6,
                    "min_frames": 0.8, "min_valid_ratio": 0.75, "min_sharpness": 60,
                    "sample_fps": 3.0, "landmark_interval": 3, "skin_min": 0.12},
            "Normal": {"conf": 0.35, "min_size": 40, "ttl": 1.2,
                       "min_frames": 0.5, "min_valid_ratio": 0.6, "min_sharpness": 40,
                       "sample_fps": 5.0, "landmark_interval": 3, "skin_min": 0.10},
            "High": {"conf": 0.22, "min_size": 24, "ttl": 2.5,
                     "min_frames": 0.25, "min_valid_ratio": 0.35, "min_sharpness": 20,
                     "sample_fps": 8.0, "landmark_interval": 4, "skin_min": 0.0},
        }[profile]

        self.last_preview = 0
        self._cap = None  # cv2.VideoCapture for cleanup

    def _load_deps(self):
        """Load heavy dependencies on worker thread to avoid blocking UI."""
        if not _ensure_torch():
            raise RuntimeError("Face detection requires PyTorch, but it is not installed.")

        YOLO = _ensure_yolo()
        if YOLO is None:
            raise RuntimeError("ultralytics package not found.")

        mp = _ensure_mediapipe()
        if mp is None:
            raise RuntimeError("mediapipe package not found.")

        self.torch = shared.torch
        self.YOLO = YOLO
        self.mp = mp

        use_cuda = self.accel == "nvidia" and self.torch.cuda.is_available()
        self.device = "cuda:0" if use_cuda else "cpu"
        if not use_cuda:
            self.accel = "cpu"

        model_path = os.path.join(APP_DIR, "models", "yolov8n-face.pt")
        self.model = YOLO(model_path)

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=2, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def stop(self):
        self._stop = True
        self._ui_alive = False
        # Release cv2.VideoCapture if open
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        # Delete heavy objects to free VRAM/RAM
        self.model = None
        if self.mp_face:
            try:
                self.mp_face.close()
            except Exception:
                pass
        self.mp_face = None
        # Clear GPU memory if torch was loaded
        if self.torch is not None:
            try:
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
            except Exception:
                pass
        self.torch = None
        self.device = None

    def disable_preview(self):
        self.preview_enabled = False

    def total_time(self):
        if not self._start_time:
            return "--:--"
        end = self._end_time or time.time()
        elapsed = max(1, int(end - self._start_time))
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _skin_ratio(self, face):
        ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        return skin.mean() / 255

    def _valid_landmarks(self, face):
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        result = self.mp_face.process(rgb)
        if not result.multi_face_landmarks:
            return False
        lm = result.multi_face_landmarks[0].landmark
        left_eye, right_eye, nose, mouth = lm[33], lm[263], lm[1], lm[13]
        if not (left_eye.y < nose.y < mouth.y):
            return False
        if abs(left_eye.x - right_eye.x) < (0.020 if self.profile == "High" else 0.030):
            return False
        return True

    def run(self):
        # Load heavy dependencies on worker thread (non-blocking for UI)
        self._load_deps()

        self._start_time = time.time()
        cap = cv2.VideoCapture(self.video)
        self._cap = cap  # Store for cleanup
        try:  # FIX: ensure cap is always released
            fps_raw = cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps_raw) if fps_raw and fps_raw > 0 else 30.0
            total_frames_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            try:
                total_frames = int(float(total_frames_raw))
            except Exception:
                total_frames = 1

            tracks = []
            outdir = build_output_dir(self.output, mode="faces",
                                      profile=self.profile, accel=self.accel)
            frame_idx = 0
            track_id = 0
            min_lm_frames = 1 if self.profile == "High" else 2
            iou_thresh = 0.45
            sample_every = max(1, int(round(fps / max(1.0, self.profile_cfg["sample_fps"]))))
            ttl_frames = max(2, int(round(self.profile_cfg["sample_fps"] * self.profile_cfg["ttl"])))
            min_required_frames = {
                "Low": 3,
                "Normal": 2,
                "High": 1,
            }.get(self.profile, 2)

            while cap.isOpened():
                if self._stop:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                should_infer = frame_idx == 1 or (frame_idx - 1) % sample_every == 0
                if not should_infer:
                    if self.previewer and self.preview_enabled:
                        now = time.time()
                        if now - self.last_preview >= PREVIEW_INTERVAL:
                            draw = frame.copy()
                            for t in tracks:
                                x1, y1, x2, y2 = t["box"]
                                cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            img = resize_for_preview(
                                Image.fromarray(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)))
                            if img and self._ui_alive:
                                self.previewer.after(
                                    0, lambda img=img: (
                                        self.previewer.update_image(img)
                                        if self._ui_alive else None))
                            self.last_preview = now

                    if self.log:
                        now = time.time()
                        if now - getattr(self, "_last_eta_update", 0) >= 0.5:
                            self._last_eta_update = now
                            detected_val = self.detected
                            done_val = self.done
                            eta_val = self._calculate_eta(frame_idx, total_frames)
                            self.log.after(0, lambda d=detected_val, c=done_val, e=eta_val:
                            self.log.write_status(detected=d, cut=c, eta=e))

                    if self.progress:
                        ratio = max(self._face_ratio, frame_idx / total_frames)
                        self._face_ratio = ratio
                        if self._ui_alive:
                            self.progress.after(0, lambda v=ratio: self.progress.update(v))
                    continue

                use_cuda = self.device.startswith("cuda")
                results = self.model.predict(
                    frame, conf=self.profile_cfg["conf"], iou=iou_thresh,
                    imgsz=640 if frame.shape[1] <= 1280 else 800,
                    device=self.device, half=use_cuda, verbose=False)[0]

                new_tracks = []
                for box in results.boxes.xyxy:
                    box = box.squeeze()
                    x1, y1, x2, y2 = map(int, box.tolist())
                    face_raw = frame[y1:y2, x1:x2]
                    w, h = x2 - x1, y2 - y1
                    if w < self.profile_cfg["min_size"] or h < self.profile_cfg["min_size"]:
                        continue
                    aspect = w / h
                    if not (0.65 <= aspect <= 1.35):
                        continue

                    h_frame, w_frame, _ = frame.shape
                    expand_x = int(w * 0.15)
                    expand_y = int(h * 0.15)  # FIX: simplify duplicate variables
                    cx1 = max(0, x1 - expand_x)
                    cy1 = max(0, y1 - expand_y)
                    cx2 = min(w_frame, x2 + expand_x)
                    cy2 = min(h_frame, y2 + expand_y)

                    skin_min = float(self.profile_cfg.get("skin_min", 0.10))
                    face_crop = frame[cy1:cy2, cx1:cx2]
                    if face_raw.size == 0:
                        continue
                    if skin_min > 0 and self._skin_ratio(face_raw) < skin_min:
                        continue

                    matched = False
                    for t in tracks:
                        if self._iou(t["box"], (x1, y1, x2, y2)) > iou_thresh:
                            t["box"] = (x1, y1, x2, y2)
                            t["ttl"] = ttl_frames
                            t["frames"] += 1

                            fh, fw = face_raw.shape[:2]
                            center_face = face_raw[int(fh * 0.25):int(fh * 0.75),
                            int(fw * 0.25):int(fw * 0.75)]
                            if center_face.size == 0:
                                center_face = face_raw
                            sharp = cv2.Laplacian(center_face, cv2.CV_64F).var()
                            if sharp > t["score"]:
                                t["score"] = sharp
                                t["face"] = face_crop.copy()
                            should_validate = (
                                    t["frames"] >= min_lm_frames and
                                    (t["valid"] == 0 or t["frames"] % self.profile_cfg["landmark_interval"] == 0)
                            )
                            if should_validate and self._valid_landmarks(face_raw):
                                t["valid"] = min(t["valid"] + 1, t["frames"])
                            matched = True
                            break

                    if not matched:
                        track_id += 1
                        fh, fw = face_raw.shape[:2]
                        center_face = face_raw[int(fh * 0.25):int(fh * 0.75),
                        int(fw * 0.25):int(fw * 0.75)]
                        if center_face.size == 0:
                            center_face = face_raw
                        sharp = cv2.Laplacian(center_face, cv2.CV_64F).var()
                        new_tracks.append({
                            "id": track_id,
                            "box": (x1, y1, x2, y2),
                            "ttl": ttl_frames,
                            "frames": 1,
                            "valid": 0,
                            "score": sharp,  # FIX: reuse calculated sharp value
                            "face": face_crop.copy()
                        })

                for t in tracks:
                    t["ttl"] -= 1
                    if t["ttl"] <= 0:
                        cfg = self.profile_cfg
                        min_required = max(min_required_frames, int(cfg["sample_fps"] * cfg["min_frames"]))
                        if (t["frames"] >= min_required and
                                (t["valid"] / max(t["frames"], 1)) >= cfg["min_valid_ratio"] and
                                t["score"] >= cfg["min_sharpness"]):
                            fname = f"face_{self.done + 1:04d}.png"
                            path = os.path.join(outdir, fname)
                            if cv2.imwrite(path, t["face"]):
                                self.done += 1
                                self.detected += 1

                tracks = [t for t in tracks if t["ttl"] > 0] + new_tracks

                if self.previewer and self.preview_enabled:
                    now = time.time()
                    if now - self.last_preview >= PREVIEW_INTERVAL:
                        draw = frame.copy()
                        for t in tracks:
                            x1, y1, x2, y2 = t["box"]
                            cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        img = resize_for_preview(
                            Image.fromarray(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)))
                        if img and self._ui_alive:
                            self.previewer.after(
                                0, lambda img=img: (
                                    self.previewer.update_image(img)
                                    if self._ui_alive else None))
                        self.last_preview = now

                # FIX: Use after() for thread-safe UI updates
                # Only update ETA every ~0.5 seconds to avoid wasteful calculations
                if self.log:
                    now = time.time()
                    if now - getattr(self, "_last_eta_update", 0) >= 0.5:
                        self._last_eta_update = now
                        detected_val = self.detected
                        done_val = self.done
                        eta_val = self._calculate_eta(frame_idx, total_frames)
                        self.log.after(0, lambda d=detected_val, c=done_val, e=eta_val:
                        self.log.write_status(detected=d, cut=c, eta=e))

                # FIX: Use after() for thread-safe UI updates
                if self.progress:
                    ratio = max(self._face_ratio, frame_idx / total_frames)
                    self._face_ratio = ratio
                    progress_ratio = ratio
                    if self._ui_alive:
                        self.progress.after(0, lambda v=progress_ratio: self.progress.update(v))

            # Process remaining tracks
            cfg = self.profile_cfg
            for t in tracks:
                min_required = max(min_required_frames, int(cfg["sample_fps"] * cfg["min_frames"]))
                if (t["frames"] >= min_required and
                        (t["valid"] / max(t["frames"], 1)) >= cfg["min_valid_ratio"] and
                        t["score"] >= cfg["min_sharpness"]):
                    fname = f"face_{self.done + 1:04d}.png"
                    path = os.path.join(outdir, fname)
                    if cv2.imwrite(path, t["face"]):
                        self.done += 1
                        self.detected += 1
        finally:
            # FIX: always release cap
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            self._cap = None
            # Clear GPU memory after face detection
            if self.torch is not None:
                try:
                    if self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
                except Exception:
                    pass

        self._end_time = time.time()
        return not self._stop

    def _calculate_eta(self, frame_idx, total_frames):
        """Calculate ETA for face detection - optimized and realistic.

        Based on frame processing rate with sanity checks.
        """
        if frame_idx == 0 or total_frames == 0:
            return "--:--"
        elapsed = time.time() - self._start_time
        if elapsed < 1:
            return "--:--"

        # Frames per second
        rate = frame_idx / elapsed
        remaining = total_frames - frame_idx
        eta_seconds = int(remaining / rate) if rate > 0 else 0

        # Sanity check
        if eta_seconds < 0 or eta_seconds > 86400:
            return "--:--"

        m, s = divmod(eta_seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ============================================================
