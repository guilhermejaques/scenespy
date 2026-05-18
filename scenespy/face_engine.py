from .shared import *
from . import shared as shared


class FaceDetectionEngine:
    """Detects and saves representative face crops from a video. Pipeline Face Detection!"""
    def __init__(self, video, output, logbox=None, progressbar=None,
                 previewer=None, profile="Normal", accel="cpu", preview_enabled=True):
        self.video = video
        self.output = output
        self.log = logbox
        self.progress = progressbar
        self.previewer = previewer
        self.preview_enabled = preview_enabled
        self._ui_alive = True
        self._last_eta_update = 0

        self.profile = profile
        self.accel = accel
        self._stop = False
        self._start_time = None
        self._end_time = None
        self.detected = 0
        self.done = 0
        self._face_ratio = 0.0
        self._saved_faces = []

        self.device = None
        self.model = None
        self.mp_face = None
        self.torch = None
        self.YOLO = None
        self.mp = None

        self.profile_cfg = {
            "Low": {"conf": 0.45, "min_size": 64, "ttl": 0.6,
                    "min_frames": 0.35, "min_valid_ratio": 0.65, "min_sharpness": 10,
                    "min_quality": 0.42, "strong_conf": 0.78},
            "Normal": {"conf": 0.35, "min_size": 40, "ttl": 1.2,
                       "min_frames": 0.30, "min_valid_ratio": 0.40, "min_sharpness": 4,
                       "min_quality": 0.30, "strong_conf": 0.62},
            "High": {"conf": 0.22, "min_size": 24, "ttl": 2.5,
                     "min_frames": 0.12, "min_valid_ratio": 0.18, "min_sharpness": 2,
                     "min_quality": 0.18, "strong_conf": 0.38},
        }[profile]

        self.last_preview = 0
        self._cap = None

    def _load_deps(self):
        """Load heavy dependencies on worker thread to avoid blocking UI."""
        if self._stop:
            return
        if not _ensure_torch():
            raise RuntimeError("Face detection requires PyTorch, but it is not installed.")
        if self._stop:
            return

        YOLO = _ensure_yolo()
        if YOLO is None:
            raise RuntimeError("ultralytics package not found.")
        if self._stop:
            return

        mp = _ensure_mediapipe()
        if mp is None:
            raise RuntimeError("mediapipe package not found.")
        if self._stop:
            return

        self.torch = shared.torch
        self.YOLO = YOLO
        self.mp = mp

        use_cuda = self.accel == "nvidia" and self.torch.cuda.is_available()
        self.device = "cuda:0" if use_cuda else "cpu"
        if not use_cuda:
            self.accel = "cpu"

        model_path = os.path.join(APP_DIR, "models", "yolov8n-face.pt")
        self.model = YOLO(model_path)
        if self._stop:
            return

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=2, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def stop(self):
        self._stop = True
        self._ui_alive = False
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

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
        if self._stop or self.mp_face is None:
            return False
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

    def _bbox_to_measurement(self, box):
        x1, y1, x2, y2 = [float(v) for v in box]
        return np.array([
            [(x1 + x2) / 2.0],
            [(y1 + y2) / 2.0],
            [max(1.0, x2 - x1)],
            [max(1.0, y2 - y1)],
        ], dtype=np.float32)

    def _measurement_to_bbox(self, measurement):
        cx, cy, w, h = [float(v) for v in measurement[:4, 0]]
        half_w = max(1.0, w) / 2.0
        half_h = max(1.0, h) / 2.0
        return (
            int(round(cx - half_w)),
            int(round(cy - half_h)),
            int(round(cx + half_w)),
            int(round(cy + half_h)),
        )

    def _new_sort_state(self, box):
        state = np.zeros((8, 1), dtype=np.float32)
        state[:4] = self._bbox_to_measurement(box)
        covariance = np.eye(8, dtype=np.float32)
        covariance[:4, :4] *= 10.0
        covariance[4:, 4:] *= 1000.0
        return state, covariance

    def _sort_predict(self, track):
        transition = np.eye(8, dtype=np.float32)
        for i in range(4):
            transition[i, i + 4] = 1.0
        process_noise = np.eye(8, dtype=np.float32) * 0.03
        process_noise[4:, 4:] *= 8.0
        track["state"] = transition @ track["state"]
        track["covariance"] = transition @ track["covariance"] @ transition.T + process_noise
        track["box"] = self._measurement_to_bbox(track["state"])

    def _sort_update(self, track, box):
        measurement = self._bbox_to_measurement(box)
        observation = np.zeros((4, 8), dtype=np.float32)
        observation[:4, :4] = np.eye(4, dtype=np.float32)
        noise = np.eye(4, dtype=np.float32) * 8.0
        innovation = measurement - observation @ track["state"]
        residual_cov = observation @ track["covariance"] @ observation.T + noise
        try:
            gain = track["covariance"] @ observation.T @ np.linalg.inv(residual_cov)
        except np.linalg.LinAlgError:
            gain = track["covariance"] @ observation.T @ np.linalg.pinv(residual_cov)
        track["state"] = track["state"] + gain @ innovation
        identity = np.eye(8, dtype=np.float32)
        track["covariance"] = (identity - gain @ observation) @ track["covariance"]
        track["box"] = self._measurement_to_bbox(track["state"])

    def _match_face_detections(self, tracks, detections, threshold=0.25):
        pairs = []
        for track_idx, track in enumerate(tracks):
            for det_idx, detection in enumerate(detections):
                score = self._iou(track["box"], detection["box"])
                if score >= threshold:
                    pairs.append((score, track_idx, det_idx))
        pairs.sort(reverse=True)
        matched_tracks = set()
        matched_detections = set()
        matches = []
        for _score, track_idx, det_idx in pairs:
            if track_idx in matched_tracks or det_idx in matched_detections:
                continue
            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)
            matches.append((track_idx, det_idx))
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
        return matches, unmatched_tracks, unmatched_detections

    def _face_quality_score(self, box, frame_shape, confidence, sharpness):
        h_frame, w_frame = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        area_ratio = min(1.0, (w * h) / max(1.0, float(w_frame * h_frame)) * 80.0)
        sharp_score = min(1.0, max(0.0, float(sharpness) / 160.0))
        conf_score = min(1.0, max(0.0, float(confidence)))

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = abs(cx - (w_frame / 2.0)) / max(1.0, w_frame / 2.0)
        dy = abs(cy - (h_frame / 2.0)) / max(1.0, h_frame / 2.0)
        center_score = max(0.0, 1.0 - min(1.0, (dx * dx + dy * dy) ** 0.5))

        return (
            sharp_score * 0.42 +
            conf_score * 0.30 +
            area_ratio * 0.18 +
            center_score * 0.10
        )

    def _track_valid_ratio(self, track):
        return track.get("valid", 0) / max(track.get("frames", 0), 1)

    def _track_passes_face_gate(self, track, cfg):
        if track.get("face") is None:
            return False

        valid_ratio = self._track_valid_ratio(track)
        sharpness = float(track.get("score", 0.0))
        quality = float(track.get("quality", 0.0))
        confidence = float(track.get("confidence", 0.0))

        landmark_pass = (
            valid_ratio >= cfg["min_valid_ratio"] and
            sharpness >= cfg["min_sharpness"]
        )
        detector_pass = (
            confidence >= cfg["strong_conf"] and
            quality >= cfg["min_quality"] and
            sharpness >= cfg["min_sharpness"]
        )
        permissive_high_pass = (
            self.profile == "High" and
            confidence >= cfg["strong_conf"] and
            quality >= cfg["min_quality"] and
            sharpness >= max(1.0, cfg["min_sharpness"] * 0.5)
        )
        return landmark_pass or detector_pass or permissive_high_pass

    def _create_face_track(self, track_id, detection, ttl_frames, frame_idx, fps):
        state, covariance = self._new_sort_state(detection["box"])
        timestamp = self._frame_time(frame_idx, fps)
        valid = 1 if detection.get("landmarks_valid") else 0
        return {
            "id": track_id,
            "box": detection["box"],
            "state": state,
            "covariance": covariance,
            "ttl": ttl_frames,
            "frames": 1,
            "valid": valid,
            "score": detection["sharp"],
            "quality": detection["quality"],
            "confidence": detection["confidence"],
            "face": detection["face_crop"].copy(),
            "first_seen": timestamp,
            "last_seen": timestamp,
            "best_time": timestamp,
            "first_frame": int(frame_idx),
            "last_frame": int(frame_idx),
            "best_frame": int(frame_idx),
        }

    def _update_face_track(self, track, detection, ttl_frames, min_lm_frames, frame_idx, fps):
        timestamp = self._frame_time(frame_idx, fps)
        self._sort_update(track, detection["box"])
        track["ttl"] = ttl_frames
        track["frames"] += 1
        track["last_seen"] = timestamp
        track["last_frame"] = int(frame_idx)
        track["confidence"] = max(track.get("confidence", 0.0), detection["confidence"])
        if detection["quality"] > track.get("quality", 0.0):
            track["score"] = detection["sharp"]
            track["quality"] = detection["quality"]
            track["face"] = detection["face_crop"].copy()
            track["best_time"] = timestamp
            track["best_frame"] = int(frame_idx)
        if detection.get("landmarks_valid") or (
                track["frames"] >= min_lm_frames and self._valid_landmarks(detection["face_raw"])):
            track["valid"] = min(track["valid"] + 1, track["frames"])

    def _save_face_track_if_valid(self, track, outdir, min_required, cfg):
        if (track["frames"] >= min_required and
                self._track_passes_face_gate(track, cfg) and
                track.get("face") is not None):
            fname = f"face_{self.done + 1:04d}.png"
            path = os.path.join(outdir, fname)
            if cv2.imwrite(path, track["face"]):
                self.done += 1
                self.detected += 1
                self._saved_faces.append(self._face_metadata(track, fname))

    def run(self):
        if self._stop:
            return False
        self._load_deps()
        if self._stop:
            if self.mp_face:
                try:
                    self.mp_face.close()
                except Exception:
                    pass
            self.mp_face = None
            self.model = None
            self.torch = None
            self.device = None
            return False

        self._start_time = time.time()
        cap = cv2.VideoCapture(self.video)
        self._cap = cap
        try:
            fps, total_frames = self._get_video_timing(cap)

            tracks = []
            outdir = build_output_dir(self.output, mode="faces",
                                      profile=self.profile, accel=self.accel)
            frame_idx = 0
            track_id = 0
            min_lm_frames = 1 if self.profile == "High" else 2
            iou_thresh = 0.45
            sample_keep, sample_cycle = {
                "High": (1, 1),
                "Normal": (2, 3),
                "Low": (1, 2),
            }.get(self.profile, (2, 3))
            effective_fps = fps * (sample_keep / sample_cycle)
            ttl_frames = max(2, int(round(effective_fps * self.profile_cfg["ttl"])))
            min_required_frames = 3

            while cap.isOpened():
                if self._stop:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                should_infer = ((frame_idx - 1) % sample_cycle) < sample_keep
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
                        ratio = min(1.0, max(self._face_ratio, frame_idx / total_frames))
                        self._face_ratio = ratio
                        if self._ui_alive:
                            self.progress.after(0, lambda v=ratio: self.progress.update(v))
                    continue

                use_cuda = self.device.startswith("cuda")
                results = self.model.predict(
                    frame, conf=self.profile_cfg["conf"], iou=iou_thresh,
                    imgsz=640 if frame.shape[1] <= 1280 else 800,
                    device=self.device, half=use_cuda, verbose=False)[0]

                detections = []
                boxes_xyxy = results.boxes.xyxy
                boxes_conf = getattr(results.boxes, "conf", None)
                for box_index, box in enumerate(boxes_xyxy):
                    box = box.squeeze()
                    x1, y1, x2, y2 = map(int, box.tolist())
                    confidence = 1.0
                    if boxes_conf is not None:
                        try:
                            confidence = float(boxes_conf[box_index].item())
                        except Exception:
                            confidence = float(boxes_conf[box_index])
                    face_raw = frame[y1:y2, x1:x2]
                    w, h = x2 - x1, y2 - y1
                    if w < self.profile_cfg["min_size"] or h < self.profile_cfg["min_size"]:
                        continue
                    aspect = w / h
                    min_aspect, max_aspect = (0.50, 1.70) if self.profile == "High" else (0.55, 1.55)
                    if not (min_aspect <= aspect <= max_aspect):
                        continue

                    h_frame, w_frame, _ = frame.shape
                    expand_x = int(w * 0.28)
                    expand_top = int(h * 0.45)
                    expand_bottom = int(h * 0.20)
                    cx1 = max(0, x1 - expand_x)
                    cy1 = max(0, y1 - expand_top)
                    cx2 = min(w_frame, x2 + expand_x)
                    cy2 = min(h_frame, y2 + expand_bottom)

                    face_crop = frame[cy1:cy2, cx1:cx2]
                    if face_raw.size == 0:
                        continue
                    skin_ratio = self._skin_ratio(face_raw)
                    skin_min = {"Low": 0.12, "Normal": 0.04, "High": 0.0}.get(self.profile, 0.04)
                    if skin_ratio < skin_min:
                        continue

                    fh, fw = face_raw.shape[:2]
                    center_face = face_raw[int(fh * 0.25):int(fh * 0.75),
                    int(fw * 0.25):int(fw * 0.75)]
                    if center_face.size == 0:
                        center_face = face_raw
                    sharp = cv2.Laplacian(center_face, cv2.CV_64F).var()
                    quality = self._face_quality_score(
                        (x1, y1, x2, y2), frame.shape, confidence, sharp)
                    landmarks_valid = self._valid_landmarks(face_raw)
                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "face_raw": face_raw,
                        "face_crop": face_crop,
                        "sharp": sharp,
                        "quality": quality,
                        "confidence": confidence,
                        "landmarks_valid": landmarks_valid,
                    })

                for t in tracks:
                    self._sort_predict(t)

                matches, unmatched_tracks, unmatched_detections = self._match_face_detections(
                    tracks, detections)
                for track_idx, det_idx in matches:
                    self._update_face_track(
                        tracks[track_idx], detections[det_idx], ttl_frames,
                        min_lm_frames, frame_idx, fps)

                cfg = self.profile_cfg
                min_required = max(min_required_frames, int(effective_fps * cfg["min_frames"]))
                active_tracks = []
                for track_idx in unmatched_tracks:
                    track = tracks[track_idx]
                    track["ttl"] -= 1
                    if track["ttl"] <= 0:
                        self._save_face_track_if_valid(track, outdir, min_required, cfg)
                    else:
                        active_tracks.append(track)

                active_tracks.extend(tracks[track_idx] for track_idx, _det_idx in matches)
                for det_idx in unmatched_detections:
                    track_id += 1
                    active_tracks.append(
                        self._create_face_track(
                            track_id, detections[det_idx], ttl_frames, frame_idx, fps))

                tracks = active_tracks

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
                    ratio = min(1.0, max(self._face_ratio, frame_idx / total_frames))
                    self._face_ratio = ratio
                    progress_ratio = ratio
                    if self._ui_alive:
                        self.progress.after(0, lambda v=progress_ratio: self.progress.update(v))

            cfg = self.profile_cfg
            min_required = max(min_required_frames, int(effective_fps * cfg["min_frames"]))
            for t in tracks:
                self._save_face_track_if_valid(t, outdir, min_required, cfg)
        finally:

            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            self._cap = None
            if self.torch is not None:
                try:
                    if self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
                except Exception:
                    pass
            self.model = None
            if self.mp_face:
                try:
                    self.mp_face.close()
                except Exception:
                    pass
            self.mp_face = None
            self.torch = None
            self.device = None

        self._end_time = time.time()
        self._write_face_metadata(outdir, fps, frame_idx)
        return not self._stop

    def _frame_time(self, frame_idx, fps):
        return max(0.0, float(frame_idx) / max(float(fps or 0.0), 1.0))

    def _format_video_time(self, seconds):
        seconds = max(0.0, float(seconds or 0.0))
        total_ms = int(round(seconds * 1000.0))
        ms = total_ms % 1000
        total_seconds = total_ms // 1000
        s = total_seconds % 60
        total_minutes = total_seconds // 60
        m = total_minutes % 60
        h = total_minutes // 60
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

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

    def _face_metadata(self, track, filename):
        first_seen = float(track.get("first_seen", 0.0))
        last_seen = float(track.get("last_seen", first_seen))
        best_time = float(track.get("best_time", first_seen))
        frames_seen = int(track.get("frames", 0))
        valid_frames = int(track.get("valid", 0))
        return {
            "track_id": int(track.get("id", 0)),
            "file": filename,
            "first_seen": self._format_video_time(first_seen),
            "last_seen": self._format_video_time(last_seen),
            "duration": round(max(0.0, last_seen - first_seen), 3),
            "best_frame_time": self._format_video_time(best_time),
            "first_frame": int(track.get("first_frame", 0)),
            "last_frame": int(track.get("last_frame", 0)),
            "best_frame": int(track.get("best_frame", 0)),
            "frames_seen": frames_seen,
            "valid_landmark_frames": valid_frames,
            "valid_landmark_ratio": round(valid_frames / max(frames_seen, 1), 4),
            "confidence": round(float(track.get("confidence", 0.0)), 4),
            "quality": round(float(track.get("quality", 0.0)), 4),
            "sharpness": round(float(track.get("score", 0.0)), 4),
        }

    def _write_face_metadata(self, outdir, fps, frames_processed):
        try:
            end_time = self._end_time or time.time()
            elapsed = max(0.0, end_time - self._start_time) if self._start_time else 0.0
            metadata = {
                "video": {
                    "file": os.path.basename(self.video),
                    "path": self.video,
                    "fps": round(float(fps or 0.0), 6),
                    "frames_processed": int(frames_processed),
                    "duration_processed": round(self._frame_time(frames_processed, fps), 4),
                },
                "process": {
                    "mode": "faces",
                    "profile": self.profile,
                    "accel": self.accel,
                    "started_at": self._format_process_timestamp(self._start_time),
                    "finished_at": self._format_process_timestamp(end_time),
                    "elapsed": self._format_elapsed(elapsed),
                    "elapsed_seconds": round(elapsed, 2),
                },
                "output": {
                    "folder": outdir,
                },
                "results": {
                    "status": "stopped" if self._stop else "completed",
                    "faces_saved": int(self.done),
                    "tracks_saved": len(self._saved_faces),
                    "identity_note": (
                        "track_id groups continuous face tracks only; it is not a global person identity."
                    ),
                },
                "faces": self._saved_faces,
            }
            with open(os.path.join(outdir, "faces.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Failed to write face metadata: {e}")

    def _get_video_timing(self, cap):
        fps = self._probe_fps()
        if not fps:
            try:
                fps_raw = cap.get(cv2.CAP_PROP_FPS)
                fps = float(fps_raw) if fps_raw and fps_raw > 0 else None
            except Exception:
                fps = None
        fps = fps or 30.0

        total_frames = self._probe_total_frames(fps)
        if not total_frames:
            try:
                total_frames_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                total_frames = int(float(total_frames_raw))
            except Exception:
                total_frames = 0
        return fps, max(1, int(total_frames or 1))

    def _probe_fps(self):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=avg_frame_rate,r_frame_rate",
               "-of", "json", self.video]
        try:
            data = json.loads(check_output_hidden(cmd).decode(errors="ignore") or "{}")
            stream = (data.get("streams") or [{}])[0]
            return (
                self._parse_frame_rate(stream.get("avg_frame_rate")) or
                self._parse_frame_rate(stream.get("r_frame_rate"))
            )
        except Exception:
            return None

    def _probe_total_frames(self, fps):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=nb_frames:stream_tags=NUMBER_OF_FRAMES,DURATION",
               "-show_entries", "format=duration",
               "-of", "json", self.video]
        try:
            data = json.loads(check_output_hidden(cmd).decode(errors="ignore") or "{}")
            stream = (data.get("streams") or [{}])[0]
            for value in (stream.get("nb_frames"), (stream.get("tags") or {}).get("NUMBER_OF_FRAMES")):
                count = self._parse_int(value)
                if count:
                    return count
            tag_duration = (stream.get("tags") or {}).get("DURATION")
            seconds = self._parse_duration_seconds(tag_duration)
            if not seconds:
                seconds = self._parse_float((data.get("format") or {}).get("duration"))
            if seconds and fps > 0:
                return int(round(seconds * fps))
        except Exception:
            pass
        return None

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

    def _parse_float(self, value):
        try:
            text = str(value or "").strip()
            if not text or text == "N/A":
                return None
            parsed = float(text)
            return parsed if parsed > 0 else None
        except Exception:
            return None

    def _parse_duration_seconds(self, value):
        try:
            text = str(value or "").strip()
            if not text:
                return None
            if ":" not in text:
                return self._parse_float(text)
            parts = text.split(":")
            if len(parts) != 3:
                return None
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        except Exception:
            return None

    def _calculate_eta(self, frame_idx, total_frames):
        """Calculate ETA for face detection.
            Based on frame processing rate.
        """
        if frame_idx == 0 or total_frames == 0:
            return "--:--"
        elapsed = time.time() - self._start_time
        if elapsed < 1:
            return "--:--"

        rate = frame_idx / elapsed
        remaining = total_frames - frame_idx
        eta_seconds = int(remaining / rate) if rate > 0 else 0

        if eta_seconds < 0 or eta_seconds > 86400:
            return "--:--"

        m, s = divmod(eta_seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
