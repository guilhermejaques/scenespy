from .shared import *

def _otsu_1d(sorted_vals):
    """Return a one-dimensional Otsu threshold, or None when the sample is too small."""
    n = len(sorted_vals)
    if n < 4:
        return None
    total_sum = sum(sorted_vals)
    if total_sum < 1e-15:
        return None
    weight_bg = 0
    sum_bg = 0.0
    max_var = 0.0
    best_t = sorted_vals[0]
    for i in range(n):
        weight_bg += 1
        sum_bg += sorted_vals[i]
        weight_fg = n - weight_bg
        if weight_fg == 0:
            break
        sum_fg = total_sum - sum_bg
        mean_bg = sum_bg / weight_bg
        mean_fg = sum_fg / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            best_t = sorted_vals[i]
    if max_var < 1e-15:
        return None
    return best_t


def _compute_ssim_simplified(img1, img2, window_size=11):
    if img1.shape != img2.shape:
        return 0.0
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    mu_a = cv2.GaussianBlur(a, (window_size, window_size), 1.5)
    mu_b = cv2.GaussianBlur(b, (window_size, window_size), 1.5)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a = cv2.GaussianBlur(a * a, (window_size, window_size), 1.5) - mu_a_sq
    sigma_b = cv2.GaussianBlur(b * b, (window_size, window_size), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (window_size, window_size), 1.5) - mu_ab
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
            (mu_a_sq + mu_b_sq + c1) * (sigma_a + sigma_b + c2) + 1e-8)
    return float(np.clip(ssim.mean(), 0.0, 1.0))


def _compute_ecr(gray1, gray2):
    edges1 = cv2.Canny(gray1, 30, 70)
    edges2 = cv2.Canny(gray2, 30, 70)
    total = cv2.countNonZero(cv2.bitwise_or(edges1, edges2))
    if total <= 0:
        return 0.0
    changed = cv2.countNonZero(cv2.bitwise_xor(edges1, edges2))
    return float(changed / total)


def _adaptive_threshold(video_path):
    """Estimate scene sensitivity from sampled motion, structure, and color changes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 32.0, 4.0

    fps_cap = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = max(1, fps_cap if fps_cap > 0 else 30)
    if total < 10:
        cap.release()
        return 32.0, 4.0

    frames_per_clip = 6
    sample_interval_s = 2.5
    frame_interval = max(3, int(fps * sample_interval_s))
    num_clips = max(15, min(50, int(total / frame_interval)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, probe = cap.read()
    if probe is not None:
        h, w = probe.shape[:2]
        target_area = 120 * 90
        cur_area = w * h
        if cur_area > target_area * 2:
            scale = (target_area / cur_area) ** 0.5
            rw, rh = int(w * scale), int(h * scale)
        else:
            rw, rh = w, h
        dim = (max(rw, 64), max(rh, 48))
    else:
        dim = (120, 90)

    BGR2GRAY = cv2.COLOR_BGR2GRAY
    BGR2YCrCb = cv2.COLOR_BGR2YCrCb
    BGR2HSV = cv2.COLOR_BGR2HSV
    HIST_NORM = cv2.NORM_MINMAX

    diff_y_list, diff_c_list, diff_hy_list, diff_hs_list = [], [], [], []
    diff_edge_list, diff_struct_list, clip_motion = [], [], []

    for _i in range(num_clips):
        pos = min(_i * frame_interval, max(0, int(total) - frames_per_clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        prev_y, prev_cbcr, prev_hist_y, prev_hist_hs = None, None, None, None
        clip_y_diffs = []

        for _j in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break

            small = cv2.resize(frame, dim)
            gray = cv2.cvtColor(small, BGR2GRAY)
            ycrcb = cv2.cvtColor(small, BGR2YCrCb)
            hsv = cv2.cvtColor(small, BGR2HSV)
            cbcr = ycrcb[:, :, 1:]

            hist_y = cv2.calcHist([gray], [0], None, [32], [0, 256])
            cv2.normalize(hist_y, hist_y, 0, 1, HIST_NORM)
            hist_hs = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist_hs, hist_hs, 0, 1, HIST_NORM)

            if prev_y is not None:
                d_y = cv2.absdiff(gray, prev_y).mean()
                d_c = cv2.absdiff(cbcr, prev_cbcr).mean()
                d_hy = cv2.compareHist(prev_hist_y, hist_y, cv2.HISTCMP_BHATTACHARYYA)
                d_hs = cv2.compareHist(prev_hist_hs, hist_hs, cv2.HISTCMP_BHATTACHARYYA)
                d_edge = _compute_ecr(prev_y, gray)
                d_struct = 1.0 - _compute_ssim_simplified(prev_y, gray)
                diff_y_list.append(d_y)
                diff_c_list.append(d_c)
                diff_hy_list.append(d_hy)
                diff_hs_list.append(d_hs)
                diff_edge_list.append(d_edge)
                diff_struct_list.append(d_struct)
                clip_y_diffs.append(d_y)

            prev_y, prev_cbcr, prev_hist_y, prev_hist_hs = gray, cbcr, hist_y, hist_hs

        if clip_y_diffs:
            clip_motion.append(sum(clip_y_diffs) / len(clip_y_diffs))

    cap.release()
    n = len(diff_y_list)
    if n < 8:
        return 32.0, 4.0

    def _robust_rank(vals):
        s = sorted(vals)
        rank = {}
        for idx, v in enumerate(s):
            if v not in rank:
                rank[v] = idx
        return [rank[v] / max(1, len(s) - 1) for v in vals]

    def _var(vals):
        m = sum(vals) / len(vals)
        return sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)

    ry = _robust_rank(diff_y_list)
    rc = _robust_rank(diff_c_list)
    rhy = _robust_rank(diff_hy_list)
    rhs = _robust_rank(diff_hs_list)
    re = _robust_rank(diff_edge_list)
    rs = _robust_rank(diff_struct_list)

    vars_list = [_var(diff_y_list), _var(diff_c_list),
                 _var(diff_hy_list), _var(diff_hs_list),
                 _var(diff_edge_list), _var(diff_struct_list)]
    v_total = sum(vars_list)
    if v_total < 1e-15:
        weights = [0.28, 0.14, 0.10, 0.12, 0.20, 0.16]
    else:
        weights = [v / v_total for v in vars_list]
        weights = [0.08 + 0.38 * w for w in weights]
        weights[4] *= 1.25
        weights[5] *= 1.15
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

    composite = [
        weights[0] * a + weights[1] * b + weights[2] * c +
        weights[3] * d + weights[4] * e + weights[5] * st
        for a, b, c, d, e, st in zip(ry, rc, rhy, rhs, re, rs)
    ]

    mf = list(composite)
    for i in range(1, n - 1):
        w = sorted([composite[i - 1], composite[i], composite[i + 1]])
        mf[i] = w[1]

    s = sorted(mf)
    q1 = s[n // 4]
    q3 = s[3 * n // 4]
    iqr_v = q3 - q1
    spike_cap = q3 + 4.0 * iqr_v
    spike_hard_level = q3 + 2.0 * iqr_v

    clean = list(mf)
    for i in range(1, n - 1):
        if mf[i] > spike_cap:
            prev_ok = mf[i - 1] < spike_hard_level
            next_ok = mf[i + 1] < spike_hard_level
            if prev_ok and next_ok:
                clean[i] = spike_cap
            else:
                clean[i] = spike_cap + (mf[i] - spike_cap) * 0.3

    motion_index = sum(clip_motion) / max(1, len(clip_motion)) if clip_motion else 0.0
    otsu_t = _otsu_1d(sorted(clean))

    if otsu_t is not None:
        grp0 = [v for v in clean if v <= otsu_t]
        grp1 = [v for v in clean if v > otsu_t]
        if grp0 and grp1:
            noise_m = sum(grp0) / len(grp0)
            trans_m = sum(grp1) / len(grp1)
            sep = trans_m - noise_m
            raw_t = noise_m + sep * 0.35
        else:
            raw_t = otsu_t
    else:
        raw_t = sorted(clean)[n // 2]

    threshold = 15.0 + raw_t * 40.0
    if motion_index > 12:
        motion_boost = min(5.0, (motion_index - 12) * 0.2)
        threshold = min(55.0, threshold + motion_boost)
    threshold = max(15.0, min(55.0, threshold))
    threshold = round(threshold, 1)

    t_norm = 1.0 - (threshold - 15.0) / 40.0
    min_dur = 1.5 + 8.5 * (t_norm ** 1.5)
    min_dur = max(1.0, min(10.0, min_dur))
    min_dur = round(min_dur, 1)

    return threshold, min_dur


def _percentile(vals, pct, default=0.0):
    if not vals:
        return default
    return float(np.percentile(np.asarray(vals, dtype=np.float32), pct))


def _median(vals, default=0.0):
    if not vals:
        return default
    return float(np.median(np.asarray(vals, dtype=np.float32)))


def _profile_name(cfg):
    label = cfg.get("label", "Normal")
    return label if label in ("Low", "Normal", "High") else "Auto"


def _make_candidate(frame, score, kind, confidence=None, source=None):
    score = float(score)
    if confidence is None:
        confidence = score / (score + 0.25)
    return {
        "frame": int(frame),
        "score": score,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "type": kind,
        "source": source or kind,
    }


def _classifier_profile_params(profile):
    return {
        "Low": {
            "accept": 0.88, "semantic": 0.95, "gradual": 0.82,
            "min_scene_s": 3.5, "refine_floor": 0.12,
        },
        "Normal": {
            "accept": 0.72, "semantic": 0.82, "gradual": 0.68,
            "min_scene_s": 0.9, "refine_floor": 0.08,
        },
        "High": {
            "accept": 0.58, "semantic": 0.68, "gradual": 0.55,
            "min_scene_s": 0.35, "refine_floor": 0.05,
        },
        "Auto": {
            "accept": 0.66, "semantic": 0.76, "gradual": 0.62,
            "min_scene_s": 0.6, "refine_floor": 0.06,
        },
    }.get(profile, {
        "accept": 0.72, "semantic": 0.82, "gradual": 0.68,
        "min_scene_s": 0.9, "refine_floor": 0.08,
    })


def _gradual_profile_params(profile):
    return {
        "Low": {"step_s": 0.35, "window": 4, "score_q": 94, "score_floor": 0.33, "min_gap_s": 4.0},
        "Normal": {"step_s": 0.25, "window": 4, "score_q": 90, "score_floor": 0.26, "min_gap_s": 1.8},
        "High": {"step_s": 0.20, "window": 3, "score_q": 84, "score_floor": 0.20, "min_gap_s": 0.7},
        "Auto": {"step_s": 0.25, "window": 4, "score_q": 88, "score_floor": 0.23, "min_gap_s": 1.2},
    }.get(profile, {"step_s": 0.25, "window": 4, "score_q": 90, "score_floor": 0.26, "min_gap_s": 1.8})


def _hist_distance(a, b):
    try:
        return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
    except Exception:
        return 0.0


def _cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(1.0 - np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _scene_embedding_from_gray_hsv(gray, hsv):
    h, w = gray.shape

    grid = []
    rows, cols = 4, 6
    for gy in range(rows):
        y1, y2 = int(gy * h / rows), int((gy + 1) * h / rows)
        for gx in range(cols):
            x1, x2 = int(gx * w / cols), int((gx + 1) * w / cols)
            cell_gray = gray[y1:y2, x1:x2]
            cell_hsv = hsv[y1:y2, x1:x2]
            grid.extend([
                float(cell_gray.mean()) / 255.0,
                float(cell_gray.std()) / 255.0,
                float(cell_hsv[:, :, 0].mean()) / 180.0,
                float(cell_hsv[:, :, 1].mean()) / 255.0,
                float(cell_hsv[:, :, 2].mean()) / 255.0,
            ])

    edges = cv2.Canny(gray, 40, 90)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=False)
    edge_grid = []
    orient_grid = []
    for gy in range(rows):
        y1, y2 = int(gy * h / rows), int((gy + 1) * h / rows)
        for gx in range(cols):
            x1, x2 = int(gx * w / cols), int((gx + 1) * w / cols)
            edge_grid.append(float(edges[y1:y2, x1:x2].mean()) / 255.0)
            cell_mag = mag[y1:y2, x1:x2]
            cell_ang = ang[y1:y2, x1:x2]
            hist, _ = np.histogram(
                cell_ang, bins=4, range=(0.0, np.pi * 2),
                weights=cell_mag)
            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-6)
            orient_grid.extend(hist.tolist())

    cy1, cy2 = int(h * 0.20), int(h * 0.80)
    cx1, cx2 = int(w * 0.18), int(w * 0.82)
    center_gray = gray[cy1:cy2, cx1:cx2]
    center_hsv = hsv[cy1:cy2, cx1:cx2]
    center = np.asarray([
        float(center_gray.mean()) / 255.0,
        float(center_gray.std()) / 255.0,
        float(center_hsv[:, :, 0].mean()) / 180.0,
        float(center_hsv[:, :, 1].mean()) / 255.0,
        float(center_hsv[:, :, 2].mean()) / 255.0,
        float(edges[cy1:cy2, cx1:cx2].mean()) / 255.0,
    ], dtype=np.float32)

    dct_src = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    dct = cv2.dct(dct_src)
    dct_low = dct[:6, :8].reshape(-1)
    dct_low = dct_low / (np.linalg.norm(dct_low) + 1e-8)

    emb = np.concatenate([
        np.asarray(grid, dtype=np.float32),
        np.asarray(edge_grid, dtype=np.float32) * 1.5,
        np.asarray(orient_grid, dtype=np.float32) * 0.55,
        center * 1.2,
        dct_low.astype(np.float32) * 0.75,
    ])
    return emb / (np.linalg.norm(emb) + 1e-8)


def _scene_embedding(frame, size=(160, 90)):
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return _scene_embedding_from_gray_hsv(gray, hsv)


def _build_transition_feature_cache(video_path, fps, total_frames, profile="Normal", stop_cb=None):
    gradual_params = _gradual_profile_params(profile)
    semantic_params = _semantic_profile_params(profile)
    step_s = min(gradual_params["step_s"], semantic_params["step_s"])
    frame_step = max(1, int(round(fps * step_s)))
    duration_s = total_frames / max(fps, 1.0) if total_frames else 0.0
    profile_caps = {"Low": 1400, "Normal": 1800, "High": 2200, "Auto": 1900}
    max_samples = profile_caps.get(profile, 1800)
    if duration_s <= 20 * 60:
        max_samples = int(max_samples * 1.25)
    elif duration_s >= 90 * 60:
        max_samples = int(max_samples * 0.85)
    if total_frames and total_frames / frame_step > max_samples:
        frame_step = max(frame_step, int(total_frames / max_samples))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    features = []
    target = (160, 90)
    try:
        for frame_idx in range(0, int(total_frames or 0), frame_step):
            if stop_cb and stop_cb():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            small = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist_hs = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
            cv2.normalize(hist_hs, hist_hs, 0, 1, cv2.NORM_L1)

            features.append({
                "frame": int(frame_idx),
                "time": frame_idx / max(fps, 1.0),
                "hist": hist_hs,
                "luma": float(gray.mean()) / 255.0,
                "contrast": float(gray.std()) / 255.0,
                "black": float((gray < 18).mean()),
                "white": float((gray > 238).mean()),
                "embedding": _scene_embedding_from_gray_hsv(gray, hsv),
            })
    finally:
        cap.release()

    return features


def _frame_signature(frame, size=(160, 90)):
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_L1)
    return gray, hist


def _signature_change_score(prev_sig, cur_sig):
    prev_gray, prev_hist = prev_sig
    cur_gray, cur_hist = cur_sig
    hist_gap = _hist_distance(prev_hist, cur_hist)
    luma_gap = float(cv2.absdiff(prev_gray, cur_gray).mean()) / 255.0
    edge_gap = _compute_ecr(prev_gray, cur_gray)
    struct_gap = 1.0 - _compute_ssim_simplified(prev_gray, cur_gray)
    return hist_gap * 0.45 + luma_gap * 0.20 + edge_gap * 0.20 + struct_gap * 0.15


def _detect_gradual_transitions(video_path, fps, total_frames, profile="Normal", stop_cb=None, features=None):
    """Find fade/dissolve candidates using windowed color/luma statistics.

    PySceneDetect is strong on hard cuts. This pass looks for slower transitions:
    a window before/after the candidate must be visually different, while the
    local sequence shows a sustained ramp or fade pattern.
    """
    params = _gradual_profile_params(profile)

    duration = total_frames / max(fps, 1.0) if total_frames else 0.0
    if duration <= 1.0:
        return []

    if features is None:
        features = _build_transition_feature_cache(
            video_path, fps, total_frames, profile=profile, stop_cb=stop_cb)
    n = len(features)
    win = params["window"]
    if n < win * 2 + 3:
        return []

    scores = []
    candidates = []
    for i in range(win, n - win):
        before = features[i - win:i]
        after = features[i + 1:i + 1 + win]
        cur = features[i]

        hist_before = np.mean([f["hist"] for f in before], axis=0).astype(np.float32)
        hist_after = np.mean([f["hist"] for f in after], axis=0).astype(np.float32)
        hist_gap = _hist_distance(hist_before, hist_after)

        luma_before = _median([f["luma"] for f in before])
        luma_after = _median([f["luma"] for f in after])
        contrast_before = _median([f["contrast"] for f in before])
        contrast_after = _median([f["contrast"] for f in after])
        luma_gap = abs(luma_after - luma_before)
        contrast_gap = abs(contrast_after - contrast_before)

        local = features[i - win:i + win + 1]
        lumas = [f["luma"] for f in local]
        blacks = [f["black"] for f in local]
        whites = [f["white"] for f in local]
        dark_peak = max(blacks)
        white_peak = max(whites)
        luma_slope = abs(lumas[-1] - lumas[0])
        fade_score = 0.0
        if dark_peak > 0.28 or white_peak > 0.25:
            fade_score = max(dark_peak, white_peak) * 0.45 + luma_slope * 0.35

        score = hist_gap * 0.70 + luma_gap * 0.20 + contrast_gap * 0.10 + fade_score
        scores.append(score)
        candidates.append((cur["frame"], cur["time"], score, hist_gap, fade_score))

    if not scores:
        return []

    threshold = max(params["score_floor"], _percentile(scores, params["score_q"]))
    min_gap_frames = max(1, int(params["min_gap_s"] * fps))
    selected = []
    for frame, _time_s, score, hist_gap, fade_score in sorted(candidates, key=lambda x: x[2], reverse=True):
        if score < threshold:
            break
        if hist_gap < 0.18 and fade_score < 0.18:
            continue
        if frame < min_gap_frames or (total_frames and frame > total_frames - min_gap_frames):
            continue
        if any(abs(frame - chosen) < min_gap_frames for chosen in selected):
            continue
        selected.append(int(frame))

    selected_set = set(selected)
    return [
        _make_candidate(frame, score, "gradual", confidence=min(1.0, score / max(threshold, 1e-6)),
                        source="fade" if fade_score >= hist_gap else "dissolve")
        for frame, _time_s, score, hist_gap, fade_score in candidates
        if int(frame) in selected_set
    ]


def _semantic_profile_params(profile):
    return {
        "Low": {"step_s": 0.55, "window": 3, "score_q": 95, "score_floor": 0.18, "min_gap_s": 4.0},
        "Normal": {"step_s": 0.40, "window": 3, "score_q": 90, "score_floor": 0.12, "min_gap_s": 1.5},
        "High": {"step_s": 0.28, "window": 2, "score_q": 84, "score_floor": 0.08, "min_gap_s": 0.6},
        "Auto": {"step_s": 0.35, "window": 3, "score_q": 88, "score_floor": 0.10, "min_gap_s": 1.0},
    }.get(profile, {"step_s": 0.40, "window": 3, "score_q": 90, "score_floor": 0.12, "min_gap_s": 1.5})


def _detect_semantic_transitions(video_path, fps, total_frames, profile="Normal", stop_cb=None, features=None):
    """Find composition/context changes missed by global frame-diff detectors."""
    params = _semantic_profile_params(profile)
    if features is None:
        features = _build_transition_feature_cache(
            video_path, fps, total_frames, profile=profile, stop_cb=stop_cb)
    samples = [(f["frame"], f["embedding"]) for f in features if "embedding" in f]

    win = params["window"]
    if len(samples) < win * 2 + 3:
        return []

    scores = []
    candidates = []
    for i in range(win, len(samples) - win):
        before = np.mean([emb for _, emb in samples[i - win:i]], axis=0)
        after = np.mean([emb for _, emb in samples[i:i + win]], axis=0)
        score = _cosine_distance(before, after)
        scores.append(score)
        candidates.append((samples[i][0], score))

    if not scores:
        return []

    threshold = max(params["score_floor"], _percentile(scores, params["score_q"]))
    min_gap_frames = max(1, int(params["min_gap_s"] * fps))
    selected = []
    for frame, score in sorted(candidates, key=lambda x: x[1], reverse=True):
        if score < threshold:
            break
        if frame < min_gap_frames or frame > total_frames - min_gap_frames:
            continue
        if any(abs(frame - chosen) < min_gap_frames for chosen in selected):
            continue
        selected.append(int(frame))

    selected_set = set(selected)
    return [
        _make_candidate(frame, score, "semantic", confidence=min(1.0, score / max(threshold, 1e-6)))
        for frame, score in candidates
        if int(frame) in selected_set
    ]


def _refine_scene_candidates(video_path, candidates, fps, total_frames, profile="Normal", stop_cb=None):
    if not candidates:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return candidates

    profile_window = {
        "Low": 0.65,
        "Normal": 0.50,
        "High": 0.35,
        "Auto": 0.60,
    }.get(profile, 0.35)
    window_frames = max(3, int(round(fps * profile_window)))
    step = max(1, int(round(fps / 18.0)))
    refined = []

    try:
        for candidate in candidates:
            if stop_cb and stop_cb():
                break
            boundary = int(candidate["frame"])
            left = max(step, int(boundary) - window_frames)
            right = min(int(total_frames) - 1, int(boundary) + window_frames)
            best_frame = int(boundary)
            best_score = -1.0
            scores = []
            prev_sig = None
            prev_idx = None

            for idx in range(left, right + 1, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                sig = _frame_signature(frame)
                if prev_sig is not None and prev_idx is not None:
                    score = _signature_change_score(prev_sig, sig)
                    scores.append(score)
                    if score > best_score:
                        best_score = score
                        best_frame = idx
                prev_sig = sig
                prev_idx = idx

            if scores:
                med = _median(scores)
                spread = _percentile(scores, 90) - _percentile(scores, 50)
                if best_score < med + max(0.08, spread * 0.65):
                    best_frame = int(boundary)
            refined_candidate = dict(candidate)
            refined_candidate["frame"] = int(best_frame)
            refined_candidate["refine_score"] = float(max(0.0, best_score))
            refined.append(refined_candidate)
    finally:
        cap.release()

    return sorted(
        (c for c in refined if 0 < int(c["frame"]) < total_frames),
        key=lambda c: int(c["frame"]))


def _read_frame_at(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ret, frame = cap.read()
    return frame if ret and frame is not None else None


def _camera_motion_compensation_score(frame_a, frame_b, orb=None):
    try:
        gray_a = cv2.cvtColor(cv2.resize(frame_a, (240, 135), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(cv2.resize(frame_b, (240, 135), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        raw_diff = float(cv2.absdiff(gray_a, gray_b).mean()) / 255.0
        if raw_diff <= 1e-5:
            return {"raw": raw_diff, "aligned": raw_diff, "explained": 0.0, "matches": 0}

        orb = orb or cv2.ORB_create(nfeatures=450, fastThreshold=12)
        kp_a, des_a = orb.detectAndCompute(gray_a, None)
        kp_b, des_b = orb.detectAndCompute(gray_b, None)
        if des_a is None or des_b is None or len(kp_a) < 10 or len(kp_b) < 10:
            return {"raw": raw_diff, "aligned": raw_diff, "explained": 0.0, "matches": 0}

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_a, des_b)
        if len(matches) < 10:
            return {"raw": raw_diff, "aligned": raw_diff, "explained": 0.0, "matches": len(matches)}

        matches = sorted(matches, key=lambda m: m.distance)[:80]
        src = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=800, confidence=0.98)
        if matrix is None or inliers is None:
            return {"raw": raw_diff, "aligned": raw_diff, "explained": 0.0, "matches": len(matches)}

        inlier_count = int(inliers.sum())
        if inlier_count < 8:
            return {"raw": raw_diff, "aligned": raw_diff, "explained": 0.0, "matches": len(matches)}

        aligned = cv2.warpAffine(
            gray_a, matrix, (gray_b.shape[1], gray_b.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        aligned_diff = float(cv2.absdiff(aligned, gray_b).mean()) / 255.0
        explained = max(0.0, min(1.0, (raw_diff - aligned_diff) / max(raw_diff, 1e-6)))
        inlier_ratio = inlier_count / max(1, len(matches))
        explained *= max(0.25, min(1.0, inlier_ratio * 1.4))
        return {
            "raw": raw_diff,
            "aligned": aligned_diff,
            "explained": float(explained),
            "matches": int(len(matches)),
            "inliers": inlier_count,
        }
    except Exception:
        return {"raw": 0.0, "aligned": 0.0, "explained": 0.0, "matches": 0}


def _add_candidate_context_scores(video_path, candidates, fps, total_frames, stop_cb=None):
    if not candidates:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return candidates

    out = []
    far = max(2, int(round(fps * 0.75)))
    near = max(1, int(round(fps * 0.25)))
    orb = cv2.ORB_create(nfeatures=450, fastThreshold=12)
    try:
        for cand in candidates:
            if stop_cb and stop_cb():
                break
            frame = int(cand["frame"])
            idxs = [
                max(0, frame - far),
                max(0, frame - near),
                min(total_frames - 1, frame + near),
                min(total_frames - 1, frame + far),
            ]
            frames = [_read_frame_at(cap, idx) for idx in idxs]
            if any(f is None for f in frames):
                out.append(cand)
                continue
            embs = [_scene_embedding(f) for f in frames]
            pre_motion = _cosine_distance(embs[0], embs[1])
            post_motion = _cosine_distance(embs[2], embs[3])
            near_cross = _cosine_distance(embs[1], embs[2])
            far_cross = _cosine_distance(embs[0], embs[3])
            local_motion = max(pre_motion, post_motion, 1e-4)
            context_ratio = far_cross / (local_motion + 0.03)
            camera_motion = _camera_motion_compensation_score(frames[1], frames[2], orb=orb)

            enriched = dict(cand)
            enriched["context_cross"] = round(float(far_cross), 4)
            enriched["context_near_cross"] = round(float(near_cross), 4)
            enriched["context_motion"] = round(float(local_motion), 4)
            enriched["context_ratio"] = round(float(context_ratio), 4)
            enriched["camera_motion_explained"] = round(float(camera_motion.get("explained", 0.0)), 4)
            enriched["camera_motion_raw"] = round(float(camera_motion.get("raw", 0.0)), 4)
            enriched["camera_motion_aligned"] = round(float(camera_motion.get("aligned", 0.0)), 4)
            enriched["camera_motion_matches"] = int(camera_motion.get("matches", 0))
            out.append(enriched)
    finally:
        cap.release()

    return out


__all__ = [name for name in globals() if not name.startswith("__")]
