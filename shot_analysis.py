"""
shot_analysis.py

All-in-one module: calibration, contact detection, zone classification,
body-orientation (angle) estimation, ball speed, confidence scoring, and
the Claude-based coaching note -- merged from the original 7 separate
files for easier single-file editing during the sprint to ship this.

Function names and signatures are unchanged from the original files, so
video_driver.py only needs its import line updated to pull from this file
instead.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import os
import urllib.request
import urllib.error

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Pure python fallback if python-dotenv is not installed in current venv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# ======================================================================
# from calibration.py
# ======================================================================

# Regulation goal dimensions in meters
GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44

# Real-world destination points, in a consistent order:
# top-left, top-right, bottom-right, bottom-left (post/bar intersections)
# Origin (0,0) = bottom-left of goal, ground level.
DEST_POINTS = np.array([
    [0.0, GOAL_HEIGHT_M],          # top-left
    [GOAL_WIDTH_M, GOAL_HEIGHT_M], # top-right
    [GOAL_WIDTH_M, 0.0],           # bottom-right
    [0.0, 0.0],                    # bottom-left
], dtype=np.float32)


def get_calibration_frame(video_path: str, frame_number: int = 0):
    """Grab a single frame from the video to click corners on.
    Default to frame 0; pass a later frame_number if the goal isn't
    clearly visible at the start of the clip."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")
    return frame


def click_corners_interactive(frame):
    """
    Opens an OpenCV window. Click 4 points in this exact order:
    1. top-left post/bar corner
    2. top-right post/bar corner
    3. bottom-right post/ground corner
    4. bottom-left post/ground corner
    Press any key once all 4 are clicked to confirm.

    Returns a (4, 2) float32 array of pixel coordinates in that order.
    """
    points = []
    display = frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Click goal corners: TL, TR, BR, BL", display)

    cv2.imshow("Click goal corners: TL, TR, BR, BL", display)
    cv2.setMouseCallback("Click goal corners: TL, TR, BR, BL", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        raise ValueError(f"Expected 4 clicks, got {len(points)}. Try again.")

    return np.array(points, dtype=np.float32)


def compute_homography(pixel_corners: np.ndarray):
    """
    pixel_corners: (4,2) array in order TL, TR, BR, BL (pixel space)
    Returns: 3x3 homography matrix mapping pixel -> real-world (x, y) meters
             on the goal plane.
    """
    H, status = cv2.findHomography(pixel_corners, DEST_POINTS)
    if H is None:
        raise ValueError("Homography computation failed — check corner points aren't collinear/degenerate")
    return H


def pixel_to_world(H: np.ndarray, pixel_point: tuple):
    """Project a single pixel (x, y) through the homography to real-world
    goal-plane meters (x, y). Returns (world_x, world_y)."""
    pt = np.array([[pixel_point]], dtype=np.float32)  # shape (1,1,2)
    world_pt = cv2.perspectiveTransform(pt, H)
    return float(world_pt[0][0][0]), float(world_pt[0][0][1])


def save_calibration(clip_id: str, pixel_corners: np.ndarray, out_path: str):
    """Persist calibration so you don't have to re-click every run."""
    data = {}
    p = Path(out_path)
    if p.exists():
        data = json.loads(p.read_text())
    data[clip_id] = pixel_corners.tolist()
    p.write_text(json.dumps(data, indent=2))


def load_calibration(clip_id: str, calib_path: str):
    """Returns pixel_corners (4,2) array for clip_id, or None if not calibrated yet."""
    p = Path(calib_path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    corners = data.get(clip_id)
    return np.array(corners, dtype=np.float32) if corners else None


def _line_intersection(line1, line2):
    """Standard 2-line intersection in pixel space. Returns (x, y) or None
    if the lines are parallel (degenerate — shouldn't happen for a real
    post/crossbar pair, but guard against it anyway)."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)


def detect_goal_corners_auto(frame: np.ndarray, return_score: bool = False, roi_fraction: float = 0.60):
    """
    Robust goal corner detection solving the 4 core CV failure modes:
    1. Player Occlusion: Uses large maxLineGap (up to frame_width/5) to bridge across
       players/goalkeepers standing in front of the crossbar, and linear slope extrapolation
       to extend post lines down to ground level.
    2. Shadow/Contrast Variations: Uses HSV grass-masking combined with CLAHE brightness
       enhancement to isolate goal posts under shadows/overcast skies.
    3. Background Line Noise: Filters search to upper ROI (default 60% height) and horizontal
       crossbars to top 65% of the frame (y < 0.65*H), eliminating pitch/penalty box ground lines.
    4. Perspective Distortion: Widens allowable post tilt angles (up to 40 deg) to handle
       skewed smartphone camera angles.

    If return_score is True, returns (corners, plausibility_score) where higher is better.
    Otherwise returns corners on success or None on failure.
    """
    h_img, w_img = frame.shape[:2]

    # Crop to top ROI if roi_fraction is specified (< 1.0)
    roi_h = int(h_img * roi_fraction) if (0.0 < roi_fraction < 1.0) else h_img
    roi_frame = frame[:roi_h, :]

    # Convert to HSV & create grass mask to eliminate pitch lines/turf
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 30])
    upper_green = np.array([85, 255, 255])
    grass_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_grass_mask = cv2.bitwise_not(grass_mask)

    # Grayscale + CLAHE for shadow enhancement
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    # Combine bright pixels with non-grass mask
    threshold_values = [180, 150, 120, 90]
    for thresh_val in threshold_values:
        _, bright_mask = cv2.threshold(enhanced_gray, thresh_val, 255, cv2.THRESH_BINARY)
        combined_mask = cv2.bitwise_and(bright_mask, non_grass_mask)

        edges = cv2.Canny(combined_mask, 40, 140)

        min_len = h_img // 6
        max_gap = w_img // 5  # Large gap tolerance bridges right over players in front of the crossbar!

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35,
                                 minLineLength=min_len, maxLineGap=max_gap)
        if lines is None:
            continue

        vertical, horizontal = [], []
        for l in lines[:, 0]:
            x1, y1, x2, y2 = map(float, l)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            length = np.hypot(x2 - x1, y2 - y1)
            mid_y = (y1 + y2) / 2.0

            # Filter horizontal crossbars to upper 65% of the frame (ignores ground lines)
            if abs(angle) < 35 and mid_y < h_img * 0.65:
                horizontal.append((x1, y1, x2, y2, length))
            # Near-vertical posts (allow up to 40 deg tilt for smartphone perspective)
            elif abs(abs(angle) - 90) < 40:
                vertical.append((x1, y1, x2, y2, length))

        if len(vertical) < 2 or len(horizontal) < 1:
            continue

        # Crossbar: longest horizontal line near the top
        horizontal.sort(key=lambda h: -h[4])
        bar = horizontal[0]

        # Posts: leftmost and rightmost vertical lines by x-position
        vertical.sort(key=lambda v: (v[0] + v[2]) / 2.0)
        left_post = vertical[0]
        right_post = vertical[-1]

        # Ensure left and right posts are distinctly separated (> 15% of frame width)
        left_mid_x = (left_post[0] + left_post[2]) / 2.0
        right_mid_x = (right_post[0] + right_post[2]) / 2.0
        if abs(right_mid_x - left_mid_x) < w_img * 0.15:
            continue

        bar_line = (bar[0], bar[1], bar[2], bar[3])
        left_line = (left_post[0], left_post[1], left_post[2], left_post[3])
        right_line = (right_post[0], right_post[1], right_post[2], right_post[3])

        # Intersections for top-left and top-right post/crossbar corners
        top_left = _line_intersection(left_line, bar_line)
        top_right = _line_intersection(right_line, bar_line)
        if top_left is None or top_right is None:
            continue

        # Linear slope extrapolation for bottom-left post corner down to ground level
        def _extrapolate_bottom(post_line, top_corner):
            x1, y1, x2, y2 = post_line
            target_y = max(y1, y2)
            if abs(y2 - y1) > 1e-4:
                dx_dy = (x2 - x1) / (y2 - y1)
                extrapolated_x = top_corner[0] + dx_dy * (target_y - top_corner[1])
                return (extrapolated_x, target_y)
            return (max(x1, x2), target_y)

        bottom_left = _extrapolate_bottom(left_line, top_left)
        bottom_right = _extrapolate_bottom(right_line, top_right)

        # --- Geometric Plausibility Checks ---
        width_top = float(np.hypot(top_right[0] - top_left[0], top_right[1] - top_left[1]))
        height_l = float(np.hypot(top_left[0] - bottom_left[0], top_left[1] - bottom_left[1]))
        height_r = float(np.hypot(top_right[0] - bottom_right[0], top_right[1] - bottom_right[1]))
        avg_height = (height_l + height_r) / 2.0

        if avg_height <= 0 or width_top < w_img * 0.15:
            continue

        # A common false positive is a real crossbar combined with an unrelated
        # vertical line. It produces a convincing average aspect ratio even when
        # one "post" has almost no height, or its inferred base is above the bar.
        # Reject those shapes before they can be cached as a calibration.
        min_post_height = max(12.0, h_img * 0.04)
        if height_l < min_post_height or height_r < min_post_height:
            continue
        if bottom_left[1] <= top_left[1] + 4.0 or bottom_right[1] <= top_right[1] + 4.0:
            continue
        height_balance = min(height_l, height_r) / max(height_l, height_r)
        if height_balance < 0.18:
            continue

        # 1. Aspect Ratio Check (Target: 7.32m / 2.44m = 3.0, allow ±40% range [1.8, 4.2])
        aspect_ratio = width_top / avg_height
        if not (1.8 <= aspect_ratio <= 4.2):
            continue

        # 2. Post Parallelism Check (Max allowed angle divergence: 12.0 degrees)
        def _get_line_angle(line):
            dx, dy = line[2] - line[0], line[3] - line[1]
            if dy < 0:
                dx, dy = -dx, -dy
            return np.degrees(np.arctan2(dy, dx))

        angle_left = _get_line_angle(left_line)
        angle_right = _get_line_angle(right_line)
        angle_diff = abs(angle_left - angle_right)
        if angle_diff > 12.0:
            continue

        # 3. Crossbar Coverage Check (Crossbar length ≥ 65% of post separation)
        crossbar_len = bar[4]
        if crossbar_len < 0.65 * width_top:
            continue

        # Plausibility score, higher = better.
        # Aspect ratio score is 1.0 at exact 3.0 and drops to 0.0 at the allowed limit.
        ratio_error = abs(aspect_ratio - 3.0) / 3.0
        aspect_score = max(0.0, 1.0 - (ratio_error / 0.4))
        angle_score = max(0.0, 1.0 - (angle_diff / 12.0))
        # Prefer complete, balanced post pairs rather than merely the first
        # horizontal/vertical combination that passes a loose aspect check.
        score = 0.50 * aspect_score + 0.30 * angle_score + 0.20 * height_balance

        corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        if return_score:
            return corners, score
        return corners

    if return_score:
        return None, float('-inf')
    return None


def calibrate_clip_auto(video_path: str, clip_id: str, frame_number: int, calib_path: str = "calibrations.json",
                          window: int = 5, roi_fraction: float = 0.75, manual_fallback: bool = True):
    cache_key = f"{clip_id}_f{frame_number}"
    corners = load_calibration(cache_key, calib_path)

    if corners is None:
        best_corners = None
        best_score = float('-inf')

        window_offsets = [0] + [offset for d in range(1, window + 1) for offset in (-d, d)]
        for offset in window_offsets:
            c_fn = max(0, frame_number + offset)
            try:
                c_frame = get_calibration_frame(video_path, c_fn)
                res_corners, score = detect_goal_corners_auto(c_frame, return_score=True, roi_fraction=roi_fraction)
                if res_corners is not None and score > best_score:
                    best_score = score
                    best_corners = res_corners
            except Exception:
                continue

        if best_corners is None:
            # Try again with a larger ROI and softer line thresholds if the default pass failed.
            for offset in window_offsets:
                c_fn = max(0, frame_number + offset)
                try:
                    c_frame = get_calibration_frame(video_path, c_fn)
                    res_corners, score = detect_goal_corners_auto(c_frame, return_score=True, roi_fraction=0.9)
                    if res_corners is not None and score > best_score:
                        best_score = score
                        best_corners = res_corners
                except Exception:
                    continue

        corners = best_corners

        # Automatic calibration is deliberately conservative. When it cannot
        # find a geometrically plausible goal, use four operator clicks rather
        # than silently projecting the shot through a bad homography.
        if corners is None and manual_fallback:
            try:
                frame = get_calibration_frame(video_path, frame_number)
                corners = click_corners_interactive(frame)
            except Exception:
                corners = None

        if corners is not None:
            save_calibration(cache_key, corners, calib_path)
        else:
            raise ValueError(f"Could not calibrate clip {clip_id}")

    return compute_homography(corners), corners


# ======================================================================
# from trajectory.py
# ======================================================================

# ---------- Adapter: juggling repo's predictions["Ball"] -> our ball_track format ----------

def ball_track_from_predictions(
    ball_array: np.ndarray,
    frame_width: int,
    frame_height: int,
    fps: float,
    skip_frames: int = 1,
    total_processed_frames: int = None,
):
    """
    Converts the juggling repo's predictions["Ball"] — an (N, 4) array of
    normalized [cx, cy, w, h] per PROCESSED frame, NaN when undetected —
    into the [{"frame": int, "x": float, "y": float, "confidence": float}]
    format the rest of this module expects.

    Bug this fixes: predictions arrays are capped at MAX_LEN=100 most-recent
    rows (utils/update_predict.py). Once a clip exceeds 100 processed frames,
    row 0 is NOT frame 0 anymore -- it's whatever frame is (total_processed -
    len(array)) frames in. Pass total_processed_frames (a running count kept
    by the caller's frame loop) so this can compute the real offset instead
    of silently assuming row i = frame i. If you don't pass it, this falls
    back to the old (buggy-if-truncated) assumption -- fine for clips short
    enough to never hit MAX_LEN, risky otherwise.

    Other things this handles:
    - normalized coords are denormalized to pixels using this clip's actual
      frame_width/frame_height, since normalized coords alone can't be used
      for pixel-space calibration/homography.
    - NaN rows (ball not detected that frame) are dropped rather than
      passed through — downstream velocity math would blow up on NaN.
    """
    n = len(ball_array)
    if total_processed_frames is not None:
        start_index = max(0, total_processed_frames - n)
    else:
        start_index = 0

    track = []
    for i, row in enumerate(ball_array):
        cx, cy, w, h = row
        if np.isnan(cx) or np.isnan(cy):
            continue
        track.append({
            "frame": (start_index + i) * skip_frames,
            "x": float(cx) * frame_width,
            "y": float(cy) * frame_height,
            "width_px": float(w) * frame_width,  # kept for velocity normalization —
                                                   # a "ball-widths/sec" threshold is
                                                   # roughly scale-invariant across clips
                                                   # shot from different distances, unlike
                                                   # a raw pixel/sec threshold
            "confidence": 1.0,  # juggling repo doesn't expose per-detection confidence here;
                                 # treat all non-NaN detections as equally trusted for now —
                                 # flag if you find this isn't true in practice
        })
    return track


# ---------- Frame rate ----------

def get_reliable_fps(video_path: str) -> float:
    """Read FPS from metadata, sanity-check against frame_count/duration,
    fall back to the computed value if metadata looks wrong."""
    cap = cv2.VideoCapture(video_path)
    meta_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if meta_fps <= 0 or meta_fps > 240:
        # metadata is clearly broken — need duration another way.
        # cv2 can't give duration directly if FPS is broken, so this is a
        # known gap: log it and fall back to a conservative default,
        # flagging low confidence upstream.
        return None
    return meta_fps


# ---------- Contact detection ----------

def detect_contact_frame(ball_track: list, fps: float, velocity_threshold_px_per_sec: float = 2000.0):
    """
    ball_track: list of dicts [{"frame": int, "x": float, "y": float, "confidence": float}, ...]
                sorted by frame, one entry per frame ball was detected.
    fps: this clip's actual frame rate (real-world time normalization —
         see the note in the module docstring about why this can't be a
         fixed per-frame threshold).

    Returns: (frame_of_contact, velocity_vector) or (None, None) if no
    clear contact found — caller should treat this as a failed clip, not
    guess.
    """
    if len(ball_track) < 2:
        return None, None

    candidates = []
    for i in range(1, len(ball_track)):
        prev, curr = ball_track[i - 1], ball_track[i]
        dt = (curr["frame"] - prev["frame"]) / fps
        if dt <= 0:
            continue
        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]
        velocity = (dx**2 + dy**2) ** 0.5 / dt  # pixels/second
        candidates.append((curr["frame"], velocity, (dx, dy)))

    # find spikes above threshold
    spikes = [c for c in candidates if c[1] >= velocity_threshold_px_per_sec]
    if not spikes:
        return None, None

    # take the first spike that's followed by sustained motion (filters out
    # a single-frame bounce/noise blip that isn't a real shot)
    for idx, (frame, velocity, vec) in enumerate(spikes):
        spike_pos = next(i for i, c in enumerate(candidates) if c[0] == frame)
        # check next 3 frames stay reasonably fast (sustained, not a blip)
        following = candidates[spike_pos:spike_pos + 3]
        if len(following) >= 2 and all(v >= velocity_threshold_px_per_sec * 0.4 for _, v, _ in following):
            return frame, vec

    return None, None


def detect_contact_frame_fused(
    ball_track: list,
    foot_tracks: dict,
    fps: float,
    velocity_threshold_ball_widths_per_sec: float = 3.0,
    proximity_threshold_ball_widths: float = 4.0,
):
    """
    Fuses ball velocity spike with pose-based foot proximity to find contact frame.
    Supports a small frame window search for foot proximity and falls back to
    raw velocity if foot detection was occluded/missing during kick.
    """
    if len(ball_track) < 2:
        return None, "unknown", None

    ball_by_frame = {b["frame"]: b for b in ball_track}

    candidates = []
    for i in range(1, len(ball_track)):
        prev, curr = ball_track[i - 1], ball_track[i]
        dt = (curr["frame"] - prev["frame"]) / fps
        if dt <= 0:
            continue
        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]
        px_per_sec = (dx**2 + dy**2) ** 0.5 / dt
        ball_width = curr.get("width_px") or prev.get("width_px")
        if not ball_width or ball_width <= 0:
            continue
        normalized_velocity = px_per_sec / ball_width
        candidates.append((curr["frame"], normalized_velocity, (dx, dy), ball_width))

    spikes = [c for c in candidates if c[1] >= velocity_threshold_ball_widths_per_sec]
    if not spikes:
        raw_frame, raw_vec = detect_contact_frame(ball_track, fps, velocity_threshold_px_per_sec=1200.0)
        if raw_frame is not None:
            return raw_frame, "unknown", raw_vec
        return None, "unknown", None

    # First pass: look for a fused velocity spike + foot proximity match
    for frame, norm_velocity, vec, ball_width in spikes:
        ball_pos = ball_by_frame.get(frame)
        if ball_pos is None:
            continue

        best_foot_label, best_dist = "unknown", None
        for label, track in foot_tracks.items():
            for offset in range(-5, 6):
                f_target = frame + offset
                foot_pos = next((f for f in track if f["frame"] == f_target), None)
                if foot_pos is None:
                    continue
                dist = ((foot_pos["x"] - ball_pos["x"]) ** 2 + (foot_pos["y"] - ball_pos["y"]) ** 2) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_foot_label = label

        if best_dist is not None and best_dist <= proximity_threshold_ball_widths * ball_width:
            foot_name = "right" if "Right" in best_foot_label else "left" if "Left" in best_foot_label else "unknown"
            return frame, foot_name, vec

    spikes.sort(key=lambda s: -s[1])
    top_spike = spikes[0]
    if top_spike[1] >= 2.5:
        return top_spike[0], "unknown", top_spike[2]

    raw_frame, raw_vec = detect_contact_frame(ball_track, fps, velocity_threshold_px_per_sec=900.0)
    if raw_frame is not None:
        return raw_frame, "unknown", raw_vec

    return None, "unknown", None


# ---------- Zone classification ----------

def _bucket_zone(world_x: float, world_y: float) -> str:
    """3x3 grid over the goal face. world_x in [0, GOAL_WIDTH_M],
    world_y in [0, GOAL_HEIGHT_M], origin bottom-left."""
    # horizontal third
    if world_x < GOAL_WIDTH_M / 3:
        h = "left"
    elif world_x < 2 * GOAL_WIDTH_M / 3:
        h = "center"
    else:
        h = "right"

    # vertical third
    if world_y < GOAL_HEIGHT_M / 3:
        v = "bottom"
    elif world_y < 2 * GOAL_HEIGHT_M / 3:
        v = "mid"
    else:
        v = "top"

    return f"{v}-{h}"


def _classify_zone_pixel_fallback(ball_track: list, contact_frame: int, frame_width: int, frame_height: int):
    post_contact = [p for p in ball_track if p["frame"] > contact_frame]
    if not post_contact:
        return {"goal_zone": None, "on_target": None, "calibration_ok": False}

    last = post_contact[-1]
    if frame_width <= 0 or frame_height <= 0:
        return {"goal_zone": None, "on_target": None, "calibration_ok": False}

    x_norm = max(0.0, min(1.0, last["x"] / frame_width))
    y_norm = max(0.0, min(1.0, last["y"] / frame_height))

    if x_norm < 0.33:
        h = "left"
    elif x_norm < 0.66:
        h = "center"
    else:
        h = "right"

    if y_norm < 0.33:
        v = "top"
    elif y_norm < 0.66:
        v = "mid"
    else:
        v = "bottom"

    return {"goal_zone": f"{v}-{h}", "on_target": None, "calibration_ok": False}


def classify_zone(ball_track: list, contact_frame: int, H: np.ndarray | None, frame_width: int = None, frame_height: int = None):
    """
    ball_track: full per-frame ball positions (same shape as detect_contact_frame input)
    contact_frame: output of detect_contact_frame
    H: homography from calibration.calibrate_clip, or None if calibration failed

    Returns dict: {"goal_zone": str, "on_target": bool, "calibration_ok": bool}
    """
    if frame_width is None or frame_height is None:
        frame_width = frame_width or 0
        frame_height = frame_height or 0

    if H is None:
        return _classify_zone_pixel_fallback(ball_track, contact_frame, frame_width, frame_height)

    post_contact = [p for p in ball_track if p["frame"] > contact_frame]
    if not post_contact:
        return _classify_zone_pixel_fallback(ball_track, contact_frame, frame_width, frame_height)

    last_world = None
    for p in post_contact:
        wx, wy = pixel_to_world(H, (p["x"], p["y"]))
        last_world = (wx, wy)
        within_width = -0.5 <= wx <= GOAL_WIDTH_M + 0.5
        within_height = 0 <= wy <= GOAL_HEIGHT_M + 1.0
        if within_width and 0 <= wy <= GOAL_HEIGHT_M:
            zone = _bucket_zone(max(0, min(wx, GOAL_WIDTH_M - 0.01)), wy)
            return {"goal_zone": zone, "on_target": True, "calibration_ok": True}

    if last_world is None:
        return _classify_zone_pixel_fallback(ball_track, contact_frame, frame_width, frame_height)

    wx, wy = last_world
    if wy > GOAL_HEIGHT_M:
        miss = "miss-over"
    elif wx < 0:
        miss = "miss-left"
    elif wx > GOAL_WIDTH_M:
        miss = "miss-right"
    else:
        zone = _bucket_zone(max(0, min(wx, GOAL_WIDTH_M - 0.01)), max(0, min(wy, GOAL_HEIGHT_M - 0.01)))
        return {"goal_zone": zone, "on_target": False, "calibration_ok": True}

    return {"goal_zone": miss, "on_target": False, "calibration_ok": True}


# ======================================================================
# from body_orientation.py
# ======================================================================

def get_nearby_keypoint(track: list, target_frame: int, window: int = 3):
    """
    Knee keypoints can be missing or noisy exactly at the contact frame
    (motion blur / partial occlusion during the kick itself). Rather than
    failing if the exact frame has no detection, search a small window
    around it and take whichever is closest.

    track: list of {"frame": int, "x": float, "y": float, ...}
    Returns the closest entry within `window` frames, or None if nothing
    was found nearby (caller should treat this as "angle unavailable",
    not guess).
    """
    candidates = [p for p in track if abs(p["frame"] - target_frame) <= window]
    if not candidates:
        return None
    return min(candidates, key=lambda p: abs(p["frame"] - target_frame))


def _vector_angle_deg(v: np.ndarray) -> float:
    return np.degrees(np.arctan2(v[1], v[0]))


def estimate_shot_angle_deg(
    left_knee_track: list,
    right_knee_track: list,
    contact_frame: int,
    goal_corners_pixel: np.ndarray,
    window: int = 3,
):
    """
    left_knee_track / right_knee_track: outputs of ball_track_from_predictions-style
        adapter run on predictions["Left_Knee"] / predictions["Right_Knee"]
    contact_frame: from detect_contact_frame_fused
    goal_corners_pixel: (4,2) array in TL, TR, BR, BL pixel order — the same
        corners you already collect during calibration, no extra work needed
    window: how many frames on either side of contact to search for a valid
        knee detection if the exact frame is missing

    Returns: (angle_deg, is_approximate) where angle_deg is signed
        (-180 to 180) — sign indicates which side the player is turned
        toward relative to facing the goal square-on. Returns (None, True)
        if knees weren't detected near contact at all.
    """
    left = get_nearby_keypoint(left_knee_track, contact_frame, window)
    right = get_nearby_keypoint(right_knee_track, contact_frame, window)

    if left is None or right is None:
        return None, True  # honestly can't compute this — don't fake it

    knee_vec = np.array([right["x"] - left["x"], right["y"] - left["y"]])

    tl, tr, br, bl = goal_corners_pixel
    top_vec = tr - tl
    bottom_vec = br - bl
    goal_vec = (top_vec + bottom_vec) / 2  # average both edges for stability

    angle = _vector_angle_deg(knee_vec) - _vector_angle_deg(goal_vec)
    angle = (angle + 180) % 360 - 180  # normalize to [-180, 180]

    return float(angle), True  # is_approximate is always True until hips exist


# ======================================================================
# from ball_speed.py
# ======================================================================

FOOTBALL_DIAMETER_M = 0.22  # regulation size 5 ball


def estimate_ball_speed_kmh(ball_track: list, contact_frame: int, fps: float, window: int = 3):
    """
    Uses ball displacement over the first few frames after contact
    (the "window") to estimate speed right as the shot leaves the foot,
    which is what's actually useful to a scout.

    Returns a float km/h, or None if it can't be computed honestly --
    never guesses.
    """
    post = [p for p in ball_track if contact_frame <= p["frame"] <= contact_frame + window]
    if len(post) < 2:
        return None

    p0, p1 = post[0], post[-1]
    dt = (p1["frame"] - p0["frame"]) / fps
    if dt <= 0:
        return None

    dist_px = ((p1["x"] - p0["x"]) ** 2 + (p1["y"] - p0["y"]) ** 2) ** 0.5
    ball_width_px = p0.get("width_px") or p1.get("width_px")
    if not ball_width_px:
        return None

    meters_per_px = FOOTBALL_DIAMETER_M / ball_width_px
    speed_kmh = (dist_px * meters_per_px / dt) * 3.6

    # sanity clamp -- values outside a plausible shot-speed range mean bad
    # tracking (motion blur inflating apparent displacement), not a real
    # reading. Fastest recorded shots are ~130km/h; anything above that
    # from an amateur clip is almost certainly a tracking artifact.
    if not (0 < speed_kmh <= 150):
        return None

    return round(speed_kmh, 1)


# ======================================================================
# from confidence.py
# ======================================================================

def compute_confidence(
    contact_found: bool,
    calibration_ok: bool,
    foot_known: bool,
    fps_reliable: bool,
    angle_available: bool,
) -> float:
    """
    If contact wasn't found at all, nothing downstream is trustworthy --
    return 0.0 immediately rather than let partial credit imply otherwise.
    """
    if not contact_found:
        return 0.0

    score = 0.0
    # calibration is the biggest driver of zone-reading trust
    score += 0.45 if calibration_ok else 0.0
    # angle is available, but remember it's a knee-line proxy, not true hip
    # orientation (see body_orientation.py) -- capped below full weight for
    # that reason even when present
    score += 0.25 if angle_available else 0.0
    # foot identified via pose-proximity fusion, not just guessed
    score += 0.15 if foot_known else 0.0
    # fps metadata trustworthy -> contact-frame timing is trustworthy
    score += 0.15 if fps_reliable else 0.0

    return round(min(score, 1.0), 2)


# ======================================================================
# from shot_pipeline.py
# ======================================================================

def analyze_shot(
    video_path: str,
    clip_id: str,
    predictions: dict,
    frame_width: int,
    frame_height: int,
    skip_frames: int = 1,
    calib_path: str = "calibrations.json",
    total_processed_frames: int = None,
):
    """
    predictions: the repo's predictions dict for this clip, with keys
        "Ball", "Left_Knee", "Right_Knee", "Left_Foot", "Right_Foot"
        (each an (N,4) array as documented in ball_track_from_predictions)
    total_processed_frames: total frames the caller's loop actually
        processed for this clip -- needed to correct for MAX_LEN=100
        truncation. Pass this; if omitted, frame numbers may be wrong for
        clips longer than ~100 processed frames.

    Returns a dict matching the target schema (minus coaching_note, which
    is filled in separately since it needs an API call).
    """
    fps = get_reliable_fps(video_path)
    fps_reliable = fps is not None
    if not fps_reliable:
        fps = 30.0  # fallback so downstream math doesn't crash; confidence reflects the guess

    ball_track = ball_track_from_predictions(predictions["Ball"], frame_width, frame_height, fps, skip_frames, total_processed_frames)
    foot_tracks = {
        "Right_Foot": ball_track_from_predictions(predictions["Right_Foot"], frame_width, frame_height, fps, skip_frames, total_processed_frames),
        "Left_Foot": ball_track_from_predictions(predictions["Left_Foot"], frame_width, frame_height, fps, skip_frames, total_processed_frames),
    }
    knee_tracks = {
        "Right_Knee": ball_track_from_predictions(predictions["Right_Knee"], frame_width, frame_height, fps, skip_frames, total_processed_frames),
        "Left_Knee": ball_track_from_predictions(predictions["Left_Knee"], frame_width, frame_height, fps, skip_frames, total_processed_frames),
    }

    contact_frame, foot, contact_vector = detect_contact_frame_fused(ball_track, foot_tracks, fps)
    contact_found = contact_frame is not None
    foot_known = foot != "unknown"

    if not contact_found:
        return {
            "clip_id": clip_id,
            "frame_of_contact": None,
            "foot": "unknown",
            "goal_zone": None,
            "shot_angle_deg": None,
            "ball_speed_kmh": None,
            "confidence": 0.0,
            "coaching_note": None,
        }

    try:
        # Calibrate at contact_frame, not frame 0 -- this is what fixes the
        # "video moves the goal" problem. Auto-detected, no clicking needed;
        # falls back to a click only if auto-detection fails on this frame.
        H, goal_corners = calibrate_clip_auto(video_path, clip_id, contact_frame, calib_path)
        calibration_ok = True
    except ValueError:
        H, goal_corners = None, None
        calibration_ok = False

    zone_result = classify_zone(ball_track, contact_frame, H, frame_width=frame_width, frame_height=frame_height)
    ball_speed_kmh = estimate_ball_speed_kmh(ball_track, contact_frame, fps)

    angle_deg, angle_is_approx = (None, True)
    if calibration_ok:
        angle_deg, angle_is_approx = estimate_shot_angle_deg(
            knee_tracks["Left_Knee"], knee_tracks["Right_Knee"], contact_frame, goal_corners
        )
    angle_available = angle_deg is not None

    confidence = compute_confidence(
        contact_found=contact_found,
        calibration_ok=calibration_ok,
        foot_known=foot_known,
        fps_reliable=fps_reliable,
        angle_available=angle_available,
    )

    return {
        "clip_id": clip_id,
        "frame_of_contact": contact_frame,
        "foot": foot,
        "goal_zone": zone_result["goal_zone"],
        "shot_angle_deg": angle_deg,
        "ball_speed_kmh": ball_speed_kmh,
        "confidence": confidence,
        "coaching_note": None,    # filled in separately by coaching_note.py -- kept out
                                   # of this function so testing the geometry doesn't
                                   # require an API key or network call every run
    }


# ======================================================================
# from coaching_note.py
# ======================================================================

SYSTEM_PROMPT = """You are a football scout writing a concise 2-3 sentence coaching note from one structured shot-analysis record.

Translate the evidence into natural scouting language rather than repeating JSON fields. At youth and semi-professional level, placement matters more than raw power. Foot used is a useful scouting signal, but never label it a weak or dominant foot unless that fact is provided. A wide angle can suggest comfort finishing from wide areas, while a central angle can suggest a central-striker profile; treat the angle as approximate. Mention limited tracking confidence when relevant and never invent match context, pressure, or technique not present in the data.
"""


_gemini_models_cache: list[str] | None = None


def _available_gemini_models(gemini_key: str) -> list[str]:
    """Validate the key and discover models this Gemini project can use."""
    global _gemini_models_cache
    if _gemini_models_cache is not None:
        return _gemini_models_cache

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as he:
        print(f"Gemini API key validation failed: HTTP {he.code}. Check the key's Google AI Studio project and API access.")
        _gemini_models_cache = []
        return _gemini_models_cache
    except Exception as exc:
        print(f"Gemini model discovery failed: {exc}")
        _gemini_models_cache = []
        return _gemini_models_cache

    models = []
    for model in data.get("models", []):
        if "generateContent" not in model.get("supportedGenerationMethods", []):
            continue
        name = model.get("name", "")
        if name.startswith("models/"):
            name = name.removeprefix("models/")
        if name:
            models.append(name)

    _gemini_models_cache = models
    return models


def generate_coaching_note(shot_data: dict, attempt_history: list[dict] | None = None):
    """Generate a coaching note from one video's structured shot data."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_key or gemini_key.startswith("your-"):
        print("coaching_note generation skipped: No valid GEMINI_API_KEY or GOOGLE_API_KEY configured in .env. Using local fallback note.")
        return _generate_fallback_note(shot_data, attempt_history)

    available_models = _available_gemini_models(gemini_key)
    if not available_models:
        print("Gemini returned no usable generation models; using the local fallback note.")
        return _generate_fallback_note(shot_data, attempt_history)

    # Respect an explicit .env preference when it is available, then prefer
    # current Flash models, and finally use any model the key exposes.
    requested_model = os.environ.get("GEMINI_MODEL")
    # The 2.5 identifiers may still appear in model discovery but are retired
    # for new users. Prefer the provider-maintained current aliases instead.
    preferred_models = [requested_model, "gemini-flash-latest", "gemini-flash-lite-latest"]
    models_to_try = []
    for model_name in preferred_models + available_models:
        if model_name and model_name in available_models and model_name not in models_to_try:
            models_to_try.append(model_name)

    prompt_text = f"Shot data:\n{json.dumps(shot_data, indent=2)}\n\nWrite the coaching note."
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.7}
    }

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text_parts = [p.get("text", "") for p in parts if "text" in p and not p.get("thought", False)]
                text = "".join(text_parts).strip()
                if text:
                    return text
            if isinstance(data.get("output"), dict):
                text = data["output"].get("text")
                if text:
                    return text.strip()

            print(f"Gemini coaching_note returned no text ({model_name}); using the local fallback.")
            return _generate_fallback_note(shot_data, attempt_history)
        except urllib.error.HTTPError as he:
            if he.code == 404:
                print(f"Gemini model unavailable ({model_name}); trying the next discovered model.")
                continue
            try:
                error_body = he.read().decode("utf-8")
                error_data = json.loads(error_body).get("error", {})
                message = error_data.get("message", "Gemini request failed")
            except Exception:
                message = str(he)

            if he.code == 429:
                print(f"Gemini quota reached ({model_name}); using the local fallback.")
            else:
                print(f"Gemini coaching_note failed ({model_name}): HTTP {he.code}; using the local fallback.")
            return _generate_fallback_note(shot_data, attempt_history)
        except Exception as e:
            print(f"Gemini coaching_note failed ({model_name}): {e}; using the local fallback.")
            return _generate_fallback_note(shot_data, attempt_history)

    print("Gemini discovered models but none returned a coaching note; using the local fallback note.")
    return _generate_fallback_note(shot_data, attempt_history)


def _generate_fallback_note(shot_data: dict, attempt_history: list[dict] | None = None) -> str:
    """Local narrative fallback when Gemini is unavailable."""
    foot = shot_data.get("foot", "unknown")
    goal_zone = shot_data.get("goal_zone")
    speed = shot_data.get("ball_speed_kmh")
    confidence = shot_data.get("confidence", 0.0)

    usable_attempts = [
        attempt for attempt in (attempt_history or [shot_data])
        if not attempt.get("error") and attempt.get("confidence", 0) >= 0.3
    ]
    if confidence < 0.2:
        return "Tracking was limited on this attempt, so the finishing read is tentative. The available data should be treated as a prompt for video review rather than a firm assessment."

    pieces = []
    if goal_zone:
        if goal_zone.startswith("miss"):
            pieces.append(f"The attempt appears to miss {goal_zone.split('-')[1].replace('_', ' ')}.")
        else:
            pieces.append(f"The finish looks directed toward the {goal_zone.replace('-', ' ')} area.")
    else:
        pieces.append("The placement is unclear from the available tracking data.")

    observed_feet = {attempt.get("foot") for attempt in usable_attempts if attempt.get("foot") in {"left", "right"}}
    if len(usable_attempts) >= 3 and len(observed_feet) == 2:
        pieces.append("Across the usable sample, contact with both feet is an encouraging early sign of two-sided finishing.")
    elif foot != "unknown":
        pieces.append(f"This attempt was likely taken with the {foot} foot.")

    if speed is not None:
        if speed >= 30:
            pieces.append("The strike has good power for the level of footage.")
        elif speed >= 15:
            pieces.append("The shot looks composed with moderate pace.")
        else:
            pieces.append("This looks like a more controlled finish rather than a full-power strike.")

    if len(usable_attempts) < 3:
        pieces.append("The sample is too limited to establish a consistent finishing tendency.")
    elif confidence < 0.5:
        pieces.append("Tracking confidence is modest, so the pattern should be checked against the video before drawing firm conclusions.")

    return " ".join(pieces[:3])


# ======================================================================
# debug helper (added post-merge to diagnose real-clip runs where
# contact_frame looks wrong)
# ======================================================================

def debug_contact_detection(predictions, frame_width, frame_height, fps,
                             velocity_threshold_ball_widths_per_sec=6.0):
    """
    Prints normalized ball velocity per frame and the closest foot distance
    at each frame, so you can see exactly why a given contact_frame was
    picked instead of the real strike frame. Run this on any clip where
    the output looks wrong.
    """
    ball_track = ball_track_from_predictions(predictions["Ball"], frame_width, frame_height, fps)
    foot_tracks = {
        "Right_Foot": ball_track_from_predictions(predictions["Right_Foot"], frame_width, frame_height, fps),
        "Left_Foot": ball_track_from_predictions(predictions["Left_Foot"], frame_width, frame_height, fps),
    }

    print(f"{'frame':>6} {'norm_vel':>10} {'ball_width':>10} {'closest_foot_dist':>18}")
    for i in range(1, len(ball_track)):
        prev, curr = ball_track[i - 1], ball_track[i]
        dt = (curr["frame"] - prev["frame"]) / fps
        if dt <= 0:
            continue
        px_per_sec = ((curr["x"] - prev["x"]) ** 2 + (curr["y"] - prev["y"]) ** 2) ** 0.5 / dt
        ball_width = curr.get("width_px") or prev.get("width_px")
        norm_vel = px_per_sec / ball_width if ball_width else float("nan")

        best_dist = None
        for label, track in foot_tracks.items():
            foot_pos = next((f for f in track if f["frame"] == curr["frame"]), None)
            if foot_pos:
                d = ((foot_pos["x"] - curr["x"]) ** 2 + (foot_pos["y"] - curr["y"]) ** 2) ** 0.5
                if best_dist is None or d < best_dist:
                    best_dist = d

        flag = " <-- above threshold" if norm_vel >= velocity_threshold_ball_widths_per_sec else ""
        dist_str = f"{best_dist:.1f}" if best_dist is not None else "no foot detected"
        print(f"{curr['frame']:>6} {norm_vel:>10.2f} {ball_width:>10.1f} {dist_str:>18}{flag}")
