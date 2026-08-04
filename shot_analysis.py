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


def detect_goal_corners_auto(frame: np.ndarray, return_score: bool = False, roi_top_ratio: float = 0.60):
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
    """
    h_img, w_img = frame.shape[:2]

    # Crop to top ROI if roi_top_ratio is specified (< 1.0)
    roi_h = int(h_img * roi_top_ratio) if (0.0 < roi_top_ratio < 1.0) else h_img
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

        # Calculate Plausibility Penalty Score (0.0 is perfect)
        aspect_penalty = abs(aspect_ratio - 3.0) / 3.0
        angle_penalty = angle_diff / 12.0
        score = 0.6 * aspect_penalty + 0.4 * angle_penalty

        corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        if return_score:
            return corners, score
        return corners

    if return_score:
        return None, float('inf')
    return None


def calibrate_clip_auto(video_path: str, clip_id: str, frame_number: int, calib_path: str = "calibrations.json"):
    cache_key = f"{clip_id}_f{frame_number}"
    corners = load_calibration(cache_key, calib_path)

    if corners is None:
        best_corners = None
        best_score = float('inf')

        # Try a 5-frame window around target frame [frame - 2 ... frame + 2]
        window_offsets = [0, -1, 1, -2, 2]
        for offset in window_offsets:
            c_fn = max(0, frame_number + offset)
            try:
                c_frame = get_calibration_frame(video_path, c_fn)
                res_corners, score = detect_goal_corners_auto(c_frame, return_score=True)
                if res_corners is not None and score < best_score:
                    best_score = score
                    best_corners = res_corners
            except Exception:
                continue

        corners = best_corners

        # Fallback: if all frames in the 5-frame window fail, try manual interactive click on target frame
        if corners is None:
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
    velocity_threshold_ball_widths_per_sec: float = 3.5,
    proximity_threshold_ball_widths: float = 3.0,
):
    """
    Fuses ball velocity spike with pose-based foot proximity to find contact frame.
    Supports a small frame window search for foot proximity and falls back to
    top velocity spike if foot detection was occluded/missing during kick.
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
        if not ball_width:
            continue
        normalized_velocity = px_per_sec / ball_width
        candidates.append((curr["frame"], normalized_velocity, (dx, dy), ball_width))

    spikes = [c for c in candidates if c[1] >= velocity_threshold_ball_widths_per_sec]
    if not spikes:
        return None, "unknown", None

    # First pass: look for a fused velocity spike + foot proximity match
    for frame, norm_velocity, vec, ball_width in spikes:
        ball_pos = ball_by_frame.get(frame)
        if ball_pos is None:
            continue

        best_foot_label, best_dist = "unknown", None
        # Check foot positions in window [frame - 3, frame + 3]
        for label, track in foot_tracks.items():
            for offset in range(-3, 4):
                f_target = frame + offset
                foot_pos = next((f for f in track if f["frame"] == f_target), None)
                if foot_pos is None:
                    continue
                dist = ((foot_pos["x"] - ball_pos["x"]) ** 2 + (foot_pos["y"] - ball_pos["y"]) ** 2) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist, best_foot_label = dist, label

        if best_dist is not None and best_dist <= proximity_threshold_ball_widths * ball_width:
            foot_name = "right" if "Right" in best_foot_label else "left" if "Left" in best_foot_label else "unknown"
            return frame, foot_name, vec

    # Fallback pass: if foot was missing/occluded near kick, pick strongest velocity spike
    spikes.sort(key=lambda s: -s[1])
    top_spike = spikes[0]
    if top_spike[1] >= 4.0:
        return top_spike[0], "unknown", top_spike[2]

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


def classify_zone(ball_track: list, contact_frame: int, H: np.ndarray | None):
    """
    ball_track: full per-frame ball positions (same shape as detect_contact_frame input)
    contact_frame: output of detect_contact_frame
    H: homography from calibration.calibrate_clip, or None if calibration failed

    Returns dict: {"goal_zone": str, "on_target": bool, "calibration_ok": bool}
    """
    if H is None:
        # Can't determine zone honestly without calibration. Don't guess.
        return {"goal_zone": None, "on_target": None, "calibration_ok": False}

    post_contact = [p for p in ball_track if p["frame"] > contact_frame]
    if not post_contact:
        return {"goal_zone": None, "on_target": None, "calibration_ok": True}

    # Walk frames after contact, project each into world coords, and find
    # the first frame where the ball is within (or crossing into) the goal
    # bounding box in world space. That's our "crossing" frame.
    last_world = None
    for p in post_contact:
        wx, wy = pixel_to_world(H, (p["x"], p["y"]))
        last_world = (wx, wy)
        within_width = -0.5 <= wx <= GOAL_WIDTH_M + 0.5   # small margin for post width
        within_height = 0 <= wy <= GOAL_HEIGHT_M + 1.0     # margin for over-the-bar misses near the frame
        if within_width and 0 <= wy <= GOAL_HEIGHT_M:
            zone = _bucket_zone(max(0, min(wx, GOAL_WIDTH_M - 0.01)), wy)
            return {"goal_zone": zone, "on_target": True, "calibration_ok": True}

    # never crossed within the goal bounds -> classify miss direction using
    # the last tracked world position
    if last_world is None:
        return {"goal_zone": None, "on_target": None, "calibration_ok": True}

    wx, wy = last_world
    if wy > GOAL_HEIGHT_M:
        miss = "miss-over"
    elif wx < 0:
        miss = "miss-left"
    else:
        miss = "miss-right"

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

    zone_result = classify_zone(ball_track, contact_frame, H)
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

MODEL_GEMINI = "gemini-1.5-flash"
MODEL_CLAUDE = "claude-3-5-sonnet-latest"

SYSTEM_PROMPT = """You are a senior football (soccer) scouting analyst writing concise evaluation notes for academy and scouting directors based on computer-vision shot metrics.

Apply these core scouting heuristics when interpreting the structured data:
- Write 2-3 complete, professional, highly scout-readable sentences. NEVER end mid-sentence or cut off.
- Do NOT simply repeat raw numbers or JSON keys back (e.g. do not say "ball_speed_kmh is 12.5"). Instead, translate the numbers into technical scouting insight.
- Placement over power: At youth/semi-pro level, placement into target zones matters more than raw speed. A 12-15 km/h placed shot in the top corner is higher quality than a wild 40 km/h miss over the bar.
- Weak Foot Signal: A shot taken with the player's non-dominant/weak foot is a notable positive scouting signal to highlight explicitly, especially when placement is accurate.
- Finishing Tendency: Acute/wide shot angles suggest a natural wide/angled finisher; central angles suggest a center-forward profile.
- Uncertainty: If overall confidence is low (< 0.5), mention that the attempt had limited tracking clarity while assessing what was visible.

Example note style:
"Demonstrates strong technical control and composure to curl a left-footed attempt cleanly into the top-right corner. Prioritizes precise placement over brute power, showing great confidence on their non-dominant side."
"""


def generate_coaching_note(shot_data: dict):
    """
    Generates a 2-3 sentence scout note using Anthropic Claude (preferred) or Gemini.
    """
    # Reload .env defensively in case it wasn't loaded at import time
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        for _line in _env_path.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

    # 1. Try Anthropic Claude if ANTHROPIC_API_KEY is configured
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key and anthropic_key.startswith("sk-ant-"):
        claude_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-latest",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
        ]
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=anthropic_key)
            for c_model in claude_models:
                try:
                    response = client.messages.create(
                        model=c_model,
                        max_tokens=500,
                        system=SYSTEM_PROMPT,
                        messages=[{
                            "role": "user",
                            "content": f"Shot data:\n{json.dumps(shot_data, indent=2)}\n\nWrite the scouting note.",
                        }],
                    )
                    text = response.content[0].text.strip()
                    if text:
                        print(f"[coaching_note] Generated via Anthropic ({c_model})")
                        return text
                except Exception as me:
                    print(f"[coaching_note] Claude model {c_model} failed: {me}")
                    continue
        except ImportError:
            print("[coaching_note] ERROR: 'anthropic' package not installed. Run: .\\venv\\Scripts\\pip.exe install anthropic")
        except Exception as e:
            print(f"[coaching_note] Claude client failed: {e}")
    else:
        if anthropic_key:
            print(f"[coaching_note] ANTHROPIC_API_KEY has unexpected format (prefix: {anthropic_key[:12]}...)")
        else:
            print("[coaching_note] ANTHROPIC_API_KEY not found in environment")

    # 2. Try Google Gemini API if GEMINI_API_KEY / GOOGLE_API_KEY is configured
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key and not gemini_key.startswith("your-"):
        models_to_try = [
            ("v1beta", "gemini-2.5-flash"),
            ("v1beta", "gemini-2.0-flash"),
            ("v1beta", "gemini-1.5-flash"),
            ("v1", "gemini-1.5-flash"),
            ("v1beta", "gemini-2.5-flash-lite"),
        ]

        payload = {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": [
                {
                    "parts": [{"text": f"Shot data:\n{json.dumps(shot_data, indent=2)}\n\nWrite the scouting note."}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 2000,
                "temperature": 0.7
            }
        }

        for api_ver, model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{model_name}:generateContent?key={gemini_key}"
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
            except urllib.error.HTTPError as he:
                if he.code == 404:
                    continue  # try next candidate model
                print(f"Gemini coaching_note failed ({model_name}): {he}")
                break
            except Exception as e:
                print(f"Gemini coaching_note failed: {e}")
                break

    print("coaching_note generation skipped: No valid ANTHROPIC_API_KEY or GEMINI_API_KEY configured in .env")
    return None


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