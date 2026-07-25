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
from anthropic import Anthropic


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


def calibrate_clip(video_path: str, clip_id: str, calib_path: str = "calibrations.json", frame_number: int = 0):
    """
    Full flow for one clip: reuse saved calibration if it exists, otherwise
    prompt for clicks and save it. Returns (homography, pixel_corners), or
    (None, None) if calibration isn't usable (e.g. goal not visible ->
    caller should treat this clip as low-confidence / non-calibrated).
    """
    corners = load_calibration(clip_id, calib_path)
    if corners is None:
        frame = get_calibration_frame(video_path, frame_number)
        corners = click_corners_interactive(frame)
        save_calibration(clip_id, corners, calib_path)
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
):
    """
    Converts the juggling repo's predictions["Ball"] — an (N, 4) array of
    normalized [cx, cy, w, h] per PROCESSED frame, NaN when undetected —
    into the [{"frame": int, "x": float, "y": float, "confidence": float}]
    format the rest of this module expects.

    Two things this handles that are easy to get wrong:
    - "frame" here means real video frame number, computed as
      row_index * skip_frames, since detect_contact_frame's dt calculation
      needs actual elapsed time, and predictions["Ball"] rows are spaced
      skip_frames apart, not 1 apart.
    - normalized coords are denormalized to pixels using this clip's actual
      frame_width/frame_height, since normalized coords alone can't be used
      for pixel-space calibration/homography.
    - NaN rows (ball not detected that frame) are dropped rather than
      passed through — downstream velocity math would blow up on NaN.
    """
    track = []
    for i, row in enumerate(ball_array):
        cx, cy, w, h = row
        if np.isnan(cx) or np.isnan(cy):
            continue
        track.append({
            "frame": i * skip_frames,
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
    velocity_threshold_ball_widths_per_sec: float = 6.0,
    proximity_threshold_ball_widths: float = 1.5,
):
    """
    Preferred over detect_contact_frame alone — fuses ball velocity with
    pose-based foot proximity so contact detection doesn't depend on a
    hand-tuned per-clip pixel threshold, and gets `foot` for free.

    ball_track: output of ball_track_from_predictions (needs "width_px" per entry)
    foot_tracks: {"Right_Foot": [...], "Left_Foot": [...]} — same shape as
                 ball_track, built with the same adapter (e.g. via
                 predictions["Right_Foot"] / predictions["Left_Foot"], if
                 your repo's POI dict exposes those the same way it does
                 "Ball" — confirm the actual key names against get_POI).

    velocity_threshold_ball_widths_per_sec: velocity expressed as multiples
        of the ball's own apparent width per second. Self-scaling across
        camera distances — a 40px ball moving 6 ball-widths/sec means the
        same real-world speed whether the ball is 20px or 80px wide in frame.
    proximity_threshold_ball_widths: how close (in ball-widths) a foot
        keypoint must be to the ball at the candidate frame to count as
        a real strike rather than a fast ball moving for some other reason
        (bounce, occlusion glitch, camera shake).

    Returns: (frame_of_contact, foot_label, velocity_vector) or
             (None, "unknown", None) if no fused match is found — caller
             should treat this as a failed/low-confidence clip.
    """
    if len(ball_track) < 2:
        return None, "unknown", None

    # index ball_track by frame for quick lookup during proximity checks
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
            continue  # can't normalize without a size reference — skip rather than guess
        normalized_velocity = px_per_sec / ball_width  # ball-widths/second
        candidates.append((curr["frame"], normalized_velocity, (dx, dy), ball_width))

    spikes = [c for c in candidates if c[1] >= velocity_threshold_ball_widths_per_sec]
    if not spikes:
        return None, "unknown", None

    for frame, norm_velocity, vec, ball_width in spikes:
        ball_pos = ball_by_frame.get(frame)
        if ball_pos is None:
            continue

        # find the closest foot (either label) to the ball at this frame
        best_foot_label, best_dist = "unknown", None
        for label, track in foot_tracks.items():
            foot_pos = next((f for f in track if f["frame"] == frame), None)
            if foot_pos is None:
                continue
            dist = ((foot_pos["x"] - ball_pos["x"]) ** 2 + (foot_pos["y"] - ball_pos["y"]) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist, best_foot_label = dist, label

        if best_dist is None:
            continue  # no foot detected this frame at all — can't confirm, try next spike

        if best_dist <= proximity_threshold_ball_widths * ball_width:
            foot_name = "right" if "Right" in best_foot_label else "left" if "Left" in best_foot_label else "unknown"
            return frame, foot_name, vec

    # velocity spiked but no foot was ever close enough — likely a bounce,
    # deflection, or tracking glitch rather than a real strike
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
):
    """
    predictions: the repo's predictions dict for this clip, with keys
        "Ball", "Left_Knee", "Right_Knee", "Left_Foot", "Right_Foot"
        (each an (N,4) array as documented in trajectory.py's adapter)

    Returns a dict matching the target schema (minus Week 3 fields).
    """
    fps = get_reliable_fps(video_path)
    fps_reliable = fps is not None
    if not fps_reliable:
        fps = 30.0  # fallback so downstream math doesn't crash; confidence reflects the guess

    ball_track = ball_track_from_predictions(predictions["Ball"], frame_width, frame_height, fps, skip_frames)
    foot_tracks = {
        "Right_Foot": ball_track_from_predictions(predictions["Right_Foot"], frame_width, frame_height, fps, skip_frames),
        "Left_Foot": ball_track_from_predictions(predictions["Left_Foot"], frame_width, frame_height, fps, skip_frames),
    }
    knee_tracks = {
        "Right_Knee": ball_track_from_predictions(predictions["Right_Knee"], frame_width, frame_height, fps, skip_frames),
        "Left_Knee": ball_track_from_predictions(predictions["Left_Knee"], frame_width, frame_height, fps, skip_frames),
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
        H, goal_corners = calibrate_clip(video_path, clip_id, calib_path)
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

MODEL = "claude-sonnet-5"  # good default: strong enough for this, cost-efficient at volume.
                            # if per-clip cost matters at scale, claude-haiku-4-5-20251001
                            # is worth A/B testing against for a task this structured.

SYSTEM_PROMPT = """You are a football (soccer) scouting analyst. You write brief notes \
for scouts based on structured data from a single tracked shot attempt.

Apply this context when interpreting the data -- these are real scouting heuristics, \
not generic commentary:
- At youth/semi-pro level, placement matters more than power. Don't praise raw speed \
alone if placement was poor.
- A shot taken with the player's weak foot is a notable positive signal worth calling \
out explicitly, especially if placement was still good.
- A single attempt says little about consistency -- don't claim a player "is" a certain \
type of finisher from one data point; describe what THIS attempt shows.
- Shot angle indicates finishing tendency: an angle close to 0 (square to goal) suggests \
a central striker profile; a wider angle suggests a natural wide/angled finisher.
- If confidence is low, say so plainly in the note rather than writing a confident-sounding \
note the data doesn't support.

Write 2-3 sentences. Scout-readable plain English -- interpret what the data means for \
this player's tendencies, don't just restate the numbers back as a data readout."""


def generate_coaching_note(shot_data: dict):
    """
    shot_data: the dict produced by shot_pipeline.analyze_shot -- goal_zone,
    shot_angle_deg, ball_speed_kmh, foot, confidence, frame_of_contact.

    Returns the note string, or None if generation fails (API error, missing
    key) -- per the spec's "don't fake it" principle, a missing note should
    stay null, not get filled with a placeholder.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Shot data:\n{json.dumps(shot_data, indent=2)}\n\nWrite the scouting note.",
            }],
        )
        return response.content[0].text.strip()
    except Exception as e:
        # network/API errors shouldn't crash the whole pipeline -- a clip with
        # good geometry but a failed note is still useful output
        print(f"coaching_note generation failed: {e}")
        return None