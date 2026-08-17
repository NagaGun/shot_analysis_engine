"""
video_driver.py

The piece that was still missing: actually running a real video through
the repo's own detection loop (get_POI -> update_measurements -> predict_KF,
same pattern as main.py/api.py) to produce the predictions dict, then
handing that to shot_pipeline.analyze_shot, then optionally generating the
coaching note.

IMPORTANT: adjust the two imports below to match your actual repo's module
paths -- these are my best inference from what you showed me (fc-juggle's
utils/vision_estimate.py and utils/update_predict.py), not confirmed against
your real file layout. If the import fails, that's the first thing to check.
"""

import cv2
import json
import sys
import os
import numpy as np

# fc_juggle is a git submodule pointing at the private juggling repo --
# add it to sys.path so `utils.vision_estimate` / `utils.update_predict`
# resolve the same way they do inside that repo's own main.py.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FC_JUGGLE_DIR = os.path.join(REPO_ROOT, "fc_juggle")
sys.path.insert(0, FC_JUGGLE_DIR)


def _is_repo_venv_python() -> bool:
    expected = os.path.normcase(os.path.abspath(os.path.join(REPO_ROOT, "venv", "Scripts", "python.exe")))
    return os.path.normcase(os.path.abspath(sys.executable)) == expected


def _validate_python_env():
    if _is_repo_venv_python():
        return

    try:
        import mediapipe as mp
        if hasattr(mp, "solutions"):
            return
    except Exception:
        pass

    venv_python = os.path.join(REPO_ROOT, "venv", "Scripts", "python.exe")
    sys.exit(
        "ERROR: Incompatible Python environment.\n"
        "This project requires the repo virtualenv interpreter because "
        "fc_juggle/utils/vision_estimate.py uses MediaPipe with mp.solutions.pose.\n\n"
        f"Run the script like:\n  {venv_python} video_driver.py <video_path> <clip_id>\n\n"
        "Future work can make vision_estimate.py support both mp.solutions and mediapipe.tasks."
    )


_validate_python_env()

# vision_estimate.py loads its YOLO weights via a path relative to the
# current working directory ("./models/finetuned.pt"), which only resolves
# correctly if cwd is fc_juggle/ itself -- true when that repo's own
# main.py runs it, not true when we import it from here. Temporarily chdir
# into fc_juggle/ just for this import, then restore cwd immediately after,
# so the rest of THIS script's relative paths (clips/, calibrations.json)
# stay relative to this repo's root as expected.
_original_cwd = os.getcwd()
os.chdir(FC_JUGGLE_DIR)
try:
    from utils.vision_estimate import get_POI
    from utils.update_predict import update_measurements, predict_KF, kalman_filter
finally:
    os.chdir(_original_cwd)

from shot_analysis import analyze_shot, generate_coaching_note


def run_on_video(video_path: str, clip_id: str, calib_path: str = "calibrations.json", generate_note: bool = True):
    # Reset global Kalman filter state so previous video clips don't leak state into this clip
    kalman_filter.clear()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"clip_id": clip_id, "error": f"could not open {video_path}"}

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # update_measurements expects these keys to already exist with empty
    # arrays -- it appends via np.vstack, it doesn't lazily create keys.
    # This matches the repo's own documented initial state
    # (measurements[point] = np.empty(shape=(0, 4))); ideally confirm this
    # against main.py/api.py's actual init code rather than trusting this
    # hardcoded list long-term.
    POI_KEYS = ["Left_Knee", "Right_Knee", "Left_Foot", "Right_Foot", "Head", "Ball"]
    measurements = {k: np.empty((0, 4)) for k in POI_KEYS}
    predictions = {k: np.empty((0, 4)) for k in POI_KEYS}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        POIs = get_POI(frame)
        measurements = update_measurements(measurements, POIs)
        predictions = predict_KF(measurements, predictions)
        frame_count += 1

    cap.release()

    # predictions arrays are capped at MAX_LEN=100 most-recent processed frames
    # (utils/update_predict.py). frame_count (total frames actually processed)
    # is passed to analyze_shot below specifically so it can compute the right
    # offset when that cap has trimmed early rows -- this used to silently
    # produce wrong frame numbers on longer clips; now it's handled.

    required_keys = ["Ball", "Right_Foot", "Left_Foot", "Right_Knee", "Left_Knee"]
    missing = [k for k in required_keys if k not in predictions or predictions[k].shape[0] == 0]
    if missing:
        return {"clip_id": clip_id, "error": f"no detections for: {missing}"}

    result = analyze_shot(
        video_path, clip_id, predictions,
        frame_width=frame_width, frame_height=frame_height,
        total_processed_frames=frame_count,
        calib_path=calib_path,
    )

    # Gate on "accepted", not confidence>0 — confidence can be nonzero on
    # a rejected attempt. Only write a note for attempts that passed the
    # hard accept/reject gate in analyze_shot.
    if generate_note and result.get("accepted"):
        result["coaching_note"] = generate_coaching_note(result)

    return result


def run_batch(clips: list, calib_path: str = "calibrations.json", generate_note: bool = True, stop_on_error: bool = True):
    """
    clips: list of (video_path, clip_id) tuples -- your 10 test clips.
    Returns a list of result dicts, and writes them to results.json so
    you have a record to show your manager alongside the code.

    By default, stop_on_error=True so the script halts on the first
    clip that fails with an error instead of continuing through the batch.
    """
    results = []
    for video_path, clip_id in clips:
        print(f"Processing {clip_id}...")
        result = run_on_video(video_path, clip_id, calib_path, generate_note)
        results.append(result)
        print(json.dumps(result, indent=2))

        if stop_on_error and result.get("error"):
            print(f"Stopping batch because clip {clip_id} failed: {result['error']}")
            break

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    # Example: python video_driver.py clips/clip_01.mp4 clip_01
    # Or run all 10 by editing CLIPS below and calling run_batch instead.
    if len(sys.argv) >= 3:
        video_path, clip_id = sys.argv[1], sys.argv[2]
        result = run_on_video(video_path, clip_id)
        print(json.dumps(result, indent=2))
    else:
        source_dir = os.path.join(FC_JUGGLE_DIR, "source_data")
        video_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".mp4")])
        CLIPS = [
            (os.path.join("fc_juggle", "source_data", v), os.path.splitext(v)[0].replace(" ", "_"))
            for v in video_files
        ]
        run_batch(CLIPS)
