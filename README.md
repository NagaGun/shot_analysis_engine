# FutbolConnect Shot Analysis

FutbolConnect Shot Analysis is a local computer-vision pipeline for extracting shot metrics from football video. It builds on the included `fc_juggle` submodule to detect the ball and player landmarks, then estimates the contact frame, striking foot, goal area, ball speed, body-orientation proxy, and a confidence score. It can also produce a short coaching note using Gemini or a local rule-based fallback.

The repository also includes the standalone `fc_juggle` project, which provides a juggle-counting CLI and FastAPI service. The shot-analysis entry point is the root-level `video_driver.py`.

## What the shot pipeline produces

For each clip, the batch runner emits a JSON object with this schema:

```json
{
  "clip_id": "center",
  "frame_of_contact": 31,
  "foot": "left",
  "goal_zone": "bottom-center",
  "shot_angle_deg": null,
  "ball_speed_kmh": 3.0,
  "confidence": 0.3,
  "coaching_note": "The finish looks directed toward the bottom center area."
}
```

- `frame_of_contact` is the detected kick frame, or `null` when it cannot be established.
- `foot` is `left`, `right`, or `unknown`.
- `goal_zone` is one of the nine goal-grid labels, a calibrated miss label (`miss-over`, `miss-left`, or `miss-right`), or `null`.
- `shot_angle_deg` is a signed, approximate angle based on the knee line and goal edge; it is `null` without usable calibration and knee landmarks.
- `ball_speed_kmh` is an estimate from the first three frames after contact; it can be `null` for inadequate or implausible tracking.
- `confidence` is a 0–1 score derived from contact detection, calibration, foot attribution, FPS metadata, and angle availability.
- `coaching_note` is generated from each clip's structured JSON when Gemini is configured; otherwise the pipeline uses a local fallback note.

Batch results are written to `results.json`. Goal-corner detections are cached in `calibrations.json`.

## How it works

```text
video (.mp4)
  -> YOLOv8 football detection + MediaPipe pose landmarks
  -> Kalman-filtered trajectories
  -> contact, foot, calibration, zone, speed, and angle estimation
  -> JSON result and optional coaching note
```

The detector tracks the ball plus the left/right foot, knee, and head. The source `fc_juggle` tracker retains the most recent 100 measurements; the root driver passes the total processed-frame count so that reported frame numbers remain aligned on longer clips.

Goal calibration uses automatic goal-post/crossbar detection near the contact frame and a homography to the regulation 7.32 m × 2.44 m goal plane. When calibration fails, zone classification falls back to the final detected ball position as a proportion of the video frame. That fallback is useful for a tentative label but is not an on-target determination.

## Repository layout

```text
futbolconnect-shot-analysis/
├── video_driver.py       # Shot-analysis CLI and batch runner
├── shot_analysis.py      # Shot metrics, calibration, and coaching-note logic
├── results.json          # Latest batch output (generated/overwritten)
├── calibrations.json     # Cached automatic goal calibrations
├── .env                  # Optional local Gemini key; never commit it
└── fc_juggle/            # Git submodule: detector, tracker, juggle CLI/API, model, sample clips
    ├── models/finetuned.pt
    ├── source_data/
    ├── main.py
    ├── api.py
    └── utils/
```

## Setup

### Prerequisites

- Git, including submodule support
- Python 3.12 (the supplied `fc_juggle/environment.yml` targets Python 3.12)
- A camera-compatible OpenCV installation; GPU acceleration is optional

Clone the repository with its submodule:

```bash
git clone --recurse-submodules <repository-url>
cd futbolconnect-shot-analysis
```

If the repository is already cloned:

```bash
git submodule update --init --recursive
```

Create and activate a virtual environment:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies required by the shot pipeline:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy opencv-contrib-python mediapipe ultralytics torch python-dotenv
```

For the `fc_juggle` FastAPI service, additionally install:

```powershell
python -m pip install fastapi "uvicorn[standard]" python-multipart scipy
```

Alternatively, the submodule supplies `fc_juggle/environment.yml` for a Conda-based environment. Its package versions are intended for the juggle project and may be used instead of the pip setup above.

## Optional Gemini coaching notes

Create a root `.env` file to enable Gemini notes:

```env
GEMINI_API_KEY=your-key
```

`GOOGLE_API_KEY` is also accepted. No API key is required for shot analysis: if no valid key is found, or every Gemini request fails, a local rule-based note is returned. The current shot pipeline does not call Anthropic/Claude; `test_anthropic.py` is a separate legacy API-key diagnostic script.

## Run shot analysis

Use the repository virtual-environment interpreter. `video_driver.py` validates its MediaPipe environment before importing the submodule's vision code.

Analyze all `.mp4` files in `fc_juggle/source_data/`:

```powershell
.\venv\Scripts\python.exe video_driver.py
```

Analyze one video:

```powershell
.\venv\Scripts\python.exe video_driver.py path\to\clip.mp4 clip_id
```

The single-clip command prints the result to standard output. A batch run generates and prints each result, including its coaching note, as soon as that video is processed; it then overwrites `results.json`. By default, the batch stops after the first clip that returns an `error`.

To force new goal calibration on later runs, replace the contents of `calibrations.json` with `{}`. To clear saved batch output, replace `results.json` with `[]`.

## Run the bundled juggle counter

From the submodule directory, run the interactive CLI against a video or webcam:

```powershell
Set-Location fc_juggle
..\venv\Scripts\python.exe main.py --video source_data\center.mp4
```

Useful CLI options:

```text
--save <directory>  Save an annotated MP4
--plot              Show the ball Y-trajectory plot
--json              Emit a headless JSON result (requires --video)
```

To start the juggle-counting HTTP API locally:

```powershell
Set-Location fc_juggle
..\venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8080
```

The API exposes `GET /health` and `POST /analyze`. The analysis endpoint accepts either a multipart `file`, a form `url`, and an optional `skip_frames` form field. Example:

```powershell
curl.exe -X POST http://localhost:8080/analyze -F "file=@source_data/center.mp4"
```

`fc_juggle` also includes a Dockerfile and `modal_app.py` for container and Modal deployments. Those serve the juggle API, not the root shot-analysis pipeline.

## Accuracy and limitations

- The fine-tuned YOLO model detects one football class at a confidence threshold of 0.3.
- MediaPipe pose tracks one primary pose, so foot attribution can be unreliable with multiple players, occlusion, or motion blur.
- The Kalman filters are module-level state in `fc_juggle`; a batch may carry state between clips. Treat batch results as a prototype workflow and restart the process between controlled evaluations when isolation matters.
- Goal calibration depends on visible posts and crossbar. Auto-calibration can fail with occlusion, unusual goal colors, weak contrast, extreme perspective, or a goal that is not visible near contact.
- Pixel-fallback goal labels are approximate and do not establish whether a shot was on target.
- Ball speed is inferred from the detected ball width using a 0.22 m ball diameter and is rejected above 150 km/h. It is an estimate, not radar data.
- The shot-angle value is a knee-line proxy, not a full body-orientation measurement.
- Results in the committed `results.json` are example output from the included clips, not ground-truth labels.

## License and data

No license file is present in this repository. Confirm usage rights before redistributing the code, the bundled model, or the sample videos.
