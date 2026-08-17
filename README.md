# FutbolConnect Shot Analysis Engine

FutbolConnect Shot Analysis is an end-to-end computer-vision and AI pipeline for extracting shot metrics from soccer video clips. It leverages the included `fc_juggle` vision engine (YOLO object detection + MediaPipe pose landmarks + Kalman filtering) to estimate contact frame, striking foot, goal placement zone, ball speed, body orientation angle, confidence quality scoring, hard acceptance gating, and AI-driven scout reports using Google Gemini.

The repository supports local CLI execution, batch processing, and headless cloud deployment on **Modal** (`modal_app.py`).

---

## Output Schema & Frontend Contract

For each video clip, `video_driver.run_on_video` emits a standardized JSON payload:

### Accepted Attempt (`accepted: true`)
```json
{
  "clip_id": "clip_01",
  "accepted": true,
  "reject_reasons": [],
  "reject_reason": null,
  "frame_of_contact": 42,
  "foot": "right",
  "goal_zone": "top-center",
  "shot_angle_deg": 14.2,
  "ball_speed_kmh": 78.5,
  "confidence": 0.85,
  "coaching_note": "A clean right-footed strike driven with solid power (78.5 km/h) into the top-center zone. Excellent follow-through and goalward body orientation."
}
```

### Rejected Attempt (`accepted: false`)
```json
{
  "clip_id": "clip_02",
  "accepted": false,
  "reject_reasons": [
    "calibration_failed",
    "foot_not_identified"
  ],
  "reject_reason": "calibration_failed",
  "frame_of_contact": 18,
  "foot": "unknown",
  "goal_zone": null,
  "shot_angle_deg": null,
  "ball_speed_kmh": 62.1,
  "confidence": 0.15,
  "coaching_note": null
}
```

### Key Field Descriptions:
- `accepted`: Hard yes/no validity gate (`true` iff attempt passed all verification criteria).
- `reject_reasons`: Array of all failed conditions (`no_contact_detected`, `foot_not_identified`, `calibration_failed`, `insufficient_post_contact_tracking`).
- `reject_reason`: Primary rejection string (for backward compatibility).
- `frame_of_contact`: Detected kick frame index (or `null` if rejected).
- `foot`: Striking foot (`left` or `right`). Rejects `unknown` feet in production.
- `goal_zone`: Goal grid placement (`top-left`, `bottom-center`, etc.) or miss classification (`miss-over`, `miss-left`, `miss-right`). `null` if uncalibrated/rejected.
- `shot_angle_deg`: Signed body orientation angle relative to goal face ($-180^\circ$ to $+180^\circ$).
- `ball_speed_kmh`: Calculated post-contact ball velocity in km/h ($0 < \text{speed} \le 150$).
- `confidence`: Soft quality score ($0.0$ to $1.0$) based on metadata stability. Not an accuracy percentage.
- `coaching_note`: 2-3 sentence AI scout report (Gemini Flash or local fallback). Guaranteed `null` when `accepted: false`.

---

## How It Works

```text
Video (.mp4)
  │
  ├──► YOLOv8 Ball Detection + MediaPipe Pose Landmarks
  ├──► Kalman Filter Trajectory Smoothing
  ├──► Fused Contact Detection (Consecutive-Spike Gate + Foot Proximity)
  ├──► Automated Goal Post Corner Detection & Homography (H-Matrix)
  ├──► 2D Parabolic Trajectory Fitting & Goal Plane Extrapolation
  ├──► Biomechanical Shot Angle & Speed Calculation
  ├──► Hard Acceptance Gate & Soft Quality Scoring
  └──► Gemini Flash Coaching Note Generation (with Retry & Fallback)
```

---

## Repository Layout

```text
futbolconnect-shot-analysis/
├── shot_analysis.py      # Core CV pipeline, calibration, math, acceptance, Gemini AI notes
├── video_driver.py       # OpenCV stream runner & batch processing driver
├── modal_app.py          # Modal cloud app definition & FastAPI web service endpoint (/shot-analyze)
├── PROJECT_COMPLETION_REPORT.md # Comprehensive end-to-end technical system architecture report
├── calibrations.json     # Cached automatic goal corner calibrations
├── results.json          # Batch processing results
├── .env                  # Local environment API keys (git-ignored)
└── fc_juggle/            # Submodule: YOLO model weights, MediaPipe tracking, Kalman filter
```

---

## Setup & Running Locally

### Prerequisites
- Python 3.10 – 3.12
- Git (with submodules)
- OpenCV with headless support

### Installation

1. **Clone with submodules**:
   ```bash
   git clone --recurse-submodules https://github.com/FutbolConnectInc/shot_analysis_engine.git
   cd shot_analysis_engine
   ```

2. **Virtual Environment & Dependencies**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install numpy opencv-python-headless mediapipe ultralytics torch torchvision python-dotenv fastapi python-multipart modal
   ```

3. **Configure Environment Keys**:
   Create a local `.env` file (git-ignored):
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Run Local Shot Analysis

Run batch analysis on test clips:
```powershell
.\venv\Scripts\python.exe video_driver.py
```

Run single clip analysis:
```powershell
.\venv\Scripts\python.exe video_driver.py path\to\video.mp4 clip_01
```

---

## Deploying to Modal (Production Serverless Cloud)

The shot analysis API runs as a serverless FastAPI service on **Modal**:

1. **Create Modal Secret**:
   ```bash
   modal secret create gemini-api-key GEMINI_API_KEY=your_gemini_api_key_here
   ```

2. **Deploy Service**:
   ```bash
   modal deploy modal_app.py
   ```

3. **Production API Endpoint**:
   - `POST /shot-analyze`: Accepts multipart MP4/MOV upload and returns the standard JSON response payload.
