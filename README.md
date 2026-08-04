# FutbolConnect — Shot Analysis Engine

> **Automated computer vision pipeline** that turns a raw football video clip into structured shot data — foot used, goal zone, ball speed, shot angle, and an AI-generated scouting note — with no special equipment beyond a phone camera.

---

## Table of Contents

- [Overview](#overview)
- [Output Format](#output-format)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Goal Zone Grid](#goal-zone-grid)
- [Key Algorithms](#key-algorithms)
- [Known Limitations](#known-limitations)
- [Dependencies](#dependencies)

---

## Overview

Manual scouting is slow, expensive, and inconsistent. This engine automates the quantitative layer of shot analysis — extracting objective, measurable data from standard `.mp4` files — and augments it with an LLM-generated qualitative coaching note.

**No radar guns. No tracking hardware. Just video.**

The pipeline uses:
- **YOLOv8** (fine-tuned on footballs) for ball detection
- **MediaPipe Pose** for player body landmark tracking
- **Kalman filtering** for trajectory smoothing
- **OpenCV Hough transforms + homography** for goal plane calibration
- **Google Gemini / Anthropic Claude** for AI-generated coaching notes

---

## Output Format

Each video clip produces one structured JSON record:

```json
{
  "clip_id": "bottom_right",
  "frame_of_contact": 60,
  "foot": "left",
  "goal_zone": "mid-right",
  "shot_angle_deg": -23.34,
  "ball_speed_kmh": 103.4,
  "confidence": 1.0,
  "coaching_note": "Player demonstrates a composed left-footed finish into the mid-right channel, showing good awareness of goal placement under pressure. The approach angle suggests a wide attacker profile — the ability to cut inside and convert at pace is a positive scouting signal. At this level, consistent placement over power will separate elite finishers."
}
```

All results are written to `results.json` after each batch run.

---

## How It Works

### End-to-End Flow

```
.mp4 video file
      │
      ▼
┌─────────────────────────────────────────────┐
│ VISION LAYER (per frame)                    │
│  • YOLOv8 fine-tuned → ball bbox [cx,cy,w,h]│
│  • MediaPipe Pose    → 5 body landmarks     │
│    (Left/Right Knee, Left/Right Foot, Head) │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ KALMAN FILTER (per POI, per axis)           │
│  Smooths noisy detections, bridges gaps     │
│  State: [position, velocity, acceleration]  │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ SHOT ANALYSIS PIPELINE                      │
│  1. detect_contact_frame_fused()            │
│     Velocity spike ≥ 6.0 ball-widths/sec   │
│     + foot within 1.5 ball-widths of ball  │
│     → contact_frame, foot label            │
│                                             │
│  2. calibrate_clip_auto()                   │
│     Hough lines → goal corners → 3×3 H     │
│     Calibrated at contact frame (not frame 0)│
│                                             │
│  3. classify_zone()                         │
│     Homography projects ball → world coords │
│     → one of 9 goal zones                  │
│                                             │
│  4. estimate_ball_speed_kmh()               │
│     3-frame displacement × size calibration │
│                                             │
│  5. estimate_shot_angle_deg()               │
│     Knee vector vs. goal edge vector        │
│                                             │
│  6. compute_confidence()                    │
│     Weighted signal score [0.0 → 1.0]      │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ AI COACHING NOTE                            │
│  Google Gemini (primary) or Claude fallback │
│  Structured JSON → 2-3 sentence scout note  │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
              results.json
```

---

## Project Structure

```
futbolconnect-shot-analysis/
│
├── shot_analysis.py          # Core pipeline (all analysis logic)
│   ├── detect_goal_corners_auto()     # Hough transform goal detection
│   ├── detect_contact_frame_fused()   # Boot-meets-ball frame detection
│   ├── calibrate_clip_auto()          # Goal homography calibration
│   ├── classify_zone()                # 3×3 goal zone classification
│   ├── estimate_ball_speed_kmh()      # Physics-based speed estimation
│   ├── estimate_shot_angle_deg()      # Body orientation vs. goal
│   ├── compute_confidence()           # Weighted signal scoring
│   └── generate_coaching_note()       # Gemini/Claude AI note
│
├── video_driver.py           # CLI entry point & batch runner
├── calibrations.json         # Cached goal homographies (per clip + frame)
├── results.json              # Last batch run output
├── .env                      # API keys (git-ignored — never commit)
│
└── fc_juggle/                # Git submodule (private)
    ├── models/finetuned.pt   # YOLOv8n fine-tuned football detector
    ├── source_data/          # Test video clips (.mp4)
    └── utils/
        ├── vision_estimate.py   # get_POI(): YOLO + MediaPipe per frame
        ├── update_predict.py    # Measurement accumulation + KF prediction
        └── KalmanFilter.py      # Kalman1D constant-acceleration filter
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- `git` with submodule support

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repo-url>
cd futbolconnect-shot-analysis
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install ultralytics mediapipe opencv-contrib-python numpy scipy anthropic
```

> **Note:** `torch` is installed automatically as an `ultralytics` dependency. GPU inference (CUDA/MPS) is auto-selected if available.

---

## Configuration

Create a `.env` file in the project root (this file is git-ignored):

```env
# .env — never commit this file

# Option A: Google Gemini (recommended)
GEMINI_API_KEY=your-gemini-api-key-here

# Option B: Anthropic Claude
# ANTHROPIC_API_KEY=sk-ant-...
```

The pipeline checks `ANTHROPIC_API_KEY` first, then falls back to `GEMINI_API_KEY`. If neither is set, `coaching_note` will be `null` in the output — the analysis pipeline itself still runs normally.

### Getting API Keys

| Provider | Where to get it |
|---|---|
| Google Gemini | [aistudio.google.com](https://aistudio.google.com) → Get API key |
| Anthropic Claude | [console.anthropic.com](https://console.anthropic.com) → API Keys |

---

## Running the Pipeline

### Run all clips in `fc_juggle/source_data/`

```bash
.\venv\Scripts\python.exe video_driver.py
# or on macOS/Linux:
./venv/bin/python video_driver.py
```

This will:
1. Auto-discover all `.mp4` files in `fc_juggle/source_data/`
2. Process each clip through the full pipeline
3. Print each result as it completes
4. Write all results to `results.json`

### Run a single clip

```bash
.\venv\Scripts\python.exe video_driver.py path/to/clip.mp4 clip_id
```

### Clear the cache before re-running

```bash
# Windows PowerShell
'[]' | Out-File results.json -Encoding utf8
'{}' | Out-File calibrations.json -Encoding utf8
```

This forces the pipeline to re-detect goal corners rather than using cached homographies.

---

## Goal Zone Grid

The goal is divided into a **3×3 grid** based on FIFA regulation dimensions (7.32m wide × 2.44m tall):

```
┌─────────────┬──────────────┬─────────────┐
│  top-left   │  top-center  │  top-right  │
│  (0–2.44m)  │ (2.44–4.88m) │ (4.88–7.32m)│
├─────────────┼──────────────┼─────────────┤  2.44m
│  mid-left   │  mid-center  │  mid-right  │
│             │              │             │
├─────────────┼──────────────┼─────────────┤  0.81m
│ bottom-left │bottom-center │bottom-right │
│             │              │             │
└─────────────┴──────────────┴─────────────┘  0m
     0m            3.66m           7.32m
```

Shots that miss are classified as `miss-over`, `miss-left`, or `miss-right`.

---

## Key Algorithms

### Contact Frame Detection

The exact frame where boot meets ball is found by fusing two signals:

1. **Velocity spike** — ball displacement normalized by ball width (scale-invariant) exceeds **6.0 ball-widths/second**
2. **Foot proximity gate** — a foot must be within **1.5 ball-widths** of the ball at that frame

This eliminates false positives from bounces, camera shake, and tracking glitches.

### Goal Calibration

Goal corners are detected using **Canny edge detection + Hough line transform** on a CLAHE-enhanced, grass-masked frame. The four detected corners are mapped to metric world coordinates via `cv2.findHomography()`. Calibration runs at the **contact frame** (not frame 0) to handle moving cameras.

### Ball Speed

```
speed_kmh = (displacement_px × 0.22m/ball_width_px / Δt) × 3.6
```

Speed is clamped to `(0, 150]` km/h — anything above 150 is treated as a tracking artifact.

### Confidence Score

| Signal | Weight |
|---|---|
| Goal calibration succeeded | 0.45 |
| Shot angle available | 0.25 |
| Foot label known | 0.15 |
| FPS metadata reliable | 0.15 |

If no contact frame is found, confidence is `0.0` and no coaching note is generated.

---

## Known Limitations

| Issue | Impact |
|---|---|
| **MediaPipe is single-person** | In clips with multiple players in frame, foot attribution may target the wrong person |
| **Non-white goal posts** | The brightness threshold (190) fails on grey or coloured goals — falls back to manual click |
| **Camera shake during shot** | Rapid pan between contact and goal crossing corrupts zone classification |
| **Ball speed on very fast shots** | Motion blur causes overestimated displacement; clamped at 150 km/h |
| **FPS fallback is 30fps** | Clips recorded at 24fps or 60fps will have timing errors if metadata is missing |
| **Kalman state not reset in batch** | State carries across clips in a single batch run — correctness risk at scale |

---

## Dependencies

| Library | Role |
|---|---|
| `ultralytics` (YOLOv8) | Fine-tuned ball detection |
| `mediapipe` | Human pose estimation (33 landmarks) |
| `opencv-contrib-python` | Video I/O, edge detection, homography |
| `numpy` | Array math, Kalman filter |
| `scipy` | Peak detection for juggle counting |
| `anthropic` | Claude API for coaching notes |
| `torch` | YOLO inference backend (auto-selects CPU/MPS/CUDA) |

---

*Built for FutbolConnect — automated shot analysis from raw mobile video.*
