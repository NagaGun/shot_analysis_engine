# Project Completion Report: FutbolConnect Shot Analysis

**Reporting period:** 25 July–5 August 2026, based on repository commit history  
**Project status:** Prototype pipeline delivered; not production-validated

## Results

| Planned deliverable | Final completion status | Evidence / notes |
|---|---|---|
| Per-clip structured shot analysis | Completed | `video_driver.py` runs the detector/tracker and writes a JSON result for each clip. |
| Contact-frame detection and striking-foot attribution | Completed, prototype quality | `shot_analysis.py` combines normalized ball velocity with foot proximity. Foot may be `unknown` when pose evidence is insufficient. |
| Goal-plane calibration and nine-zone classification | Completed with fallback | Automatic post/crossbar detection and homography are implemented. If calibration fails, the pipeline emits an approximate frame-relative zone rather than an on-target decision. |
| Ball-speed estimate | Completed, prototype quality | Uses ball-size scaling over the first three post-contact frames and rejects values above 150 km/h. |
| Body-orientation / shot-angle estimate | Partially completed | A knee-line proxy is implemented; it is not a true hip or body-orientation measurement and may be `null`. |
| Confidence score | Completed | Score combines contact detection, calibration, foot attribution, FPS reliability, and angle availability. |
| AI coaching note | Completed with resilient fallback | Gemini is used when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is configured. A local rule-based note is used when no valid key is available or Gemini fails. |
| Batch processing and output persistence | Completed | The default command scans `fc_juggle/source_data/`, prints results, and overwrites `results.json`; calibrations are cached in `calibrations.json`. |
| Reuse of existing juggling CV stack | Completed | The root pipeline integrates the `fc_juggle` submodule’s YOLO, MediaPipe, and Kalman-filter tracker. |
| Documentation and operator instructions | Completed | Root `README.md` now documents setup, commands, outputs, API boundaries, and limitations. |
| Production-ready model validation / robust goal-post model | Not completed | No labeled ground-truth evaluation, multi-person tracking, or goalpost-specific model has been delivered. |

The included `results.json` contains ten example clip outputs, confirming an end-to-end batch run. Those results are demonstration output, not ground-truth validation.

## Metrics

| Metric | Final value | Assessment |
|---|---:|---|
| Demonstration clips processed | 10 | End-to-end output is present in `results.json`. |
| Implemented shot-result fields | 8 | `clip_id`, `frame_of_contact`, `foot`, `goal_zone`, `shot_angle_deg`, `ball_speed_kmh`, `confidence`, and `coaching_note`. |
| Code delivery window evidenced by commits | 12 calendar days | First relevant commit: 25 July 2026; latest pipeline-resilience commit: 5 August 2026. |
| Budget spent | Not recorded | No budget, invoice, time-sheet, or cost data exists in the repository; final spend cannot be calculated from available evidence. |
| Timeline adherence | Not measurable | The repository contains an executive brief dated 29 July but no approved milestone schedule or completion target. The implementation window is documented above, but variance from plan is unknown. |

For future reporting, record the approved budget, planned milestone dates, hours by role, cloud/GPU spend, Gemini API usage, and actual completion dates in a project tracker.

## Takeaways

| Challenge | Solution delivered | Remaining consideration |
|---|---|---|
| The existing detector/tracker was designed for juggling, not shots. | Added an adapter that converts normalized rolling predictions into pixel-space shot tracks and corrects frame offsets for the 100-frame history cap. | Validate all metrics against labeled shot footage before using them for scouting decisions. |
| Contact identification is sensitive to camera and detection noise. | Fused velocity spikes with foot proximity and added a diagnostic helper for problematic clips. | Thresholds are hand-tuned and require calibration against ground truth. |
| Goal geometry is unreliable in uncontrolled footage. | Implemented automatic Hough-line goal-corner detection, calibration caching, homography projection, and a clear pixel-based fallback. | Occlusion, poor contrast, non-white goals, and extreme perspective still reduce accuracy. |
| External AI service availability and credentials vary by environment. | Gemini calls are optional and fail safely to a local coaching-note generator. | Store keys outside source control and monitor API errors/costs in deployment. |
| Root integration depends on the submodule’s import paths and MediaPipe API. | The driver resolves the submodule path and verifies a compatible local Python/MediaPipe environment before importing. | Pin and test dependencies in a single reproducible environment. |
| Kalman filter state can carry between batch clips. | The risk is documented for operators. | Reset the module-level Kalman state before each clip; this is the highest-priority correctness fix before production use. |

## Next Steps and Operational Handoff

1. Set up the environment using the root `README.md`, initialize `fc_juggle`, and verify `fc_juggle/models/finetuned.pt` is present.
2. Store `GEMINI_API_KEY` or `GOOGLE_API_KEY` only in a local/deployment secret store. The pipeline remains usable without it.
3. Run a controlled smoke test with:

   ```powershell
   .\venv\Scripts\python.exe video_driver.py
   ```

   Confirm that `results.json` is valid JSON, every expected clip has an entry, and no batch-stopping `error` is present.

4. For new clips, run:

   ```powershell
   .\venv\Scripts\python.exe video_driver.py path\to\clip.mp4 clip_id
   ```

   Review `confidence` before operational use. Treat low-confidence outputs, unknown feet, null angles, and fallback zones as candidates for manual review.

5. Retain `calibrations.json` only while it matches the exact clips and contact frames. Clear it (`{}`) when source footage, clip identifiers, or calibration assumptions change.
6. Before a batch evaluation or deployment, fix and test per-clip Kalman-state reset in `fc_juggle/utils/update_predict.py`. Add a regression test that processes two independent clips sequentially and compares each result with an isolated run.
7. Build a labeled validation set with contact frame, foot, goal zone, speed reference, and outcome labels. Report accuracy and confidence calibration by condition (lighting, goal visibility, camera angle, and player count).
8. Prioritize robust goal detection and multi-person pose association for field use. A goalpost-labeled model and hip landmarks are the clearest upgrades to placement and angle reliability.
9. For API operations, the bundled `fc_juggle` FastAPI service is started from that directory with Uvicorn. It serves juggle counting only; do not present it as a shot-analysis endpoint without an explicit integration effort.

## Handoff Assets

- `README.md` — setup and operator guide
- `video_driver.py` — shot-analysis command-line entry point
- `shot_analysis.py` — core calculation and coaching-note logic
- `results.json` — latest demonstration batch results
- `calibrations.json` — cached goal calibrations
- `MEETING_BRIEF.md` — detailed technical design, assumptions, and known risks
- `fc_juggle/` — included submodule containing the model, tracker, juggle CLI, API, Dockerfile, and Modal deployment definition
