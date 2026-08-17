"""Modal container definition for the shot-analysis service."""

import os
import tempfile

import modal
from fastapi import FastAPI, File, HTTPException, UploadFile


# Define the custom Modal container image with required dependencies.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "fastapi",
        "python-multipart",
        "opencv-python-headless",
        "mediapipe",
        "ultralytics",
        "numpy",
        "torch",
        "torchvision",
    )
)


app = modal.App("shot-analysis-service", image=image)

web_app = FastAPI(title="FutbolConnect Shot Analysis API")


def run_pipeline(video_path: str) -> dict:
    """Run shot analysis for a locally staged video.

    Invokes video_driver.run_on_video() which processes keypoints via YOLO &
    MediaPipe, performs homography calibration, calculates physics & biomechanics,
    gates acceptance, and generates a Gemini coaching note.
    """
    from video_driver import run_on_video

    clip_id = os.path.splitext(os.path.basename(video_path))[0]
    return run_on_video(video_path=video_path, clip_id=clip_id)


@web_app.post("/shot-analyze")
async def shot_analyze(video: UploadFile = File(...)):
    """Analyze a multipart video upload."""
    filename = video.filename or ""
    if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format.")

    tmp_path = ""
    try:
        # Save the multipart file to a temporary path for pipeline consumption.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(await video.read())
            tmp_path = tmp_file.name

        return run_pipeline(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.function(
    gpu="A10G",
    timeout=300,
    secrets=[modal.Secret.from_name("gemini-api-key")],
)
@modal.asgi_app()
def fastapi_app():
    """Expose the FastAPI application through Modal."""
    return web_app
