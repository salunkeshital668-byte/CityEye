import os
import time
import json
import threading
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import config
from detector import YOLODetector
from tracker import MultiObjectTracker
from event_detector import EventDetector

# Ensure required directories exist
for directory in [config.VIDEOS_DIR, config.OUTPUT_DIR, config.DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

app = FastAPI(
    title="CityEye AI Video Analytics",
    description="Real-time CCTV & Traffic AI Event Detection System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and output directories
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/output", StaticFiles(directory=str(config.OUTPUT_DIR)), name="output")

# Processing state tracking
processing_lock = threading.Lock()
pipeline_state = {
    "is_processing": False,
    "progress_pct": 0,
    "current_frame": 0,
    "total_frames": 0,
    "latest_message": "Idle",
    "last_run_summary": None
}

# Global instances (lazy-loaded or initialized)
detector_instance = None
tracker_instance = None
event_detector_instance = None


def get_detector():
    global detector_instance
    if detector_instance is None:
        detector_instance = YOLODetector()
    return detector_instance


def process_video_pipeline(video_path: str, output_path: str) -> dict:
    """
    Core video processing pipeline: reads MP4, detects objects with YOLO,
    tracks with ByteTrack, detects 4 traffic events, and writes output video.
    """
    global pipeline_state, tracker_instance, event_detector_instance

    if not os.path.exists(video_path):
        err_msg = "traffic.mp4 not found. Put an MP4 traffic video inside videos/traffic.mp4"
        print(f"[CityEye Error] {err_msg}")
        return {
            "status": "error",
            "message": err_msg,
            "events_count": 0
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        err_msg = f"Failed to open video file: {video_path}"
        print(f"[CityEye Error] {err_msg}")
        return {"status": "error", "message": err_msg, "events_count": 0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer (using mp4v / avc1 compatible codec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    det = get_detector()
    tracker_instance = MultiObjectTracker()
    event_detector_instance = EventDetector(det, tracker_instance, config.CAMERA_ID)

    pipeline_state["is_processing"] = True
    pipeline_state["total_frames"] = total_frames
    pipeline_state["current_frame"] = 0
    pipeline_state["latest_message"] = "Processing video frames..."

    frame_no = 0
    start_time = time.time()
    events_logged = 0

    print(f"\n[CityEye Pipeline] Started processing: {video_path} ({total_frames} frames, {width}x{height} @ {fps:.1f} FPS)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_no += 1
            annotated_frame, alerts, new_events = event_detector_instance.process_frame(frame, frame_no, fps)
            writer.write(annotated_frame)

            events_logged += len(new_events)

            if total_frames > 0:
                pct = int((frame_no / total_frames) * 100)
                pipeline_state["progress_pct"] = min(100, pct)
            pipeline_state["current_frame"] = frame_no

            if frame_no % 30 == 0:
                print(f"[CityEye] Frame {frame_no}/{total_frames} ({pipeline_state['progress_pct']}%) - Events so far: {len(event_detector_instance.events)}")

    finally:
        cap.release()
        writer.release()

    elapsed = round(time.time() - start_time, 2)
    stats = event_detector_instance.get_summary_statistics()

    summary = {
        "status": "success",
        "message": f"Successfully processed {frame_no} frames in {elapsed}s",
        "total_frames_processed": frame_no,
        "elapsed_seconds": elapsed,
        "output_video": output_path,
        "statistics": stats,
        "events_count": len(event_detector_instance.events)
    }

    pipeline_state["is_processing"] = False
    pipeline_state["progress_pct"] = 100
    pipeline_state["latest_message"] = "Processing complete."
    pipeline_state["last_run_summary"] = summary

    print("\n========================================")
    print("  CITYEYE AI CCTV PROCESSING COMPLETE   ")
    print("========================================")
    print(f"Total Frames Processed: {frame_no}")
    print(f"Time Taken:             {elapsed}s")
    print(f"Total Events Detected:  {len(event_detector_instance.events)}")
    print(f"Output Video Saved:     {output_path}")
    print("========================================\n")

    return summary


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """Serves the main CityEye CCTV Dashboard UI."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h2>CityEye Dashboard template not found</h2>", status_code=404)
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
def health_check():
    """Returns API and system health status."""
    video_exists = os.path.exists(config.VIDEO_PATH)
    output_exists = os.path.exists(config.OUTPUT_PATH)
    model_exists = os.path.exists(config.MODEL_PATH)

    return {
        "status": "ok",
        "camera_id": config.CAMERA_ID,
        "traffic_video_available": video_exists,
        "processed_output_available": output_exists,
        "model_file_exists": model_exists,
        "model_path": config.MODEL_PATH,
        "helmet_detection_enabled": config.HELMET_MODEL_PATH is not None,
        "pipeline_state": pipeline_state
    }


@app.get("/events")
def get_events():
    """Returns stored events from data/events.json and aggregate statistics."""
    events = []
    if os.path.exists(config.EVENTS_JSON_PATH):
        try:
            with open(config.EVENTS_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                events = data.get("events", [])
        except Exception as e:
            print(f"[API] Error reading events.json: {e}")

    # Compute breakdown statistics
    triple_riding = sum(1 for e in events if e.get("event") == "triple_riding")
    wrong_way = sum(1 for e in events if e.get("event") == "wrong_way_driving")
    stopped = sum(1 for e in events if e.get("event") == "vehicle_stopped")
    helmet = sum(1 for e in events if e.get("event") == "helmet_violation")

    # Get distinct vehicle IDs and person IDs if available
    unique_vehicles = len(set(e.get("vehicle_id") for e in events if "vehicle_id" in e))

    return {
        "total_events": len(events),
        "statistics": {
            "triple_riding": triple_riding,
            "wrong_way_driving": wrong_way,
            "vehicle_stopped": stopped,
            "helmet_violation": helmet,
            "total_vehicles": unique_vehicles,
            "total_persons": triple_riding * 3  # estimate from events if offline
        },
        "events": list(reversed(events))  # Return newest events first
    }


@app.get("/status")
def get_pipeline_status():
    """Returns real-time progress status of the video analysis pipeline."""
    return pipeline_state


@app.post("/process")
def trigger_process():
    """
    Triggers video processing on videos/traffic.mp4 asynchronously in a background thread.
    Returns immediately so the browser UI can display live progress without timeout.
    """
    if not os.path.exists(config.VIDEO_PATH):
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": "traffic.mp4 not found. Put an MP4 traffic video inside videos/traffic.mp4",
                "expected_path": config.VIDEO_PATH,
                "events_count": 0
            }
        )

    if pipeline_state["is_processing"]:
        return {
            "status": "in_progress",
            "message": "Video analysis is already running.",
            "progress_pct": pipeline_state["progress_pct"]
        }

    # Reset pipeline state
    pipeline_state["is_processing"] = True
    pipeline_state["progress_pct"] = 0
    pipeline_state["current_frame"] = 0
    pipeline_state["total_frames"] = 0
    pipeline_state["latest_message"] = "Initializing YOLO & ByteTrack..."
    pipeline_state["last_run_summary"] = None

    # Start processing in background thread
    worker_thread = threading.Thread(
        target=process_video_pipeline,
        args=(config.VIDEO_PATH, config.OUTPUT_PATH),
        daemon=True
    )
    worker_thread.start()

    return {
        "status": "started",
        "message": "AI Video processing started in background.",
        "video_path": config.VIDEO_PATH
    }


@app.get("/output-video")
def get_processed_video():
    """Serves the processed video file for the frontend player."""
    if not os.path.exists(config.OUTPUT_PATH):
        raise HTTPException(status_code=404, detail="Processed video not found. Run /process first.")
    return FileResponse(config.OUTPUT_PATH, media_type="video/mp4")


@app.post("/create-sample-video")
def create_sample_video_endpoint():
    """
    Helper endpoint to create a synthetic CCTV test video containing
    Helmet violation, Triple-riding motorcycle, Wrong-way car, and Stopped car.
    """
    try:
        from generate_sample import generate_cctv_sample
        out_file = generate_cctv_sample(config.VIDEO_PATH)
        return {
            "status": "success",
            "message": f"Sample CCTV traffic video generated at {out_file}",
            "video_path": out_file
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
