import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent

# File paths
IMAGES_DIR = BASE_DIR / "images"
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

IMAGE_PATH = str(IMAGES_DIR / "traffic.jpg")
IMAGE_OUTPUT_PATH = str(OUTPUT_DIR / "detected_image.jpg")
VIDEO_PATH = str(VIDEOS_DIR / "traffic.mp4")
VIDEO_PATH = str(VIDEOS_DIR / "actual_video.mp4")
EVENTS_JSON_PATH = str(DATA_DIR / "events.json")

# Model configuration
# Lightweight YOLO model for general traffic object detection
MODEL_PATH = str(BASE_DIR / "yolov8n.pt")

# Real Helmet YOLO model path
HELMET_MODEL_PATH = str(MODELS_DIR / "helmet_model.pt")

# Detection settings
CONFIDENCE_THRESHOLD = 0.35
HELMET_CONFIDENCE_THRESHOLD = 0.30
CAMERA_ID = "cam_01"

# COCO Class IDs
CLASS_PERSON = 0
CLASS_CAR = 2
CLASS_MOTORCYCLE = 3
CLASS_BUS = 5
CLASS_TRUCK = 7

VEHICLE_CLASS_IDS = {CLASS_CAR, CLASS_MOTORCYCLE, CLASS_BUS, CLASS_TRUCK}
ALL_TRACKED_CLASS_IDS = {CLASS_PERSON, CLASS_CAR, CLASS_MOTORCYCLE, CLASS_BUS, CLASS_TRUCK}

# Event detection parameters
# Expected traffic flow direction: "LEFT", "RIGHT", "UP", "DOWN"
EXPECTED_DIRECTION = "RIGHT"

# Minimum pixel displacement to determine direction
MIN_MOVEMENT = 15.0

# Number of consecutive frames a vehicle must be stationary to trigger stopped/accident alert
STOPPED_FRAMES = 30

# Maximum speed (pixels/frame) below which a vehicle is considered stationary
STOPPED_SPEED_THRESHOLD = 2.0

# Number of consecutive frames 3+ persons must be on the same bike
TRIPLE_RIDING_FRAMES = 5

# Horizontal margin (pixels) to associate person with motorcycle
PERSON_MOTORCYCLE_DISTANCE = 80

# Cooldown frames to prevent spamming duplicate events for the same track
EVENT_COOLDOWN = 60

# History length for trajectory analysis
TRAJECTORY_HISTORY = 30
