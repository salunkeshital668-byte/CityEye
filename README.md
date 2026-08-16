# CityEye — AI Traffic & CCTV Event Detection

CityEye is a Python-based AI traffic analytics system designed for CCTV and traffic monitoring. It supports **both Image and MP4 Video inputs**, tracking vehicles and pedestrians, and detecting critical safety events.

---

## 🎯 Features & Detection Capabilities

1. **Image Mode (`images/traffic.jpg`)**:
   - Runs Ultralytics YOLOv8 object detection.
   - Detects persons, motorcycles, and vehicles.
   - Spatial association for **Triple Riding Detection** ($\ge 3$ persons on a single motorcycle) without requiring tracking.
   - Draws bounding boxes, class labels, and confidence scores.
   - Saves annotated output to `output/detected_image.jpg`.
   - Gracefully handles missing image files with a clean notice.

2. **Video Mode (`videos/traffic.mp4`)**:
   - Uses **ByteTrack** for persistent multi-object tracking across frames.
   - **Triple Riding Detection**: Multi-person motorcycle violation detection with frame persistence.
   - **Wrong-Way Driving**: Identifies vehicles traveling against expected traffic flow (`EXPECTED_DIRECTION = "RIGHT"`).
   - **Accident / Vehicle Stopped**: Flags stationary vehicles exceeding `STOPPED_FRAMES`.
   - Draws trajectory trails, bounding boxes, labels, confidence, and CCTV HUD overlay.
   - Saves annotated output video to `output/processed_video.mp4` and logs events to `data/events.json`.
   - Gracefully handles missing video files.

---

## 📁 Project Structure

```
CityEye/
│
├── main.py                     # Primary Python CLI runner (Image & Video)
├── detector.py                 # Ultralytics YOLO object detector
├── tracker.py                  # Supervision ByteTrack multi-object tracking wrapper
├── event_detector.py           # Core detection logic for images & video traffic events
├── config.py                   # Central configuration & tunable parameters
├── test_event_logic.py         # Unit & logic test suite
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── images/
│   └── traffic.jpg             # Input image file (tested first)
│
├── videos/
│   └── traffic.mp4             # Input MP4 video file (tested second if present)
│
├── output/
│   ├── detected_image.jpg      # Annotated output image
│   └── processed_video.mp4     # Annotated output video
│
└── data/
    └── events.json             # Stored detection events JSON log
```

---

## 🚀 Running CityEye (Python-Only)

### 1. Default Run (Image first, then Video if available)
```powershell
python main.py
```

### 2. Image-Only Mode
```powershell
python main.py --mode image --image images/traffic.jpg
```

### 3. Video-Only Mode
```powershell
python main.py --mode video --video videos/traffic.mp4
```

### 4. Custom Output Paths
```powershell
python main.py --image images/my_traffic.jpg --output-image output/my_detection.jpg
```

---

## 🧪 Running the Test Suite

Run the comprehensive unit test suite to verify detection logic (single-image triple riding, video triple riding, wrong-way driving, stopped vehicle):

```powershell
python test_event_logic.py
```

---

## ⚙️ Key Configuration Settings (`config.py`)

- `IMAGE_PATH`: Path to default input image (`images/traffic.jpg`)
- `IMAGE_OUTPUT_PATH`: Path to default output image (`output/detected_image.jpg`)
- `VIDEO_PATH`: Path to default input video (`videos/traffic.mp4`)
- `OUTPUT_PATH`: Path to default output video (`output/processed_video.mp4`)
- `CONFIDENCE_THRESHOLD`: YOLO detection confidence threshold (default: `0.35`)
- `PERSON_MOTORCYCLE_DISTANCE`: Horizontal pixel margin to associate persons with motorcycles (default: `80`)
- `EXPECTED_DIRECTION`: Expected traffic flow (`"RIGHT"`, `"LEFT"`, `"UP"`, `"DOWN"`)
- `STOPPED_FRAMES`: Consecutive stationary frames to flag stopped vehicle / accident (default: `30`)
