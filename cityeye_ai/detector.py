from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "input.mp4"
OUTPUT_DIR = BASE_DIR / "output"


def run_detection():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO model...")

    model = YOLO("yolo11n.pt")

    print(f"Input video: {VIDEO_PATH}")
    print("Starting object detection...")

    results = model.predict(
        source=str(VIDEO_PATH),
        save=True,
        project=str(OUTPUT_DIR),
        name="cityeye_detection",
        conf=0.35,
        verbose=True
    )

    print("\nYOLO detection completed.")
    print(f"Output directory: {OUTPUT_DIR / 'cityeye_detection'}")

    return results


if __name__ == "__main__":
    run_detection()