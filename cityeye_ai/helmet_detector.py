from pathlib import Path
import json

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "input.mp4"
MODEL_PATH = BASE_DIR / "models" / "helmet_best.pt"

OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def run_helmet_detection():

    print("Loading helmet model...")
    model = YOLO(str(MODEL_PATH))

    print("Model classes:")
    print(model.names)

    print(f"\nInput video: {VIDEO_PATH}")
    print("Starting helmet detection...\n")

    results = model.predict(
        source=str(VIDEO_PATH),
        save=True,
        project=str(OUTPUT_DIR),
        name="helmet_detection",
        conf=0.35,
        verbose=True
    )

    events = []

    for frame_number, result in enumerate(results, start=1):

        if result.boxes is None:
            continue

        if len(result.boxes) == 0:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]
            if (
    "helmet" in class_name.lower()
    and (
        "no" in class_name.lower()
        or "without" in class_name.lower()
    )
):


                event = {
                    "type": "NO_HELMET",
                    "frame": frame_number,
                    "confidence": round(confidence, 3)
                }

                events.append(event)

                print(
                    f"NO HELMET detected | "
                    f"frame={frame_number} | "
                    f"confidence={confidence:.2f}"
                )

    output = {
        "video": str(VIDEO_PATH),
        "event": "NO_HELMET",
        "detected": len(events) > 0,
        "count": len(events),
        "events": events
    }

    output_file = DATA_DIR / "helmet_events.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("\n--------------------------------")
    print("Helmet Detection Complete")
    print("--------------------------------")

    print(f"No Helmet detections: {len(events)}")
    print(f"JSON saved to: {output_file}")
    print(
        f"Video output: "
        f"{OUTPUT_DIR / 'helmet_detection'}"
    )


if __name__ == "__main__":
    run_helmet_detection()