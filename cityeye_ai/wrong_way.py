from pathlib import Path
import json

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "input.mp4"
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

MODEL_PATH = "yolo11n.pt"

# Prototype assumption:
# Vehicles moving LEFT are considered wrong-way.
# Change to "RIGHT" if your road's correct direction is LEFT.
CORRECT_DIRECTION = "RIGHT"

MIN_MOVEMENT = 8
CONFIRM_FRAMES = 5


def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def run_wrong_way_detection():

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Starting wrong-way detection...")
    print(f"Expected direction: {CORRECT_DIRECTION}")

    results = model.track(
        source=str(VIDEO_PATH),
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        conf=0.35,
        verbose=False
    )

    previous_positions = {}
    wrong_way_frames = {}
    events = []

    frame_number = 0
    fps = 25.0

    for result in results:

        frame_number += 1

        if result.boxes is None:
            continue

        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy()

        for box, cls, track_id in zip(
            boxes,
            classes,
            track_ids
        ):

            class_id = int(cls)
            track_id = int(track_id)

            # COCO vehicle classes:
            # car = 2
            # motorcycle = 3
            # bus = 5
            # truck = 7

            if class_id not in [2, 3, 5, 7]:
                continue

            current_center = get_center(box)

            if track_id in previous_positions:

                previous_center = previous_positions[track_id]

                dx = current_center[0] - previous_center[0]

                # ----------------------------------------
                # Direction check
                # ----------------------------------------

                if abs(dx) >= MIN_MOVEMENT:

                    moving_right = dx > 0

                    wrong_direction = (
                        CORRECT_DIRECTION == "RIGHT"
                        and not moving_right
                    ) or (
                        CORRECT_DIRECTION == "LEFT"
                        and moving_right
                    )

                    if wrong_direction:

                        count = wrong_way_frames.get(
                            track_id,
                            0
                        )

                        wrong_way_frames[track_id] = count + 1

                        # Confirm only after several frames
                        if (
                            wrong_way_frames[track_id]
                            == CONFIRM_FRAMES
                        ):

                            vehicle_names = {
                                2: "car",
                                3: "motorcycle",
                                5: "bus",
                                7: "truck"
                            }

                            event = {
                                "type": "WRONG_WAY",
                                "frame": frame_number,
                                "timestamp": round(
                                    frame_number / fps,
                                    2
                                ),
                                "track_id": track_id,
                                "vehicle": vehicle_names[class_id],
                                "direction": "LEFT"
                                if not moving_right
                                else "RIGHT"
                            }

                            events.append(event)

                            print(
                                "WRONG WAY detected | "
                                f"frame={frame_number} | "
                                f"vehicle={vehicle_names[class_id]} | "
                                f"id={track_id}"
                            )

                    else:
                        wrong_way_frames[track_id] = 0

            previous_positions[track_id] = current_center

    # ----------------------------------------
    # Save JSON
    # ----------------------------------------

    output = {
        "video": str(VIDEO_PATH),
        "event": "WRONG_WAY",
        "correct_direction": CORRECT_DIRECTION,
        "detected": len(events) > 0,
        "count": len(events),
        "events": events
    }

    output_file = DATA_DIR / "wrong_way_events.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print("\n--------------------------------")
    print("Wrong-Way Detection Complete")
    print("--------------------------------")

    print(f"Wrong-way detections: {len(events)}")
    print(f"JSON saved to: {output_file}")


if __name__ == "__main__":
    run_wrong_way_detection()