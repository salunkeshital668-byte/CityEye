from pathlib import Path
import json
import math

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "input.mp4"
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

MODEL_PATH = "yolo11n.pt"

# Vehicles: car, motorcycle, bus, truck
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Distance threshold in pixels.
# This is a prototype value and may need tuning for another camera.
DISTANCE_THRESHOLD = 45

# Number of consecutive frames required
# before declaring a possible accident.
CONFIRM_FRAMES = 5


def get_center(box):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def run_accident_detection():

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Starting accident detection...")
    print(f"Input video: {VIDEO_PATH}")

    results = model.track(
        source=str(VIDEO_PATH),
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        conf=0.35,
        verbose=False
    )

    frame_number = 0
    fps = 25.0

    close_frames = {}
    events = []

    for result in results:

        frame_number += 1

        if result.boxes is None:
            continue

        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy()

        vehicles = []

        for box, cls, track_id in zip(
            boxes,
            classes,
            track_ids
        ):

            class_id = int(cls)

            if class_id not in VEHICLE_CLASSES:
                continue

            vehicles.append({
                "box": box,
                "class_id": class_id,
                "track_id": int(track_id),
                "center": get_center(box)
            })

        # ---------------------------------------------
        # Compare every vehicle pair
        # ---------------------------------------------

        for i in range(len(vehicles)):

            for j in range(i + 1, len(vehicles)):

                vehicle1 = vehicles[i]
                vehicle2 = vehicles[j]

                id1 = vehicle1["track_id"]
                id2 = vehicle2["track_id"]

                d = distance(
                    vehicle1["center"],
                    vehicle2["center"]
                )

                pair_key = tuple(sorted([id1, id2]))

                # -------------------------------------
                # Vehicles are unusually close
                # -------------------------------------

                if d < DISTANCE_THRESHOLD:

                    count = close_frames.get(
                        pair_key,
                        0
                    )

                    close_frames[pair_key] = count + 1

                    # Confirm only after consecutive frames
                    if (
                        close_frames[pair_key]
                        == CONFIRM_FRAMES
                    ):

                        event = {
                            "type": "POSSIBLE_ACCIDENT",
                            "frame": frame_number,
                            "timestamp": round(
                                frame_number / fps,
                                2
                            ),
                            "vehicle_1": {
                                "type": VEHICLE_CLASSES[
                                    vehicle1["class_id"]
                                ],
                                "track_id": id1
                            },
                            "vehicle_2": {
                                "type": VEHICLE_CLASSES[
                                    vehicle2["class_id"]
                                ],
                                "track_id": id2
                            },
                            "distance": round(d, 2)
                        }

                        events.append(event)

                        print(
                            "POSSIBLE ACCIDENT detected | "
                            f"frame={frame_number} | "
                            f"{VEHICLE_CLASSES[vehicle1['class_id']]} "
                            f"(ID {id1}) + "
                            f"{VEHICLE_CLASSES[vehicle2['class_id']]} "
                            f"(ID {id2})"
                        )

                else:

                    # Vehicles moved apart
                    close_frames[pair_key] = 0

    # ---------------------------------------------
    # Save result
    # ---------------------------------------------

    output = {
        "video": str(VIDEO_PATH),
        "event": "ACCIDENT",
        "detected": len(events) > 0,
        "count": len(events),
        "events": events
    }

    output_file = DATA_DIR / "accident_events.json"

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
    print("Accident Detection Complete")
    print("--------------------------------")

    print(
        f"Possible accident events: {len(events)}"
    )

    print(
        f"JSON saved to: {output_file}"
    )


if __name__ == "__main__":
    run_accident_detection()