from pathlib import Path
import json

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "input.mp4"
OUTPUT_DIR = BASE_DIR / "output"
EVENT_DIR = BASE_DIR / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVENT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_NAME = "yolo11n.pt"

# Minimum number of people associated with one motorcycle
TRIPLE_RIDING_THRESHOLD = 3

# How many consecutive frames should confirm the event
CONFIRM_FRAMES = 5


def center(box):
    """Return center point of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_inside_expanded_box(point, box, padding=0.35):
    """
    Checks whether a person's center is inside
    an expanded motorcycle bounding box.
    """

    px, py = point
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    x1 -= width * padding
    x2 += width * padding

    y1 -= height * padding
    y2 += height * padding

    return x1 <= px <= x2 and y1 <= py <= y2


def detect_triple_riding():

    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    event_details = []

    consecutive_frames = {}

    frame_number = 0
    fps = 25.0

    print("Starting Triple Riding detection...")
    print(f"Input: {VIDEO_PATH}")

    results = model.track(
        source=str(VIDEO_PATH),
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        conf=0.35,
        verbose=False
    )

    for result in results:

        frame_number += 1

        if result.boxes is None:
            continue

        boxes = result.boxes

        if boxes.xyxy is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy()
        else:
            track_ids = [-1] * len(xyxy)

        motorcycles = []
        persons = []

        # --------------------------------------------------
        # Separate motorcycles and persons
        # --------------------------------------------------

        for box, cls, track_id in zip(xyxy, classes, track_ids):

            class_id = int(cls)

            # COCO:
            # person = 0
            # motorcycle = 3

            if class_id == 3:
                motorcycles.append(
                    {
                        "box": box,
                        "track_id": int(track_id)
                    }
                )

            elif class_id == 0:
                persons.append(
                    {
                        "box": box,
                        "track_id": int(track_id)
                    }
                )

        # --------------------------------------------------
        # Associate persons with motorcycles
        # --------------------------------------------------

        for motorcycle in motorcycles:

            motorcycle_box = motorcycle["box"]
            motorcycle_id = motorcycle["track_id"]

            passenger_count = 0
            passenger_ids = []

            for person in persons:

                person_center = center(person["box"])

                if point_inside_expanded_box(
                    person_center,
                    motorcycle_box
                ):

                    passenger_count += 1
                    passenger_ids.append(person["track_id"])

            # --------------------------------------------------
            # Triple riding condition
            # --------------------------------------------------

            if passenger_count >= TRIPLE_RIDING_THRESHOLD:

                previous_count = consecutive_frames.get(
                    motorcycle_id,
                    0
                )

                consecutive_frames[motorcycle_id] = previous_count + 1

                if (
                    consecutive_frames[motorcycle_id]
                    == CONFIRM_FRAMES
                ):

                    timestamp = frame_number / fps

                    event = {
                        "type": "TRIPLE_RIDING",
                        "frame": frame_number,
                        "timestamp": round(timestamp, 2),
                        "vehicle": "motorcycle",
                        "track_id": motorcycle_id,
                        "passenger_count": passenger_count,
                        "passenger_ids": passenger_ids
                    }

                    event_details.append(event)

                    print(
                        f"TRIPLE RIDING detected | "
                        f"frame={frame_number} | "
                        f"motorcycle_id={motorcycle_id} | "
                        f"persons={passenger_count}"
                    )

            else:

                # Reset confirmation if condition disappears
                consecutive_frames[motorcycle_id] = 0

    # ------------------------------------------------------
    # Save result
    # ------------------------------------------------------

    output = {
        "video": str(VIDEO_PATH),
        "event": "TRIPLE_RIDING",
        "detected": len(event_details) > 0,
        "count": len(event_details),
        "events": event_details
    }

    output_file = EVENT_DIR / "triple_riding_events.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\n--------------------------------")
    print("Triple Riding Detection Complete")
    print("--------------------------------")

    print(f"Detected events: {len(event_details)}")
    print(f"JSON saved to: {output_file}")


if __name__ == "__main__":
    detect_triple_riding()