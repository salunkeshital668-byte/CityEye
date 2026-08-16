import cv2
import json
import argparse
from collections import defaultdict, deque

import numpy as np
import supervision as sv
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
CONF = 0.35

# COCO classes
PERSON = 0
MOTORCYCLE = 3
CAR = 2
BUS = 5
TRUCK = 7

VEHICLES = {CAR, MOTORCYCLE, BUS, TRUCK}

STOP_TIME = 5
STOP_SPEED = 2.0
HISTORY = 12

# Camera traffic direction
# Change to "left" if required
EXPECTED_DIRECTION = "right"


def center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1+x2)/2, (y1+y2)/2])


def direction(points):
    if len(points) < 5:
        return None

    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]

    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"

    return "down" if dy > 0 else "up"


def person_on_motorcycle(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2

    # person center should be reasonably close to motorcycle
    return (
        bx1 - 80 <= pcx <= bx2 + 80
        and
        by1 - 180 <= pcy <= by2 + 100
    )


def run(source, output):

    print("Loading YOLO...")
    model = YOLO(MODEL_PATH)

    print("Starting ByteTrack...")
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("ERROR: Video cannot be opened")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    positions = defaultdict(
        lambda: deque(maxlen=HISTORY)
    )

    stopped_since = {}
    stopped_flagged = set()
    wrong_way_flagged = set()
    triple_flagged = set()

    events = []

    frame_no = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        result = model(
            frame,
            conf=CONF,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(result)

        # Save all detections for triple-seat analysis
        all_boxes = detections.xyxy.copy()
        all_classes = detections.class_id.copy()

        # Vehicle detections for tracking
        if len(detections) > 0:

            mask = np.isin(
                detections.class_id,
                list(VEHICLES)
            )

            vehicle_detections = detections[mask]

        else:
            vehicle_detections = detections

        vehicle_detections = tracker.update_with_detections(
            vehicle_detections
        )

        # =====================================
        # TRIPLE SEAT DETECTION
        # =====================================

        person_boxes = []

        for box, cls in zip(all_boxes, all_classes):

            if int(cls) == PERSON:
                person_boxes.append(box)

        for i in range(len(vehicle_detections)):

            cls = int(vehicle_detections.class_id[i])

            if cls != MOTORCYCLE:
                continue

            bike_box = vehicle_detections.xyxy[i]

            bike_id = vehicle_detections.tracker_id[i]

            if bike_id is None:
                continue

            bike_id = int(bike_id)

            persons = 0

            for pbox in person_boxes:

                if person_on_motorcycle(
                    pbox,
                    bike_box
                ):
                    persons += 1

            if persons >= 3:

                if bike_id not in triple_flagged:

                    event = {
                        "event": "triple_seat_violation",
                        "track_id": bike_id,
                        "persons_detected": persons,
                        "severity": "medium",
                        "frame": frame_no
                    }

                    events.append(event)

                    triple_flagged.add(bike_id)

                    print("[EVENT]", event)

        # =====================================
        # VEHICLE EVENTS
        # =====================================

        for i in range(len(vehicle_detections)):

            box = vehicle_detections.xyxy[i]

            cls = int(vehicle_detections.class_id[i])

            track_id = vehicle_detections.tracker_id[i]

            if track_id is None:
                continue

            track_id = int(track_id)

            positions[track_id].append(
                center(box)
            )

            name = model.names.get(
                cls,
                str(cls)
            )

            label = f"{name} ID:{track_id}"

            # =================================
            # STOPPED VEHICLE
            # =================================

            if len(positions[track_id]) >= 2:

                p1 = positions[track_id][-2]
                p2 = positions[track_id][-1]

                speed = np.linalg.norm(p2 - p1)

                if speed < STOP_SPEED:

                    if track_id not in stopped_since:
                        stopped_since[track_id] = (
                            frame_no / fps
                        )

                    stopped_time = (
                        frame_no / fps
                        - stopped_since[track_id]
                    )

                    if stopped_time >= STOP_TIME:

                        label += " POSSIBLE ACCIDENT"

                        if track_id not in stopped_flagged:

                            event = {
                                "event":
                                "vehicle_stopped_possible_accident",

                                "track_id": track_id,

                                "severity": "high",

                                "stopped_seconds":
                                round(stopped_time, 1),

                                "frame": frame_no
                            }

                            events.append(event)

                            stopped_flagged.add(track_id)

                            print("[EVENT]", event)

                else:

                    stopped_since.pop(
                        track_id,
                        None
                    )

            # =================================
            # WRONG WAY
            # =================================

            move = direction(
                positions[track_id]
            )

            if move:

                wrong = (
                    EXPECTED_DIRECTION == "right"
                    and move == "left"
                ) or (
                    EXPECTED_DIRECTION == "left"
                    and move == "right"
                )

                if wrong:

                    label += " WRONG WAY"

                    if track_id not in wrong_way_flagged:

                        event = {
                            "event": "wrong_way_driving",
                            "track_id": track_id,
                            "direction": move,
                            "severity": "high",
                            "frame": frame_no
                        }

                        events.append(event)

                        wrong_way_flagged.add(track_id)

                        print("[EVENT]", event)

            # =================================
            # DRAW VEHICLE
            # =================================

            x1, y1, x2, y2 = map(
                int,
                box
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "CityEye Event Detection",
            frame
        )

        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_no += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # =====================================
    # SAVE EVENTS
    # =====================================

    with open(
        "events.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "camera_id": "cam_01",
                "total_events": len(events),
                "events": events
            },
            f,
            indent=4
        )

    print("\n============================")
    print("CITYEYE PROCESSING COMPLETE")
    print("============================")
    print("Output:", output)
    print("Events: events.json")
    print("Total events:", len(events))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True
    )

    parser.add_argument(
        "--output",
        default="event_output.mp4"
    )

    args = parser.parse_args()

    source = args.source

    if source.isdigit():
        source = int(source)

    run(
        source,
        args.output
    )