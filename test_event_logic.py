import os
import json
import numpy as np
import cv2
import supervision as sv
import config
from detector import YOLODetector
from tracker import MultiObjectTracker
from event_detector import EventDetector


def test_helmet_detection_logic():
    print("\n--- Testing Helmet / No-Helmet Detection Logic ---")
    det = YOLODetector()
    tracker = MultiObjectTracker()
    event_det = EventDetector(det, tracker, "cam_01")
    event_det.events = []

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rider_box = np.array([200, 100, 300, 350])
    bike_box = np.array([180, 250, 420, 500])

    # 1. Test when helmet model is not present (no faking, graceful handling)
    res = det.check_rider_helmet(dummy_frame, rider_box, bike_box)
    assert not res["available"], "Helmet check must report available=False when model is not loaded"
    print("✓ Graceful missing helmet model handling verified (no faking).")

    # 2. Test Helmet Detection with mock helmet model
    # Simulate a loaded helmet YOLO model with classes {0: 'helmet', 1: 'no_helmet'}
    det.helmet_model_available = True
    det.helmet_class_names = {0: "helmet", 1: "no_helmet"}
    det.helmet_classes = {0}
    det.no_helmet_classes = {1}

    class MockBox:
        def __init__(self, cls_id, conf):
            self.cls = [cls_id]
            self.conf = [conf]

    class MockResult:
        def __init__(self, boxes):
            self.boxes = boxes

    # Case A: Helmet detected (cls 0)
    det.helmet_model = lambda crop, **kwargs: [MockResult([MockBox(0, 0.93)])]
    res_helmet = det.check_rider_helmet(dummy_frame, rider_box, bike_box)
    assert res_helmet["available"] is True
    assert res_helmet["status"] == "HELMET"
    assert res_helmet["confidence"] == 0.93
    print(f"✓ HELMET detection verified: {res_helmet}")

    # Case B: No Helmet detected (cls 1)
    det.helmet_model = lambda crop, **kwargs: [MockResult([MockBox(1, 0.88)])]
    res_no_helmet = det.check_rider_helmet(dummy_frame, rider_box, bike_box)
    assert res_no_helmet["available"] is True
    assert res_no_helmet["status"] == "NO HELMET"
    assert res_no_helmet["confidence"] == 0.88
    print(f"✓ NO HELMET detection verified: {res_no_helmet}")

    # 3. Test image processing with No Helmet logging helmet_violation event
    test_img_path = os.path.join(config.DATA_DIR, "test_helmet_traffic.jpg")
    test_out_path = os.path.join(config.OUTPUT_DIR, "test_helmet_output.jpg")
    cv2.imwrite(test_img_path, dummy_frame)

    orig_detect = det.detect
    try:
        mock_boxes = np.array([bike_box, rider_box], dtype=np.float32)
        mock_classes = np.array([config.CLASS_MOTORCYCLE, config.CLASS_PERSON], dtype=int)
        mock_confs = np.array([0.92, 0.89], dtype=np.float32)
        det.detect = lambda frame: sv.Detections(xyxy=mock_boxes, confidence=mock_confs, class_id=mock_classes)

        event_det.events = []
        img_res = event_det.process_image(test_img_path, test_out_path)
        assert img_res["status"] == "success"
        
        # Check that helmet violation event was recorded
        helmet_events = [e for e in event_det.events if e["event"] == "helmet_violation"]
        assert len(helmet_events) > 0, "No-helmet detection must log a 'helmet_violation' event"
        assert helmet_events[0]["status"] == "NO HELMET"
        print(f"✓ Helmet violation event recorded in events: {helmet_events[0]}")
    finally:
        det.detect = orig_detect
        det.helmet_model_available = False
        det.helmet_model = None
        if os.path.exists(test_img_path):
            os.remove(test_img_path)
        if os.path.exists(test_out_path):
            os.remove(test_out_path)


def test_single_image_triple_riding_logic():
    print("\n--- Testing Single Image Triple Riding Logic (No Tracking) ---")
    det = YOLODetector()
    tracker = MultiObjectTracker()
    event_det = EventDetector(det, tracker, "cam_01")
    event_det.events = []

    # 1. Test geometric association directly
    bike_box = np.array([200, 300, 450, 500])
    p1_box = np.array([220, 200, 290, 400])
    p2_box = np.array([280, 190, 350, 390])
    p3_box = np.array([340, 200, 410, 400])
    p_far_box = np.array([800, 300, 870, 500])

    assert event_det._is_person_on_motorcycle(p1_box, bike_box), "p1 should be on motorcycle"
    assert event_det._is_person_on_motorcycle(p2_box, bike_box), "p2 should be on motorcycle"
    assert event_det._is_person_on_motorcycle(p3_box, bike_box), "p3 should be on motorcycle"
    assert not event_det._is_person_on_motorcycle(p_far_box, bike_box), "p_far should NOT be on motorcycle"

    # 2. Test missing image handling
    missing_result = event_det.process_image("non_existent_image_path.jpg")
    assert missing_result["status"] == "error"
    assert "not found" in missing_result["message"]
    print("✓ Missing image handling verified successfully.")

    # 3. Create a mock test image and run process_image with mock detector
    test_img_path = os.path.join(config.DATA_DIR, "test_mock_traffic.jpg")
    test_out_path = os.path.join(config.OUTPUT_DIR, "test_mock_output.jpg")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imwrite(test_img_path, dummy_img)

    original_detect = det.detect
    try:
        mock_boxes = np.array([bike_box, p1_box, p2_box, p3_box, p_far_box], dtype=np.float32)
        mock_classes = np.array([config.CLASS_MOTORCYCLE, config.CLASS_PERSON, config.CLASS_PERSON, config.CLASS_PERSON, config.CLASS_PERSON], dtype=int)
        mock_confs = np.array([0.91, 0.88, 0.86, 0.89, 0.94], dtype=np.float32)
        det.detect = lambda frame: sv.Detections(xyxy=mock_boxes, confidence=mock_confs, class_id=mock_classes)

        res = event_det.process_image(test_img_path, test_out_path)
        assert res["status"] == "success"
        assert res["persons_detected"] == 4
        assert res["motorcycles_detected"] == 1
        assert res["triple_riding_count"] == 1
        assert os.path.exists(test_out_path), "Annotated image output file must exist"
        print(f"✓ Single-image Triple Riding detection verified! Persons: {res['persons_detected']}, Motorcycles: {res['motorcycles_detected']}, Triple Riding: {res['triple_riding_count']}")
    finally:
        det.detect = original_detect
        if os.path.exists(test_img_path):
            os.remove(test_img_path)
        if os.path.exists(test_out_path):
            os.remove(test_out_path)


def test_triple_riding_detection():
    print("\n--- Testing Video Triple Riding Detection (Multi-frame Tracking) ---")
    det = YOLODetector()
    tracker = MultiObjectTracker()
    event_det = EventDetector(det, tracker, "cam_01")
    event_det.events = []

    bike_box = np.array([100, 100, 300, 250])
    p1_box = np.array([110, 50, 160, 180])
    p2_box = np.array([160, 50, 210, 180])
    p3_box = np.array([210, 50, 260, 180])

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for frame_no in range(1, 8):
        boxes = np.array([bike_box, p1_box, p2_box, p3_box], dtype=np.float32)
        class_ids = np.array([config.CLASS_MOTORCYCLE, config.CLASS_PERSON, config.CLASS_PERSON, config.CLASS_PERSON], dtype=int)
        confidences = np.array([0.92, 0.88, 0.89, 0.91], dtype=np.float32)
        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids
        )
        detections.tracker_id = np.array([10, 101, 102, 103])
        tracker.trajectories[10].append((200.0, 175.0))
        tracker.all_seen_vehicles.add(10)

        _, alerts, new_events = event_det.process_frame(dummy_frame, frame_no, tracked_detections=detections)

    triple_events = [e for e in event_det.events if e["event"] == "triple_riding"]
    assert len(triple_events) > 0, "Triple riding event must be triggered after consecutive frames"
    ev = triple_events[0]
    assert ev["vehicle_id"] == 10
    assert ev["person_count"] == 3
    print(f"✓ Video Triple Riding successfully detected! Event: {ev}")


def test_wrong_way_detection():
    print("\n--- Testing Video Wrong-Way Driving Logic ---")
    det = YOLODetector()
    tracker = MultiObjectTracker()
    config.EXPECTED_DIRECTION = "RIGHT"
    event_det = EventDetector(det, tracker, "cam_01")
    event_det.events = []

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Vehicle ID 20 moving LEFT (x decreases from 400 to 100)
    for frame_no in range(1, 15):
        x = 400 - (frame_no * 15)
        car_box = np.array([x, 200, x + 100, 260], dtype=np.float32)
        detections = sv.Detections(
            xyxy=np.array([car_box]),
            confidence=np.array([0.90]),
            class_id=np.array([config.CLASS_CAR]),
            tracker_id=np.array([20])
        )
        tracker.trajectories[20].append(((car_box[0] + car_box[2])/2, (car_box[1] + car_box[3])/2))
        tracker.all_seen_vehicles.add(20)

        _, alerts, new_events = event_det.process_frame(dummy_frame, frame_no, tracked_detections=detections)

    wrong_way_events = [e for e in event_det.events if e["event"] == "wrong_way_driving"]
    assert len(wrong_way_events) > 0, "Wrong-way driving must be triggered when vehicle moves opposite to expected direction"
    ev = wrong_way_events[0]
    assert ev["vehicle_id"] == 20
    assert ev.get("movement_direction") == "LEFT"
    print(f"✓ Wrong-way driving successfully detected! Event: {ev}")


def test_stopped_vehicle_detection():
    print("\n--- Testing Video Vehicle Stopped / Possible Accident Logic ---")
    det = YOLODetector()
    tracker = MultiObjectTracker()
    event_det = EventDetector(det, tracker, "cam_01")
    event_det.events = []

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Vehicle ID 30 stationary at (300, 300) for 35 frames
    for frame_no in range(1, 36):
        car_box = np.array([300, 300, 420, 360], dtype=np.float32)
        detections = sv.Detections(
            xyxy=np.array([car_box]),
            confidence=np.array([0.95]),
            class_id=np.array([config.CLASS_CAR]),
            tracker_id=np.array([30])
        )
        tracker.trajectories[30].append((360.0, 330.0))
        tracker.all_seen_vehicles.add(30)

        _, alerts, new_events = event_det.process_frame(dummy_frame, frame_no, tracked_detections=detections)

    stopped_events = [e for e in event_det.events if e["event"] == "vehicle_stopped"]
    assert len(stopped_events) > 0, "Stopped vehicle event must be triggered when stationary >= STOPPED_FRAMES"
    ev = stopped_events[0]
    assert ev["vehicle_id"] == 30
    print(f"✓ Stopped vehicle / accident successfully detected! Event: {ev}")


if __name__ == "__main__":
    print("\n==========================================")
    print("  RUNNING CITYEYE LOGIC TEST SUITE")
    print("==========================================")
    test_helmet_detection_logic()
    test_single_image_triple_riding_logic()
    test_triple_riding_detection()
    test_wrong_way_detection()
    test_stopped_vehicle_detection()
    print("\n==========================================")
    print("  ALL EVENT DETECTION LOGIC VERIFIED! 🎉")
    print("==========================================\n")
