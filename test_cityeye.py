import os
import json
from fastapi.testclient import TestClient
import config
from app import app
from generate_sample import generate_cctv_sample

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["camera_id"] == config.CAMERA_ID
    print("✓ Health endpoint verified:", data)

def test_events_endpoint():
    response = client.get("/events")
    assert response.status_code == 200
    data = response.json()
    assert "total_events" in data
    assert "statistics" in data
    assert "events" in data
    print("✓ Events endpoint verified:", data["statistics"])

def test_missing_video_handling():
    # If video path doesn't exist, /process should return 404 with friendly message
    temp_path = config.VIDEO_PATH + ".bak"
    if os.path.exists(config.VIDEO_PATH):
        os.rename(config.VIDEO_PATH, temp_path)
    
    try:
        response = client.post("/process")
        assert response.status_code == 404
        data = response.json()
        assert "traffic.mp4 not found" in data["message"]
        print("✓ Missing video handling verified:", data["message"])
    finally:
        if os.path.exists(temp_path):
            os.rename(temp_path, config.VIDEO_PATH)

def test_video_pipeline_and_output():
    # Generate sample video if traffic.mp4 doesn't exist
    if not os.path.exists(config.VIDEO_PATH):
        print("Generating test traffic.mp4...")
        generate_cctv_sample(config.VIDEO_PATH, duration_sec=6)

    assert os.path.exists(config.VIDEO_PATH), "videos/traffic.mp4 must exist for pipeline test"
    
    # Trigger processing
    response = client.post("/process")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    print("✓ Processing pipeline successful:", data["message"])
    print(f"  Total frames: {data['total_frames_processed']}, Elapsed: {data['elapsed_seconds']}s")

    # Verify output video file created
    assert os.path.exists(config.OUTPUT_PATH), "output/processed_video.mp4 must be created"
    output_size = os.path.getsize(config.OUTPUT_PATH)
    assert output_size > 1000, f"output video size too small: {output_size} bytes"
    print(f"✓ Output video verified: {config.OUTPUT_PATH} ({output_size} bytes)")

    # Verify events.json created and valid
    assert os.path.exists(config.EVENTS_JSON_PATH), "data/events.json must exist"
    with open(config.EVENTS_JSON_PATH, "r", encoding="utf-8") as f:
        events_data = json.load(f)
    assert "events" in events_data
    print(f"✓ Events JSON verified: {len(events_data['events'])} events recorded in data/events.json")

def test_dashboard_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert "CITYEYE" in response.text
    assert "TRIPLE RIDING" in response.text
    print("✓ Dashboard UI template rendered successfully")

if __name__ == "__main__":
    print("\n==========================================")
    print("  RUNNING CITYEYE TEST SUITE")
    print("==========================================\n")
    test_health_endpoint()
    test_events_endpoint()
    test_missing_video_handling()
    test_dashboard_ui()
    test_video_pipeline_and_output()
    print("\n==========================================")
    print("  ALL TESTS PASSED SUCCESSFULLY! 🎉")
    print("==========================================\n")
