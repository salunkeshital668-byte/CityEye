import os
import cv2
import numpy as np
import config

def generate_cctv_sample(output_file: str = config.VIDEO_PATH, duration_sec: int = 12) -> str:
    """
    Creates videos/traffic.mp4.
    If a source traffic.mkv exists in the videos folder, it extracts a realistic clip.
    Otherwise, it synthesizes a multi-event CCTV traffic feed.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    mkv_source = os.path.join(os.path.dirname(output_file), "traffic.mkv")

    # If traffic.mkv exists, extract a test segment
    if os.path.exists(mkv_source):
        print(f"[Sample Generator] Found source '{mkv_source}'. Extracting clip to '{output_file}'...")
        cap = cv2.VideoCapture(mkv_source)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            max_frames = int(fps * duration_sec)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

            # Skip first 400 frames to get into active traffic
            for _ in range(400):
                if not cap.grab():
                    break

            count = 0
            while count < max_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                writer.write(frame)
                count += 1

            cap.release()
            writer.release()
            print(f"[Sample Generator] Successfully exported {count} frames from traffic.mkv to {output_file}")
            return output_file

    # Synthetic fallback simulation
    width, height = 1280, 720
    fps = 25
    total_frames = fps * duration_sec

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    def draw_road(frame):
        frame[:] = (35, 38, 45)
        cv2.rectangle(frame, (0, 260), (width, 640), (60, 65, 75), -1)
        for x in range(0, width, 90):
            cv2.rectangle(frame, (x, 445), (x + 45, 455), (230, 230, 230), -1)

    def draw_car(frame, x, y, color=(0, 165, 255), label="CAR"):
        cv2.rectangle(frame, (int(x), int(y)), (int(x + 140), int(y + 60)), color, -1)
        cv2.rectangle(frame, (int(x + 25), int(y - 30)), (int(x + 115), int(y)), (160, 200, 230), -1)
        cv2.circle(frame, (int(x + 30), int(y + 60)), 14, (20, 20, 20), -1)
        cv2.circle(frame, (int(x + 110), int(y + 60)), 14, (20, 20, 20), -1)

    def draw_motorcycle_with_riders(frame, x, y, rider_count=3):
        cv2.circle(frame, (int(x + 20), int(y + 55)), 16, (20, 20, 20), -1)
        cv2.circle(frame, (int(x + 100), int(y + 55)), 16, (20, 20, 20), -1)
        cv2.line(frame, (int(x + 20), int(y + 55)), (int(x + 60), int(y + 30)), (0, 0, 220), 7)
        cv2.line(frame, (int(x + 60), int(y + 30)), (int(x + 100), int(y + 55)), (0, 0, 220), 7)

        for i in range(rider_count):
            rx = int(x + 30 + (i * 26))
            ry = int(y - 25 - (i * 4))
            cv2.circle(frame, (rx, ry), 13, (210, 170, 130), -1)
            cv2.rectangle(frame, (rx - 10, ry + 12), (rx + 10, ry + 50), (40, 90, 200), -1)

    print(f"[Sample Generator] Generating synthetic CCTV traffic MP4 to {output_file}...")
    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_road(frame)
        cv2.putText(frame, "CCTV CAM_01 - LIVE TRAFFIC FEED", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        bike_x = (f * 4) % (width + 200) - 100
        draw_motorcycle_with_riders(frame, bike_x, 480, rider_count=3)

        car1_x = (50 + f * 5) % (width + 200) - 100
        draw_car(frame, car1_x, 340, color=(0, 180, 255), label="CAR")

        wrong_x = width - ((f * 5) % (width + 200))
        draw_car(frame, wrong_x, 530, color=(255, 80, 80), label="WRONG WAY CAR")

        if f < fps * 2:
            stopped_x = 200 + (f * 4)
        else:
            stopped_x = 200 + (fps * 2 * 4)
        draw_car(frame, stopped_x, 380, color=(200, 140, 0), label="STOPPED VEHICLE")

        writer.write(frame)

    writer.release()
    print(f"[Sample Generator] Done! Generated {total_frames} frames ({duration_sec}s). Saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    generate_cctv_sample()
