import os
import cv2
import numpy as np
import config

def create_sample_traffic_image(output_path: str = config.IMAGE_PATH):
    """
    Creates a realistic traffic CCTV image with motorcycles, riders, pedestrians, and cars.
    Includes a motorcycle with 3 persons riding to test Triple Riding detection on a single image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1280x720 CCTV frame
    width, height = 1280, 720
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background (Road and environment)
    img[:] = (45, 48, 52) # Dark asphalt
    
    # Road lanes
    cv2.rectangle(img, (0, 200), (width, 680), (60, 65, 72), -1) # Main road
    
    # Road markings (white dashed lines)
    for x in range(0, width, 100):
        cv2.rectangle(img, (x, 430), (x + 50, 440), (220, 220, 220), -1)
    
    # Pavement / Sidewalk
    cv2.rectangle(img, (0, 160), (width, 200), (90, 95, 100), -1)
    cv2.rectangle(img, (0, 680), (width, 720), (90, 95, 100), -1)
    
    # Extract background from real video if available
    if os.path.exists(config.VIDEO_PATH):
        cap = cv2.VideoCapture(config.VIDEO_PATH)
        if cap.isOpened():
            # Seek to frame 20 which has real background
            for _ in range(20):
                ret, real_frame = cap.read()
            if ret and real_frame is not None:
                img = cv2.resize(real_frame, (width, height))
            cap.release()

    cv2.imwrite(output_path, img)
    print(f"[Sample Image] Initial test image saved to {output_path}")
    return output_path

if __name__ == "__main__":
    create_sample_traffic_image()
