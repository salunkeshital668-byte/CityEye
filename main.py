"""
CityEye — AI Traffic & CCTV Event Detection System
==================================================
Python-only CLI runner supporting BOTH Image and MP4 Video inputs.

- Image Mode:
    * Loads 'images/traffic.jpg'
    * YOLO object detection (Person & Motorcycle)
    * Single-image triple-riding detection (no tracking needed)
    * Draws bounding boxes, labels, and confidence scores
    * Saves annotated image to 'output/detected_image.jpg'
    * Handles missing image gracefully with clear message

- Video Mode:
    * Loads 'videos/traffic.mp4'
    * ByteTrack multi-object tracking
    * Wrong-way driving & vehicle-stopped/accident detection (video only)
    * Triple-riding detection (both image & video)
    * Saves annotated video to 'output/processed_video.mp4'
    * Saves event log to 'data/events.json'
    * Handles missing video gracefully with clear message
"""

import os
import sys
import argparse
from pathlib import Path

import config
from detector import YOLODetector
from tracker import MultiObjectTracker
from event_detector import EventDetector


def run_cityeye(
    mode: str = "both",
    image_path: str = config.IMAGE_PATH,
    video_path: str = config.VIDEO_PATH,
    image_output_path: str = config.IMAGE_OUTPUT_PATH,
    video_output_path: str = config.OUTPUT_PATH
):
    print("=" * 60)
    print("        CITYEYE AI TRAFFIC & CCTV ANALYTICS        ")
    print("=" * 60)

    # Ensure output and data directories exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.IMAGES_DIR, exist_ok=True)
    os.makedirs(config.VIDEOS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Initialize YOLO detector and tracker
    detector = YOLODetector()
    tracker = MultiObjectTracker()
    event_detector = EventDetector(detector=detector, tracker=tracker, camera_id=config.CAMERA_ID)

    # Report Helmet Model Status
    print("-" * 60)
    if detector.helmet_model_available:
        print(f"• Helmet AI Status:   [ACTIVE / LOADED]")
        print(f"  Model Path:         {config.HELMET_MODEL_PATH}")
        print(f"  Classes Detected:   {detector.helmet_class_names}")
    else:
        print(f"• Helmet AI Status:   [NOT CONFIGURED / NOT FOUND]")
        print(f"  Expected Path:      {config.HELMET_MODEL_PATH}")
        print(f"  Note: To enable real helmet detection, place your model at: models/helmet_model.pt")
    print("-" * 60)

    image_results = None
    video_results = None

    # =========================================================================
    # 1. IMAGE DETECTION (Tested first as specified)
    # =========================================================================
    if mode in ("both", "image"):
        print("\n" + "-" * 60)
        print(" [1/2] RUNNING IMAGE DETECTION")
        print("-" * 60)

        if not os.path.exists(image_path):
            print(f"[CityEye Image] Notice: Image file '{image_path}' was not found.")
            print(f"                Place an image at '{image_path}' to run image detection.")
            image_results = {
                "status": "skipped",
                "message": f"Image file '{image_path}' not found."
            }
        else:
            image_results = event_detector.process_image(
                image_path=image_path,
                output_path=image_output_path
            )

    # =========================================================================
    # 2. VIDEO DETECTION (Tested second if videos/traffic.mp4 exists)
    # =========================================================================
    if mode in ("both", "video"):
        print("\n" + "-" * 60)
        print(" [2/2] RUNNING VIDEO DETECTION")
        print("-" * 60)

        if not os.path.exists(video_path):
            print(f"[CityEye Video] Notice: Video file '{video_path}' was not found.")
            print(f"                Place an MP4 video at '{video_path}' to run video detection.")
            video_results = {
                "status": "skipped",
                "message": f"Video file '{video_path}' not found."
            }
        else:
            video_results = event_detector.process_video(
                video_path=video_path,
                output_path=video_output_path
            )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("                   PROCESSING SUMMARY                      ")
    print("=" * 60)

    if image_results:
        print(f"• Image Analysis: [{image_results.get('status', 'unknown').upper()}]")
        if image_results.get("status") == "success":
            print(f"    - Persons Detected:     {image_results.get('persons_detected', 0)}")
            print(f"    - Motorcycles Detected: {image_results.get('motorcycles_detected', 0)}")
            print(f"    - Triple Riding Events: {image_results.get('triple_riding_count', 0)}")
            print(f"    - Annotated Image:      {image_results.get('output_path')}")
        else:
            print(f"    - Reason: {image_results.get('message')}")

    if video_results:
        print(f"• Video Analysis: [{video_results.get('status', 'unknown').upper()}]")
        if video_results.get("status") == "success":
            print(f"    - Frames Processed:     {video_results.get('total_frames_processed', 0)}")
            print(f"    - Elapsed Time:         {video_results.get('elapsed_seconds', 0)}s")
            print(f"    - Total Events Logged:  {video_results.get('events_count', 0)}")
            print(f"    - Annotated Video:      {video_results.get('output_path')}")
        else:
            print(f"    - Reason: {video_results.get('message')}")

    print("=" * 60 + "\n")
    return image_results, video_results


def main():
    parser = argparse.ArgumentParser(
        description="CityEye — AI Traffic CCTV Event Detection System (Image & MP4 Video)"
    )
    parser.add_argument(
        "--mode",
        choices=["both", "image", "video"],
        default=None,
        help="Processing mode: 'both' (default), 'image', or 'video'"
    )
    parser.add_argument(
        "--image",
        default=None,
        help=f"Path to input image (default: {config.IMAGE_PATH})"
    )
    parser.add_argument(
        "--video",
        default=None,
        help=f"Path to input MP4 video (default: {config.VIDEO_PATH})"
    )
    parser.add_argument(
        "--output-image",
        default=config.IMAGE_OUTPUT_PATH,
        help=f"Path for output detected image (default: {config.IMAGE_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--output-video",
        default=config.OUTPUT_PATH,
        help=f"Path for output processed video (default: {config.OUTPUT_PATH})"
    )

    args = parser.parse_args()

    # Determine mode based on explicit arguments
    if args.mode:
        mode = args.mode
    elif args.image and not args.video:
        mode = "image"
    elif args.video and not args.image:
        mode = "video"
    else:
        mode = "both"

    image_path = args.image if args.image else config.IMAGE_PATH
    video_path = args.video if args.video else config.VIDEO_PATH

    run_cityeye(
        mode=mode,
        image_path=image_path,
        video_path=video_path,
        image_output_path=args.output_image,
        video_output_path=args.output_video
    )


if __name__ == "__main__":
    main()