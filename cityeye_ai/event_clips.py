from pathlib import Path
import json
import cv2


BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "input.mp4"
EVENTS_PATH = BASE_DIR / "data" / "final_events.json"

OUTPUT_DIR = BASE_DIR / "output" / "event_clips"

BEFORE_SECONDS = 2
AFTER_SECONDS = 3


def load_events():

    with open(
        EVENTS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data.get("events", {})


def get_event_frame(event):

    if "frame" in event:
        return event["frame"]

    if "start_frame" in event:
        return event["start_frame"]

    return None


def safe_name(name):

    return (
        name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    events = load_events()

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        print("ERROR: Cannot open input video.")

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print("\n================================")
    print("CITYEYE EVENT CLIP GENERATOR")
    print("================================")

    print(
        f"FPS: {fps}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"Output folder: {OUTPUT_DIR}"
    )

    event_counter = 0

    for event_type, event_list in events.items():

        if not isinstance(
            event_list,
            list
        ):
            continue

        for index, event in enumerate(
            event_list,
            start=1
        ):

            frame = get_event_frame(
                event
            )

            if frame is None:
                continue

            event_counter += 1

            start_frame = max(
                1,
                int(
                    frame -
                    BEFORE_SECONDS * fps
                )
            )

            end_frame = min(
                total_frames,
                int(
                    frame +
                    AFTER_SECONDS * fps
                )
            )

            event_name = safe_name(
                event_type
            )

            filename = (
                f"{event_name}_"
                f"{index:02d}_"
                f"frame_{frame}.mp4"
            )

            output_path = (
                OUTPUT_DIR /
                filename
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                start_frame
            )

            current_frame = start_frame

            while current_frame <= end_frame:

                ret, video_frame = cap.read()

                if not ret:
                    break

                # Add event information
                cv2.putText(
                    video_frame,
                    "CITYEYE EVENT",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    video_frame,
                    f"EVENT: {event_type}",
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    video_frame,
                    f"Frame: {frame}",
                    (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                writer.write(
                    video_frame
                )

                current_frame += 1

            writer.release()

            print(
                f"Created: {filename}"
            )

    cap.release()

    print("\n================================")
    print("EVENT CLIP GENERATION COMPLETE")
    print("================================")

    print(
        f"Total clips: {event_counter}"
    )

    print(
        f"Saved in: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()