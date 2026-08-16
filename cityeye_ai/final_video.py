from pathlib import Path
import json
import cv2


BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "input.mp4"
EVENTS_PATH = BASE_DIR / "data" / "final_events.json"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_PATH = OUTPUT_DIR / "cityeye_final_output.mp4"


# How long event label stays visible
EVENT_DISPLAY_FRAMES = 75


def load_events():

    with open(
        EVENTS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data.get("events", {})


def add_event(
    frame_events,
    frame_number,
    event_name,
    duration=EVENT_DISPLAY_FRAMES
):

    start = max(1, frame_number)

    end = frame_number + duration

    for frame in range(start, end + 1):

        frame_events.setdefault(
            frame,
            []
        ).append(event_name)


def add_event_range(
    frame_events,
    start_frame,
    end_frame,
    event_name
):

    start = max(1, start_frame)

    end = end_frame + EVENT_DISPLAY_FRAMES

    for frame in range(start, end + 1):

        frame_events.setdefault(
            frame,
            []
        ).append(event_name)


def draw_text(
    frame,
    text,
    position,
    scale=0.65
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading final events...")

    events = load_events()

    frame_events = {}

    # =========================================
    # NO HELMET
    # =========================================

    for event in events.get(
        "no_helmet",
        []
    ):

        start_frame = event.get(
            "start_frame"
        )

        end_frame = event.get(
            "end_frame",
            start_frame
        )

        if start_frame is not None:

            add_event_range(
                frame_events,
                start_frame,
                end_frame,
                "NO HELMET"
            )

    # =========================================
    # TRIPLE RIDING
    # =========================================

    for event in events.get(
        "triple_riding",
        []
    ):

        frame = event.get(
            "frame"
        )

        if frame is not None:

            add_event(
                frame_events,
                frame,
                "TRIPLE RIDING"
            )

    # =========================================
    # WRONG WAY
    # =========================================

    for event in events.get(
        "wrong_way",
        []
    ):

        frame = event.get(
            "frame"
        )

        if frame is not None:

            add_event(
                frame_events,
                frame,
                "WRONG WAY"
            )

    # =========================================
    # ACCIDENT
    # =========================================

    for event in events.get(
        "accident",
        []
    ):

        frame = event.get(
            "frame"
        )

        if frame is not None:

            add_event(
                frame_events,
                frame,
                "POSSIBLE ACCIDENT"
            )

    print(
        f"Total event frames prepared: "
        f"{len(frame_events)}"
    )

    # =========================================
    # OPEN VIDEO
    # =========================================

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        print(
            "ERROR: Cannot open input video."
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height)
    )

    print("\n================================")
    print("CREATING CITYEYE FINAL VIDEO")
    print("================================")

    print(
        f"Resolution: {width}x{height}"
    )

    print(
        f"FPS: {fps}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # =====================================
        # TITLE
        # =====================================

        draw_text(
            frame,
            "CITYEYE - AI TRAFFIC MONITORING",
            (15, 30),
            0.65
        )

        # =====================================
        # CURRENT EVENTS
        # =====================================

        current_events = frame_events.get(
            frame_number,
            []
        )

        # Remove duplicates
        current_events = list(
            dict.fromkeys(current_events)
        )

        y = 65

        for event_name in current_events:

            draw_text(
                frame,
                f"EVENT: {event_name}",
                (15, y),
                0.65
            )

            y += 30

        # =====================================
        # FRAME NUMBER
        # =====================================

        draw_text(
            frame,
            f"Frame: {frame_number}",
            (15, height - 15),
            0.5
        )

        writer.write(frame)

    cap.release()
    writer.release()

    print("\n================================")
    print("FINAL VIDEO CREATED")
    print("================================")

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":

    main()