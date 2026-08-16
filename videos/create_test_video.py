import cv2
import numpy as np
import os

OUTPUT = "test_traffic.mp4"

WIDTH = 1280
HEIGHT = 720
FPS = 25

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT,
    fourcc,
    FPS,
    (WIDTH, HEIGHT)
)


def background(frame):
    frame[:] = (45, 45, 45)

    # Road
    cv2.rectangle(
        frame,
        (0, 250),
        (WIDTH, 650),
        (80, 80, 80),
        -1
    )

    # Road lines
    for x in range(0, WIDTH, 100):
        cv2.rectangle(
            frame,
            (x, 440),
            (x + 50, 450),
            (220, 220, 220),
            -1
        )


def title(frame, text):
    cv2.rectangle(
        frame,
        (20, 20),
        (650, 80),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        text,
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )


def car(frame, x, y, label="CAR"):
    cv2.rectangle(
        frame,
        (x, y),
        (x + 150, y + 70),
        (0, 150, 255),
        -1
    )

    cv2.rectangle(
        frame,
        (x + 30, y - 35),
        (x + 120, y),
        (100, 180, 220),
        -1
    )

    cv2.circle(
        frame,
        (x + 35, y + 70),
        15,
        (20, 20, 20),
        -1
    )

    cv2.circle(
        frame,
        (x + 115, y + 70),
        15,
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        label,
        (x, y - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


def motorcycle(frame, x, y, people=1, helmet=False):
    # wheels
    cv2.circle(
        frame,
        (x + 25, y + 65),
        18,
        (20, 20, 20),
        -1
    )

    cv2.circle(
        frame,
        (x + 115, y + 65),
        18,
        (20, 20, 20),
        -1
    )

    # motorcycle body
    cv2.line(
        frame,
        (x + 25, y + 65),
        (x + 65, y + 35),
        (0, 0, 255),
        8
    )

    cv2.line(
        frame,
        (x + 65, y + 35),
        (x + 115, y + 65),
        (0, 0, 255),
        8
    )

    # people
    for i in range(people):

        px = x + 35 + i * 30
        py = y - 30 - i * 5

        cv2.circle(
            frame,
            (px, py),
            15,
            (200, 150, 100),
            -1
        )

        cv2.rectangle(
            frame,
            (px - 12, py + 12),
            (px + 12, py + 55),
            (50, 100, 200),
            -1
        )

        if helmet:
            cv2.circle(
                frame,
                (px, py - 5),
                17,
                (30, 30, 30),
                4
            )


# =====================================================
# SCENE 1 - HELMET VIOLATION
# =====================================================

for frame_no in range(FPS * 5):

    frame = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8
    )

    background(frame)

    title(
        frame,
        "CITYEYE - HELMET VIOLATION"
    )

    x = 100 + frame_no * 3

    motorcycle(
        frame,
        x,
        430,
        people=1,
        helmet=False
    )

    cv2.putText(
        frame,
        "EVENT: HELMET VIOLATION",
        (700, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        3
    )

    writer.write(frame)


# =====================================================
# SCENE 2 - TRIPLE SEAT
# =====================================================

for frame_no in range(FPS * 5):

    frame = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8
    )

    background(frame)

    title(
        frame,
        "CITYEYE - TRIPLE SEAT"
    )

    x = 100 + frame_no * 3

    motorcycle(
        frame,
        x,
        430,
        people=3,
        helmet=False
    )

    cv2.putText(
        frame,
        "EVENT: TRIPLE SEAT VIOLATION",
        (650, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        3
    )

    writer.write(frame)


# =====================================================
# SCENE 3 - WRONG WAY
# =====================================================

for frame_no in range(FPS * 5):

    frame = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8
    )

    background(frame)

    title(
        frame,
        "CITYEYE - WRONG WAY"
    )

    # normal vehicle
    normal_x = 150 + frame_no * 4

    car(
        frame,
        normal_x,
        350,
        "NORMAL"
    )

    # wrong-way vehicle moving left
    wrong_x = 1000 - frame_no * 4

    car(
        frame,
        wrong_x,
        520,
        "WRONG WAY"
    )

    cv2.putText(
        frame,
        "EVENT: WRONG-WAY DRIVING",
        (700, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        3
    )

    writer.write(frame)


# =====================================================
# SCENE 4 - ACCIDENT / STOPPED VEHICLE
# =====================================================

for frame_no in range(FPS * 7):

    frame = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8
    )

    background(frame)

    title(
        frame,
        "CITYEYE - ACCIDENT / STOPPED VEHICLE"
    )

    # Vehicle moves initially
    if frame_no < FPS * 2:

        x = 100 + frame_no * 5

    else:

        # vehicle stops
        x = 100 + FPS * 2 * 5

    car(
        frame,
        x,
        420,
        "VEHICLE"
    )

    if frame_no > FPS * 4:

        cv2.putText(
            frame,
            "STOPPED",
            (x, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "EVENT: POSSIBLE ACCIDENT",
            (700, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3
        )

    writer.write(frame)


writer.release()

print()
print("==============================")
print("CITYEYE TEST VIDEO CREATED")
print("==============================")
print("File:", os.path.abspath(OUTPUT))
print("Duration: approximately 22 seconds")
print()