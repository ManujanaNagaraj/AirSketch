"""
Gesture detection functions for AirSketch.

Provides utilities to interpret MediaPipe hand landmarks as gestures,
including drawing state, erase mode, active brush color, and fingertip
pixel coordinates.
"""

import math


def count_fingers(landmarks):
    """Count the number of extended fingers using tip-vs-PIP y-value comparisons.

    Checks index, middle, ring, and pinky fingers. A finger is considered
    extended when its tip y-value is above (less than) its PIP joint y-value.

    Args:
        landmarks: MediaPipe hand landmarks object.

    Returns:
        int: Number of extended fingers (0–4).
    """
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return sum(
        1
        for tip_index, pip_index in finger_pairs
        if landmarks.landmark[tip_index].y < landmarks.landmark[pip_index].y
    )


def get_index_tip(landmarks, frame_width, frame_height):
    """Return the pixel coordinates of the index fingertip (landmark 8).

    Converts the normalised MediaPipe coordinate to pixel space using the
    supplied frame dimensions.

    Args:
        landmarks: MediaPipe hand landmarks object.
        frame_width (int): Width of the video frame in pixels.
        frame_height (int): Height of the video frame in pixels.

    Returns:
        tuple[int, int]: (x, y) pixel position of the index fingertip.
    """
    index_tip = landmarks.landmark[8]
    x = int(index_tip.x * frame_width)
    y = int(index_tip.y * frame_height)
    return (x, y)


def is_drawing(landmarks):
    """Return True when only the index finger is extended and all others are curled.

    Uses y-value comparisons between fingertip and PIP joint landmarks to
    determine finger state.

    Args:
        landmarks: MediaPipe hand landmarks object.

    Returns:
        bool: True if the drawing gesture is active.
    """
    index_extended = landmarks.landmark[8].y < landmarks.landmark[6].y
    middle_curled = landmarks.landmark[12].y > landmarks.landmark[10].y
    ring_curled = landmarks.landmark[16].y > landmarks.landmark[14].y
    pinky_curled = landmarks.landmark[20].y > landmarks.landmark[18].y
    return index_extended and middle_curled and ring_curled and pinky_curled


def is_erasing(landmarks, frame_width, frame_height):
    """Return True when the thumb tip and index fingertip are within 40 pixels.

    Computes the Euclidean distance between landmark 4 (thumb tip) and
    landmark 8 (index fingertip) in pixel space.

    Args:
        landmarks: MediaPipe hand landmarks object.
        frame_width (int): Width of the video frame in pixels.
        frame_height (int): Height of the video frame in pixels.

    Returns:
        bool: True if the pinch/erase gesture is active.
    """
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]

    thumb_x = int(thumb_tip.x * frame_width)
    thumb_y = int(thumb_tip.y * frame_height)
    index_x = int(index_tip.x * frame_width)
    index_y = int(index_tip.y * frame_height)

    distance = math.dist((thumb_x, thumb_y), (index_x, index_y))
    return distance < 40


def get_active_color(landmarks):
    """Return a BGR color tuple based on the number of extended fingers.

    Mapping:
        1 finger  → white  (255, 255, 255)
        2 fingers → red    (0, 0, 255)
        3 fingers → green  (0, 255, 0)
        4 fingers → blue   (255, 0, 0)
    Falls back to white for any other count.

    Args:
        landmarks: MediaPipe hand landmarks object.

    Returns:
        tuple[int, int, int]: BGR color tuple for the active brush.
    """
    color_map = {
        1: (255, 255, 255),
        2: (0, 0, 255),
        3: (0, 255, 0),
        4: (255, 0, 0),
    }
    return color_map.get(count_fingers(landmarks), (255, 255, 255))
