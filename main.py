"""
AirSketch main entry point.

Handles the webcam loop and delegates all gesture detection to gesture_engine
and all canvas/UI rendering to canvas. Press 'q' to quit. Press 'c' to clear
the canvas.
"""

import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from canvas import (
    blend_frame,
    break_stroke,
    create_canvas,
    draw_clear_message,
    draw_palette,
    reset_canvas,
    update_stroke,
)
from gesture_engine import get_active_color, get_index_tip, is_erasing
from hand_utils import is_drawing

PALETTE_COLORS = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]

_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Hand skeleton connections (pairs of landmark indices)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class _LandmarksWrapper:
    """Wraps a Tasks-API landmark list to expose the .landmark[n] interface
    that gesture_engine expects."""

    def __init__(self, landmark_list):
        self.landmark = landmark_list


def _draw_hand_landmarks(frame, landmarks_wrapper):
    """Draw the hand skeleton and joint dots onto *frame* using OpenCV."""
    h, w = frame.shape[:2]
    lm = landmarks_wrapper.landmark
    for a, b in _HAND_CONNECTIONS:
        x1, y1 = int(lm[a].x * w), int(lm[a].y * h)
        x2, y2 = int(lm[b].x * w), int(lm[b].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for pt in lm:
        cx, cy = int(pt.x * w), int(pt.y * h)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)


def _ensure_model():
    """Download hand_landmarker.task from Google if not already cached."""
    if not _MODEL_PATH.exists():
        print("Downloading hand_landmarker.task model (first run only)…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")


def main() -> None:
    """Open the webcam, process hand landmarks each frame, and display the AirSketch UI.

    Reads frames from the default webcam, runs MediaPipe HandLandmarker on each
    frame, calls gesture_engine to interpret landmarks, calls canvas utilities to
    update the drawing layer and render UI elements, and shows the final
    blended output in the AirSketch window.
    """
    _ensure_model()

    cap = cv2.VideoCapture(0)

    # Initialise MediaPipe Hands via the Tasks API (mp.solutions was removed in 0.10)
    mp_hands = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
    )
    mp_draw = mp_vision.drawing_utils  # noqa: F841  (kept for API parity)

    draw_points = []
    fingertip_history = deque(maxlen=5)
    canvas = None
    active_color = (255, 255, 255)
    clear_message_frames = 0
    timestamp_ms = 0

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        mp_hands.close()
        return

    cv2.namedWindow("AirSketch", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break

            frame_height, frame_width = frame.shape[:2]
            if canvas is None:
                canvas = create_canvas(frame_height, frame_width)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = mp_hands.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 33  # ~30 fps

            if results.hand_landmarks:
                for lm_list in results.hand_landmarks:
                    hand_landmarks = _LandmarksWrapper(lm_list)
                    active_color = get_active_color(hand_landmarks)
                    index_tip = get_index_tip(hand_landmarks, frame_width, frame_height)
                    print(f"Index fingertip: {index_tip}")

                    if is_drawing(hand_landmarks):
                        print("DRAWING")
                        update_stroke(draw_points, fingertip_history, index_tip, canvas, active_color)
                    else:
                        print("PAUSED")
                        break_stroke(draw_points, fingertip_history)

                    if is_erasing(hand_landmarks, frame_width, frame_height):
                        cv2.circle(frame, index_tip, 12, (0, 0, 255), 2)

                    _draw_hand_landmarks(frame, hand_landmarks)

            blended = blend_frame(frame, canvas)
            draw_palette(blended, PALETTE_COLORS, active_color)
            clear_message_frames = draw_clear_message(blended, frame_width, frame_height, clear_message_frames)

            cv2.imshow("AirSketch", blended)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                canvas = reset_canvas(draw_points, fingertip_history, frame_height, frame_width)
                clear_message_frames = 60
            if key == ord("q"):
                break
    except Exception as e:
        print(f"Error during main loop: {e}")
    finally:
        mp_hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
