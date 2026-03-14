"""
AirSketch main entry point.

Handles the webcam loop and delegates all gesture detection to gesture_engine
and all canvas/UI rendering to canvas. Press 'q' to quit. Press 'c' to clear
the canvas.
"""

from collections import deque

import cv2
import mediapipe as mp

from canvas import (
    blend_frame,
    break_stroke,
    create_canvas,
    draw_clear_message,
    draw_palette,
    reset_canvas,
    update_stroke,
)
from gesture_engine import get_active_color, get_index_tip, is_drawing, is_erasing

PALETTE_COLORS = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]


def main() -> None:
    """Open the webcam, process hand landmarks each frame, and display the AirSketch UI.

    Reads frames from the default webcam, runs MediaPipe Hands on each frame,
    calls gesture_engine to interpret landmarks, calls canvas utilities to
    update the drawing layer and render UI elements, and shows the final
    blended output in the AirSketch window.
    """
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    draw_points = []
    fingertip_history = deque(maxlen=5)
    canvas = None
    active_color = (255, 255, 255)
    clear_message_frames = 0

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break

            frame_height, frame_width = frame.shape[:2]
            if canvas is None:
                canvas = create_canvas(frame_height, frame_width)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
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

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
