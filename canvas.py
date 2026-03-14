"""
Canvas drawing utilities for AirSketch.

Provides functions for creating and managing the drawing canvas, rendering
strokes with smoothing, drawing the color palette bar, blending the canvas
onto the webcam frame, and displaying the 'Canvas Cleared' message.
"""

import cv2
import numpy as np


def create_canvas(height, width):
    """Create a blank black drawing canvas.

    Args:
        height (int): Height of the canvas in pixels.
        width (int): Width of the canvas in pixels.

    Returns:
        numpy.ndarray: A zero-filled BGR image of shape (height, width, 3).
    """
    return np.zeros((height, width, 3), dtype=np.uint8)


def update_stroke(draw_points, fingertip_history, index_tip, canvas, active_color):
    """Smooth the fingertip position and draw a line segment onto the canvas.

    Appends the raw fingertip to the history deque, computes the rolling
    average position for jitter reduction, stores the smoothed point in
    draw_points, and renders a line between the last two valid points.

    Args:
        draw_points (list): Accumulated stroke points, with None as stroke breaks.
        fingertip_history (collections.deque): Rolling window of raw fingertip positions.
        index_tip (tuple[int, int]): Current raw (x, y) fingertip coordinate.
        canvas (numpy.ndarray): Drawing canvas to render the line onto.
        active_color (tuple[int, int, int]): BGR color for the stroke.
    """
    fingertip_history.append(index_tip)
    smoothed_x = int(sum(pt[0] for pt in fingertip_history) / len(fingertip_history))
    smoothed_y = int(sum(pt[1] for pt in fingertip_history) / len(fingertip_history))
    smoothed_point = (smoothed_x, smoothed_y)

    draw_points.append(smoothed_point)
    if len(draw_points) >= 2:
        previous_point = draw_points[-2]
        current_point = draw_points[-1]
        if previous_point is not None and current_point is not None:
            cv2.line(canvas, previous_point, current_point, active_color, 4)


def break_stroke(draw_points, fingertip_history):
    """End the current stroke by appending a None sentinel and clearing the smoothing deque.

    The None value in draw_points prevents the next stroke from connecting to
    the previous one. Clearing the deque prevents stale positions from
    influencing the smoothing of the next stroke.

    Args:
        draw_points (list): Accumulated stroke points list to break.
        fingertip_history (collections.deque): Rolling window deque to clear.
    """
    draw_points.append(None)
    fingertip_history.clear()


def blend_frame(frame, canvas):
    """Blend the drawing canvas onto the webcam frame using weighted addition.

    The canvas is composited at 60% opacity over the full-brightness webcam frame.

    Args:
        frame (numpy.ndarray): Live webcam frame (BGR).
        canvas (numpy.ndarray): Drawing canvas (BGR).

    Returns:
        numpy.ndarray: Blended output frame.
    """
    return cv2.addWeighted(frame, 1.0, canvas, 0.6, 0)


def draw_palette(frame, palette_colors, active_color):
    """Draw color swatches along the top of the frame.

    Renders four 50x30 filled rectangles starting at y=10. A white border is
    drawn around the swatch that matches the current active_color.

    Args:
        frame (numpy.ndarray): Frame to draw the palette onto (modified in-place).
        palette_colors (list[tuple]): Ordered list of BGR color tuples for each swatch.
        active_color (tuple[int, int, int]): Currently selected BGR brush color.
    """
    palette_y = 10
    swatch_width = 50
    swatch_height = 30
    palette_x_start = 10

    for index, swatch_color in enumerate(palette_colors):
        x1 = palette_x_start + (index * swatch_width)
        y1 = palette_y
        x2 = x1 + swatch_width
        y2 = y1 + swatch_height

        cv2.rectangle(frame, (x1, y1), (x2, y2), swatch_color, -1)
        if swatch_color == active_color:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)


def draw_clear_message(frame, frame_width, frame_height, clear_message_frames):
    """Render a centered 'Canvas Cleared' message for a fixed number of frames.

    Decrements the counter each call. Does nothing once the counter reaches zero.

    Args:
        frame (numpy.ndarray): Frame to draw the text onto (modified in-place).
        frame_width (int): Width of the frame in pixels, used for centering.
        frame_height (int): Height of the frame in pixels, used for centering.
        clear_message_frames (int): Remaining frames to display the message.

    Returns:
        int: Decremented frame counter (clamped to 0).
    """
    if clear_message_frames > 0:
        message = "Canvas Cleared"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size, _ = cv2.getTextSize(message, font, font_scale, thickness)
        text_x = (frame_width - text_size[0]) // 2
        text_y = (frame_height + text_size[1]) // 2
        cv2.putText(
            frame,
            message,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        clear_message_frames -= 1
    return clear_message_frames


def reset_canvas(draw_points, fingertip_history, height, width):
    """Clear all stroke state and return a fresh blank canvas.

    Empties draw_points and fingertip_history in-place, then creates and
    returns a new zero-filled canvas of the given dimensions.

    Args:
        draw_points (list): Stroke points list to clear.
        fingertip_history (collections.deque): Smoothing deque to clear.
        height (int): Height for the new canvas.
        width (int): Width for the new canvas.

    Returns:
        numpy.ndarray: A fresh blank black canvas.
    """
    draw_points.clear()
    fingertip_history.clear()
    return create_canvas(height, width)
