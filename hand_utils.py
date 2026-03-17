"""Hand utility compatibility module.

Most gesture helpers are provided by gesture_engine. The drawing gesture helper
is defined here to keep imports stable for callers that expect hand_utils.
"""

from gesture_engine import count_fingers, get_active_color, get_index_tip, is_erasing


def is_drawing(landmarks):
	"""Return True when index is extended and middle/ring/pinky are curled.

	Finger state is detected using tip-vs-PIP y-value comparisons:
	- extended: tip.y < pip.y
	- curled:   tip.y > pip.y
	"""
	index_extended = landmarks.landmark[8].y < landmarks.landmark[6].y
	middle_curled = landmarks.landmark[12].y > landmarks.landmark[10].y
	ring_curled = landmarks.landmark[16].y > landmarks.landmark[14].y
	pinky_curled = landmarks.landmark[20].y > landmarks.landmark[18].y
	return index_extended and middle_curled and ring_curled and pinky_curled


__all__ = ["count_fingers", "get_active_color", "get_index_tip", "is_drawing", "is_erasing"]
