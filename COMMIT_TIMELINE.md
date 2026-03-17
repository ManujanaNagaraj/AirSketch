# Commit Timeline
1. feat: add webcam initialization with error handling
2. feat: add MediaPipe Hands setup with confidence thresholds
3. feat: add landmark drawing on live webcam frame
4. feat: add get_index_tip function in hand_utils.py
5. feat: add fingertip coordinate extraction per frame
6. feat: add is_drawing gesture detection logic
7. feat: wire is_drawing to main loop draw/pause toggle
8. feat: add is_erasing pinch gesture detection
9. feat: show red circle on screen when erase mode active
10. feat: add count_fingers utility function
11. feat: add get_active_color gesture color switching
12. feat: wire color switching into main loop
13. feat: create blank numpy canvas same size as frame
14. feat: draw strokes on canvas using cv2.line
15. feat: overlay canvas on webcam frame with addWeighted
16. feat: add stroke break on pause using None in draw_points
17. feat: add color palette bar at top of frame
18. feat: highlight active color swatch with border
19. feat: add clear canvas on keypress c
20. feat: show Canvas Cleared message for 60 frames
21. feat: add deque-based fingertip smoothing
22. feat: apply moving average to reduce stroke jitter
23. refactor: move gesture functions to gesture_engine.py
24. refactor: move canvas logic to canvas.py
25. refactor: simplify main.py to only handle webcam loop
26. fix: correct MediaPipe Hands attribute initialization
27. fix: add cv2.namedWindow before main loop
28. fix: handle no hand detected edge case gracefully
29. fix: prevent index error when draw_points is empty
30. fix: correct BGR color tuples for palette swatches
31. style: add docstrings to all functions in gesture_engine.py
