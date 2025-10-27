"""main.py

Run: python main.py

Opens webcam, runs YOLOv8 Pose, computes angles, applies simple rules for push-ups/squats/plank,
shows overlay and simple rep counter + textual feedback.
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from pose_utils import calculate_angle, analyze_pushup, analyze_squat, analyze_plank

# Select exercise: 'pushup', 'squat', or 'plank'
EXERCISE = 'pushup'

# Load YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0)
prev_time = 0

rep_count = 0
state = None  # used for simple rep counting

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        break

    frame = cv2.flip(frame, 1)
    results = model(frame, verbose=False)
    keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

    feedback_text = ''

    if keypoints is not None and len(keypoints) > 0:
        keypoints = keypoints[0]

        if EXERCISE == 'pushup':
            feedback_text, rep_event = analyze_pushup(keypoints)
        elif EXERCISE == 'squat':
            feedback_text, rep_event = analyze_squat(keypoints)
        elif EXERCISE == 'plank':
            feedback_text, rep_event = analyze_plank(keypoints)
        else:
            feedback_text, rep_event = ('Unknown exercise', None)

        if rep_event == 'down':
            state = 'down'
        elif rep_event == 'up' and state == 'down':
            rep_count += 1
            state = 'up'

        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame
        feedback_text = 'No person detected'

    cv2.rectangle(annotated_frame, (0, 0), (300, 80), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f'Exercise: {EXERCISE}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(annotated_frame, f'Reps: {rep_count}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(annotated_frame, feedback_text, (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (460, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow('AI Exercise Form Analyzer (YOLOv8)', annotated_frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()