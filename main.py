"""
Enhanced main.py with improved state management and UI
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from pose_utils import analyze_pushup, analyze_squat, analyze_plank

# Configuration
EXERCISE = 'pushup'  # Options: 'pushup' 'squat', 'plank'
CONFIDENCE_THRESHOLD = 0.5

# Load YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# State management
class ExerciseTracker:
    def __init__(self):
        self.rep_count = 0
        self.state = None
        self.plank_start_time = None
        self.plank_duration = 0
        self.form_quality_history = []
        
    def update_reps(self, new_state):
        """Update rep count based on state transitions."""
        if new_state == 'down':
            self.state = 'down'
        elif new_state == 'up' and self.state == 'down':
            self.rep_count += 1
            self.state = 'up'
            print(f"Rep completed! Total: {self.rep_count}")
        # Don't change state if new_state is None (transitional)
            
    def update_plank(self, quality):
        """Track plank hold time."""
        if quality in ['excellent', 'good']:
            if self.plank_start_time is None:
                self.plank_start_time = time.time()
            self.plank_duration = time.time() - self.plank_start_time
        else:
            # Reset if not in proper plank position or poor form
            self.plank_start_time = None
            self.plank_duration = 0
            
    def update_form_quality(self, quality):
        """Track form quality over time."""
        self.form_quality_history.append(quality)
        if len(self.form_quality_history) > 30:  # Keep last 30 frames
            self.form_quality_history.pop(0)
            
    def get_form_score(self):
        """Calculate overall form score (0-100)."""
        if not self.form_quality_history:
            return 0
        good_count = sum(1 for q in self.form_quality_history if q == 'good')
        return int((good_count / len(self.form_quality_history)) * 100)

tracker = ExerciseTracker()
prev_time = time.time()

print(f"Starting {EXERCISE} analysis...")
print("Press 'r' to reset counter, 'q' or ESC to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    
    # Run YOLO inference
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

    feedback_text = 'No person detected'
    form_quality = 'unknown'

    if keypoints is not None and len(keypoints) > 0:
        keypoints = keypoints[0]

        # Analyze based on exercise type
        if EXERCISE == 'pushup':
            feedback_text, rep_event, form_quality = analyze_pushup(keypoints)
            tracker.update_reps(rep_event)
            
        elif EXERCISE == 'squat':
            feedback_text, rep_event, form_quality = analyze_squat(keypoints)
            tracker.update_reps(rep_event)
            
        elif EXERCISE == 'plank':
            feedback_text, hold_quality, form_quality = analyze_plank(
                keypoints, tracker.plank_duration
            )
            tracker.update_plank(hold_quality)

        tracker.update_form_quality(form_quality)
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame

    # Draw UI overlay
    overlay = annotated_frame.copy()
    
    # Header panel
    cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 100), (0, 0, 0), -1)
    
    # Exercise info
    cv2.putText(overlay, f'{EXERCISE.upper()}', (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    
    if EXERCISE != 'plank':
        cv2.putText(overlay, f'Reps: {tracker.rep_count}', (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Form score
    form_score = tracker.get_form_score()
    score_color = (0, 255, 0) if form_score > 70 else (0, 165, 255) if form_score > 40 else (0, 0, 255)
    cv2.putText(overlay, f'Form: {form_score}%', (250, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)
    
    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(overlay, f'FPS: {int(fps)}', (annotated_frame.shape[1] - 150, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Feedback box
    feedback_bg = (40, 40, 40) if form_quality == 'good' else (0, 0, 100)
    cv2.rectangle(overlay, (10, 120), (annotated_frame.shape[1] - 10, 170), 
                  feedback_bg, -1)
    cv2.putText(overlay, feedback_text, (20, 155), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)

    cv2.imshow('AI Exercise Form Analyzer', annotated_frame)

    # Handle keyboard input
    key = cv2.waitKey(5) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or 'q'
        break
    elif key == ord('r'):  # Reset counter
        tracker = ExerciseTracker()
        print("Counter reset!")

cap.release()
cv2.destroyAllWindows()
print(f"\nSession complete! Total reps: {tracker.rep_count}")