"""
Enhanced pose_utils.py with improved analysis and bilateral checking
"""

import numpy as np
from exercises.pushup_rules import PUSHUP
from exercises.squat_rules import SQUAT
from exercises.plank_rules import PLANK


def calculate_angle(a, b, c):
    """Calculate angle between three points (a-b-c) where b is the vertex."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    
    # Handle zero vectors
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0:
        return 0
    
    cosine_angle = np.dot(ab, cb) / (norm_ab * norm_cb)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def get_bilateral_angle(keypoints, left_indices, right_indices):
    """Calculate average angle from both sides of the body."""
    left_angle = calculate_angle(*[keypoints[i] for i in left_indices])
    right_angle = calculate_angle(*[keypoints[i] for i in right_indices])
    return (left_angle + right_angle) / 2, left_angle, right_angle


def analyze_pushup(keypoints):
    """
    Improved pushup analysis with bilateral checking and form feedback.
    Returns: (feedback_text, rep_event, form_quality)
    """
    # Bilateral elbow angle
    left_indices = [5, 7, 9]   # shoulder, elbow, wrist
    right_indices = [6, 8, 10]
    elbow_angle, left_angle, right_angle = get_bilateral_angle(
        keypoints, left_indices, right_indices
    )
    
    # Check back alignment (shoulder-hip-knee)
    back_angle_left = calculate_angle(keypoints[5], keypoints[11], keypoints[13])
    back_angle_right = calculate_angle(keypoints[6], keypoints[12], keypoints[14])
    back_angle = (back_angle_left + back_angle_right) / 2
    
    # Form quality checks
    form_issues = []
    asymmetry = abs(left_angle - right_angle)
    if asymmetry > 15:
        form_issues.append(f"Uneven arms ({asymmetry:.0f}° diff)")
    
    if back_angle < PUSHUP['back_min']:
        form_issues.append("Keep back straight")
    
    # Rep state detection - STRICT thresholds
    state = None  # Default to None
    
    if elbow_angle < PUSHUP['elbow_down_max']:
        # Only register 'down' if form is reasonable
        if back_angle >= PUSHUP['back_min'] - 20:  # Allow some tolerance
            state = 'down'
            feedback = 'Down position ✓'
        else:
            feedback = f'Too low, fix form (Elbow: {int(elbow_angle)}°)'
    elif elbow_angle > PUSHUP['elbow_up_min']:
        state = 'up'
        feedback = 'Up position ✓'
    else:
        feedback = f'Transitioning (Elbow: {int(elbow_angle)}°)'
    
    # Add form issues to feedback
    if form_issues:
        feedback += ' | ' + ' | '.join(form_issues)
    
    form_quality = 'good' if not form_issues else 'needs_improvement'
    
    return feedback, state, form_quality


def analyze_squat(keypoints):
    """
    Improved squat analysis with proper rep counting states.
    Returns: (feedback_text, rep_event, form_quality)
    """
    # Bilateral knee angle
    left_indices = [11, 13, 15]  # hip, knee, ankle
    right_indices = [12, 14, 16]
    knee_angle, left_angle, right_angle = get_bilateral_angle(
        keypoints, left_indices, right_indices
    )
    
    # Back angle (shoulder-hip-knee for torso lean)
    back_angle_left = calculate_angle(keypoints[5], keypoints[11], keypoints[13])
    back_angle_right = calculate_angle(keypoints[6], keypoints[12], keypoints[14])
    back_angle = (back_angle_left + back_angle_right) / 2
    
    form_issues = []
    asymmetry = abs(left_angle - right_angle)
    if asymmetry > 15:
        form_issues.append(f"Uneven legs ({asymmetry:.0f}°)")
    
    if back_angle < SQUAT['back_min']:
        form_issues.append("Torso too forward")
    
    # Improved rep state detection - STRICT
    state = None  # Default to None
    
    if knee_angle <= SQUAT['knee_min']:
        # Too deep
        state = None
        feedback = f'Too deep! (Knee: {int(knee_angle)}°)'
    elif knee_angle <= SQUAT['knee_max']:
        # Good depth for squat
        state = 'down'
        feedback = 'Good depth ✓'
    elif knee_angle >= 165:
        # Fully standing
        state = 'up'
        feedback = 'Standing ✓'
    else:
        # Between good depth and standing
        feedback = f'Not deep enough (Knee: {int(knee_angle)}°)'
    
    if form_issues:
        feedback += ' | ' + ' | '.join(form_issues)
    
    form_quality = 'good' if not form_issues else 'needs_improvement'
    
    return feedback, state, form_quality


def is_plank_position(keypoints):
    """
    Check if person is in a plank position (horizontal body, hands/elbows on ground).
    Returns: (is_plank, reason)
    """
    # Get key points
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    # Calculate average positions
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_y = (left_hip[1] + right_hip[1]) / 2
    knee_y = (left_knee[1] + right_knee[1]) / 2
    ankle_y = (left_ankle[1] + right_ankle[1]) / 2
    wrist_y = (left_wrist[1] + right_wrist[1]) / 2
    nose_y = nose[1]
    
    # Check 1: Body should be relatively horizontal (shoulders and hips at similar height)
    vertical_diff = abs(shoulder_y - hip_y)
    if vertical_diff > 150:  # Too much vertical difference
        return False, "Not horizontal - get into plank position"
    
    # Check 2: Hands should be on ground (wrists should be lower or at similar level as shoulders)
    if wrist_y < shoulder_y - 100:  # Hands way above shoulders
        return False, "Hands not on ground"
    
    # Check 3: Body should be elevated (not lying flat - nose should be higher than in lying position)
    # In plank, nose should be below shoulders but not too far
    if nose_y > shoulder_y + 100:  # Head too low (lying flat)
        return False, "Body too low - lift up"
    
    # Check 4: Legs should be extended (knees and ankles should be at similar level to hips)
    if knee_y < hip_y - 100:  # Knees bent too much
        return False, "Extend your legs"
    
    # Check 5: Not standing (ankles should be at similar or lower level than hips)
    if ankle_y < shoulder_y - 50:  # Ankles way above shoulders (standing)
        return False, "You're standing - get into plank position"
    
    return True, "In plank position"


def analyze_plank(keypoints, duration=0):
    """
    Improved plank analysis with duration tracking and position validation.
    Returns: (feedback_text, hold_quality, form_quality)
    """
    # First check if in plank position
    is_plank, position_msg = is_plank_position(keypoints)
    
    if not is_plank:
        return (position_msg, 'not_plank', 'needs_improvement')
    
    # Body alignment (shoulder-hip-ankle)
    left_indices = [5, 11, 15]
    right_indices = [6, 12, 16]
    body_angle, left_angle, right_angle = get_bilateral_angle(
        keypoints, left_indices, right_indices
    )
    
    form_issues = []
    
    if body_angle > PLANK['back_good']:
        quality = 'excellent'
        feedback = f'Perfect form! {duration:.1f}s'
    elif body_angle > 140:
        quality = 'good'
        feedback = f'Good form (slight sag) {duration:.1f}s'
        form_issues.append("Minor sagging")
    else:
        quality = 'poor'
        feedback = f'Hips too low/high {duration:.1f}s'
        form_issues.append("Fix hip position")
    
    form_quality = 'good' if quality in ['excellent', 'good'] else 'needs_improvement'
    
    return feedback, quality, form_quality