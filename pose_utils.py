import numpy as np
from exercises.pushup_rules import PUSHUP
from exercises.squat_rules import SQUAT
from exercises.plank_rules import PLANK

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def analyze_pushup(keypoints):
    left_shoulder, left_elbow, left_wrist = keypoints[5], keypoints[7], keypoints[9]
    angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    if angle < PUSHUP['elbow_down_max']:
        return ('Down position', 'down')
    elif angle > PUSHUP['elbow_up_min']:
        return ('Up position', 'up')
    else:
        return (f'Elbow angle {int(angle)}', None)


def analyze_squat(keypoints):
    left_hip, left_knee, left_ankle = keypoints[11], keypoints[13], keypoints[15]
    angle = calculate_angle(left_hip, left_knee, left_ankle)

    if angle > SQUAT['knee_max']:
        return ('Not low enough', None)
    elif angle < SQUAT['knee_min']:
        return ('Too deep', 'down')
    else:
        return ('Good depth', None)


def analyze_plank(keypoints):
    left_shoulder, left_hip, left_ankle = keypoints[5], keypoints[11], keypoints[15]
    angle = calculate_angle(left_shoulder, left_hip, left_ankle)

    if angle > PLANK['back_good']:
        return ('Good form', None)
    elif angle > 140:
        return ('Slight sagging', None)
    else:
        return ('Hips too low or high', None)