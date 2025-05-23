import cv2
import dlib
import numpy as np
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("res\shape_predictor_68_face_landmarks.dat")

calibration_frames = 30
relative_landmarks_list = []


def get_face_landmarks(gray):
    faces = detector(gray)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    return face_utils.shape_to_np(shape)


def normalize_to_local_frame(landmarks, idx1=0, idx2=16):
    # Create a local frame using points 1 and 17
    p1, p2 = landmarks[idx1], landmarks[idx2]
    center = (p1 + p2) / 2
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    # Create rotation matrix to align with x-axis
    R = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])

    # Normalize all landmarks
    norm_landmarks = np.dot(landmarks - center, R.T)
    return norm_landmarks


cap = cv2.VideoCapture(0)
print("Calibrating... Please hold your face steady.")

# === Calibration Phase ===
while len(relative_landmarks_list) < calibration_frames:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    landmarks = get_face_landmarks(gray)
    if landmarks is not None:
        rel_landmarks = normalize_to_local_frame(landmarks)
        relative_landmarks_list.append(rel_landmarks)
        cv2.putText(
            frame,
            "Calibrating...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyWindow("Calibration")

# Compute mean of normalized landmark positions
median_relative_landmarks = np.median(np.array(relative_landmarks_list), axis=0)
print("Calibration complete.")

# === Tracking Phase ===
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    landmarks = get_face_landmarks(gray)

    if landmarks is not None:
        # Convert current landmarks to local face frame
        current_rel_landmarks = normalize_to_local_frame(landmarks)

        # Compare current relative landmarks to calibrated
        deviations = np.linalg.norm(
            current_rel_landmarks - median_relative_landmarks, axis=1
        )

        # Draw points and deviations
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"{deviations[i]:.1f}",
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_PLAIN,
                0.6,
                (255, 255, 255),
                1,
            )

    cv2.imshow("Face Tracking (Local)", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
