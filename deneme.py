import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points a, b, c.
    a, b, c are numpy arrays with (x, y, z) coordinates.
    """
    ba = a - b
    bc = c - b

    # Calculate dot product and magnitudes
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    # Avoid division by zero
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0
    
    # Calculate angle in radians and convert to degrees
    angle = np.arccos(dot_product / (magnitude_ba * magnitude_bc))
    return np.degrees(angle)

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find pose landmarks
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates of key points
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z])
        
        left_elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z])
        
        left_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z])
        
        # Calculate the angle at the left elbow
        angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Display the angle on the frame
        cv2.putText(frame, f'Elbow Angle: {int(angle)} degrees',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Pose Estimation', frame)

    # Break the loop with the 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
