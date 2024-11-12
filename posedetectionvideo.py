import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Function to calculate the angle between two points
def calculate_angle(landmark_1, landmark_2):
    x1, y1, z1 = landmark_1.x, landmark_1.y, landmark_1.z
    x2, y2, z2 = landmark_2.x, landmark_2.y, landmark_2.z
    
    slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
    angle_radians = math.atan(slope)
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

# Define input and output video files
input_video_path = 'input_video.mp4'
output_video_path = 'output_video.mp4'

# Capture video from file instead of webcam
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize VideoWriter for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

landmark_visibility_count = {landmark.name: 0 for landmark in mp_pose.PoseLandmark}
landmark_connections_count = {}
pose_connections = mp_pose.POSE_CONNECTIONS
VISIBILITY_THRESHOLD = 0.5

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Process the image
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            # Update visibility counts for landmarks
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if landmark.visibility > VISIBILITY_THRESHOLD:
                    landmark_name = mp_pose.PoseLandmark(idx).name
                    landmark_visibility_count[landmark_name] += 1

            # Iterate through pose connections to calculate angles
            for connection in pose_connections:
                landmark_1_idx, landmark_2_idx = connection
                landmark_1 = results.pose_landmarks.landmark[landmark_1_idx]
                landmark_2 = results.pose_landmarks.landmark[landmark_2_idx]
                angle = calculate_angle(landmark_1, landmark_2)

                if landmark_1.visibility > VISIBILITY_THRESHOLD and landmark_2.visibility > VISIBILITY_THRESHOLD:
                    sorted_connection = tuple(sorted([landmark_1_idx, landmark_2_idx]))
                    if sorted_connection in landmark_connections_count:
                        count, _ = landmark_connections_count[sorted_connection]
                        landmark_connections_count[sorted_connection] = (count + 1, angle)
                    else:
                        landmark_connections_count[sorted_connection] = (1, angle)

                    # Display angle on midpoint of connection
                    x1, y1 = int(landmark_1.x * image.shape[1]), int(landmark_1.y * image.shape[0])
                    x2, y2 = int(landmark_2.x * image.shape[1]), int(landmark_2.y * image.shape[0])
                    midpoint_x, midpoint_y = (x1 + x2) // 2, (y1 + y2) // 2
                    angle_text = f'{angle:.2f}°'
                    cv2.putText(image, angle_text, (midpoint_x, midpoint_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw the pose landmarks on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Write the frame to the output video
        out.write(image)

    # Release resources
    cap.release()
    out.release()
