import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


pose_landmarks = [
    "nose",  # 0
    "left_eye_inner",  # 1
    "left_eye",  # 2
    "left_eye_outer",  # 3
    "right_eye_inner",  # 4
    "right_eye",  # 5
    "right_eye_outer",  # 6
    "left_ear",  # 7
    "right_ear",  # 8
    "mouth_left",  # 9
    "mouth_right",  # 10
    "left_shoulder",  # 11
    "right_shoulder",  # 12
    "left_elbow",  # 13
    "right_elbow",  # 14
    "left_wrist",  # 15
    "right_wrist",  # 16
    "left_pinky",  # 17
    "right_pinky",  # 18
    "left_index",  # 19
    "right_index",  # 20
    "left_thumb",  # 21
    "right_thumb",  # 22
    "left_hip",  # 23
    "right_hip",  # 24
    "left_knee",  # 25
    "right_knee",  # 26
    "left_ankle",  # 27
    "right_ankle",  # 28
    "left_heel",  # 29
    "right_heel",  # 30
    "left_foot_index",  # 31
    "right_foot_index"  # 32
]


landmark_angle_rules = {
    ('LEFT_SHOULDER', 'LEFT_HIP'): {  # Example connection
        (0,20):0,
        (21, 60): 0,  # Count for angles between 20° and 30°
        (60, 120): 0,  # Count for angles between 31° and 40°
    },
    ('RIGHT_SHOULDER', 'RIGHT_HIP'): {  # Example connection
        (0,20):0,
        (21, 60): 0,  # Count for angles between 20° and 30°
        (60, 120): 0,  # Count for angles between 31° and 40°
    },
    ('LEFT_SHOULDER','LEFT_ELBOW'):{
        (0,20):0,
        (21,45):0,
        (45,89):0,
        (90,120):0,
    },
    ('RIGHT_SHOULDER','RIGHT_ELBOW'):{
        (0,20):0,
        (21,45):0,
        (45,89):0,
        (90,120):0,
    },
    ('LEFT_ELBOW','LEFT_WRIST'):{
        (0,60):0,
        (60,100):0,
        (100,120):0,
    },
    ('RIGHT_ELBOW','RIGHT_WRIST'):{
        (0,60):0,
        (60,100):0,
        (100,120):0,
    }
}


def update_angle_rule_counts(landmark_1_name, landmark_2_name, angle):
    connection = (landmark_1_name, landmark_2_name)
    if connection in landmark_angle_rules:
        # Ensure angle is a scalar (in case it's passed as a NumPy value)
        if isinstance(angle, np.ndarray):
            angle = angle.item()
        for angle_range, count in landmark_angle_rules[connection].items():
            if angle_range[0] <= angle <= angle_range[1]:
                landmark_angle_rules[connection][angle_range] += 1
                break  # Increment only one matching range


# Açı hesaplama fonksiyonu (iki nokta arasındaki açı)
def calculate_angle(landmark_1, landmark_2, degrees=True):
    # landmark_1 ve landmark_2'nin x, y, z koordinatlarını alıyoruz
    x1, y1, z1 = landmark_1.x, landmark_1.y, landmark_1.z
    x2, y2, z2 = landmark_2.x, landmark_2.y, landmark_2.z

    vector1 = np.array([x1, y1, z1])
    vector2 = np.array([x2, y2, z2])

    dot_product = np.dot(vector1, vector2)

    magnitude_v1 = np.linalg.norm(vector1)
    magnitude_v2 = np.linalg.norm(vector2)  
    
    # Calculate the angle in radians
    angle_rad = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))


    # Optionally, convert the angle to degrees
    if degrees:
        return np.degrees(angle_rad)
    return angle_rad
        

# For webcam input:
cap = cv2.VideoCapture(0)
landmark_visibility_count = {landmark.name: 0 for landmark in mp_pose.PoseLandmark}

# Define a dictionary to track the connections between landmarks and their count, including angle.
# Format: (landmark_1, landmark_2): (count, angle)
landmark_connections_count = {}
CUSTOM_CONNECTIONS = [
    (11, 15),  # LEFT_SHOULDER to LEFT_WRIST (custom connection)
    (12, 16),  # RIGHT_SHOULDER to RIGHT_WRIST (custom connection)
    (11, 12),  # LEFT_SHOULDER to RIGHT_SHOULDER
]

# Define landmark pairs that are connected (based on MediaPipe Pose's POSE_CONNECTIONS).
# These connections will represent the edges in a pose skeleton.
pose_connections = mp_pose.POSE_CONNECTIONS

VISIBILITY_THRESHOLD = 0.5

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            # Update the visibility count for each landmark if visible.
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if landmark.visibility > VISIBILITY_THRESHOLD:
                    landmark_name = mp_pose.PoseLandmark(idx).name
                    landmark_visibility_count[landmark_name] += 1

            # Iterate through each connection in POSE_CONNECTIONS
            for connection in CUSTOM_CONNECTIONS:
                landmark_1_idx, landmark_2_idx = connection

                # Get the landmarks using the indices
                landmark_1 = results.pose_landmarks.landmark[landmark_1_idx]
                landmark_2 = results.pose_landmarks.landmark[landmark_2_idx]

                # Calculate the angle between the two landmarks
                angle = calculate_angle(landmark_1, landmark_2)

                # Check if both landmarks are visible above the threshold
                if landmark_1.visibility > VISIBILITY_THRESHOLD and landmark_2.visibility > VISIBILITY_THRESHOLD:
                    # Sort the indices to avoid counting the same connection twice (e.g., (1, 2) and (2, 1))
                    sorted_connection = tuple(sorted([landmark_1_idx, landmark_2_idx]))

                    # If the connection exists in the dictionary, increment its count and update the angle
                    if sorted_connection in landmark_connections_count:
                        count, _ = landmark_connections_count[sorted_connection]
                        landmark_connections_count[sorted_connection] = (count + 1, angle)
                    else:
                        landmark_connections_count[sorted_connection] = (1, angle)

                    # Get the midpoint of the two landmarks for placing the angle text
                    x1, y1 = int(landmark_1.x * image.shape[1]), int(landmark_1.y * image.shape[0])
                    x2, y2 = int(landmark_2.x * image.shape[1]), int(landmark_2.y * image.shape[0])

                    # Calculate midpoint
                    midpoint_x = (x1 + x2) // 2
                    midpoint_y = (y1 + y2) // 2

                    # Draw the angle text at the midpoint
                    angle_text = f'{angle:.2f}°'
                    cv2.putText(image, angle_text, (midpoint_x, midpoint_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Print the current landmark visibility counts
            print("Current landmark visibility counts:")
            for landmark_name, count in landmark_visibility_count.items():
                print(f"{landmark_name}: {count}")

            # Print the current connection counts with angles
            print("Current landmark connection counts with angles:")
            for connection, (count, angle) in landmark_connections_count.items():
                landmark_1_name = mp_pose.PoseLandmark(connection[0]).name
                landmark_2_name = mp_pose.PoseLandmark(connection[1]).name
                update_angle_rule_counts(landmark_1_name, landmark_2_name, angle)
                print(f"Connection between {landmark_1_name} and {landmark_2_name}: {count} times, Angle: {angle:.2f}°")

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(landmark_angle_rules)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()