import cv2
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# Açı hesaplama fonksiyonu (iki nokta arasındaki açı)
def calculate_angle(landmark_1, landmark_2):
    # landmark_1 ve landmark_2'nin x, y, z koordinatlarını alıyoruz
    x1, y1, z1 = landmark_1.x, landmark_1.y, landmark_1.z
    x2, y2, z2 = landmark_2.x, landmark_2.y, landmark_2.z
    
    slope = (y2 - y1) / (x2 - x1)
    
    # Açıyı radyan cinsinden hesapla
    angle_radians = math.atan(slope)
    
    # Açıyı dereceye çevir
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees
    

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# For webcam input:
cap = cv2.VideoCapture(0)
landmark_visibility_count = {landmark.name: 0 for landmark in mp_pose.PoseLandmark}

# Define a dictionary to track the connections between landmarks and their count, including angle.
# Format: (landmark_1, landmark_2): (count, angle)
landmark_connections_count = {}

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
            for connection in pose_connections:
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
                print(f"Connection between {landmark_1_name} and {landmark_2_name}: {count} times, Angle: {angle:.2f}°")

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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