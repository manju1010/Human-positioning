import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open video capture (0 for webcam or replace with video file path)
#cap = cv2.VideoCapture(0)  # Replace 0 with "video.mp4" to use a video file
# Specify the video file path
video_file_path = "run1.mp4"  # Replace "video.mp4" with the path to your video file

# Open video capture
cap = cv2.VideoCapture(video_file_path)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame or end of video.")
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(frame_rgb)

    # Draw landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Landmarks
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   # Connections
        )

        # Extract and print landmark data
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z}, visibility: {landmark.visibility})")

    # Resize frame for display (optional)
    display_frame = cv2.resize(frame, (960, 540))

    # Display the processed frame
    cv2.imshow('Pose Estimation', display_frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
