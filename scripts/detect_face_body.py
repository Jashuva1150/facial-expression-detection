import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection and Pose
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def detect_face_and_body():
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
            mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2) as pose:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert the BGR frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Perform face detection
            face_results = face_detection.process(image)

            # Perform pose detection
            pose_results = pose.process(image)

            # Convert the image color back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw face detection results
            if face_results.detections:
                for detection in face_results.detections:
                    mp_drawing.draw_detection(image, detection)

            # Draw pose detection results
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the output
            cv2.imshow('Face and Body Detection', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
