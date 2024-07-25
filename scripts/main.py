import cv2
import mediapipe as mp
from detect_emotion import detect_emotion
from sentiment_analysis import analyze_sentiment

# Initialize MediaPipe Face Detection and Pose
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    # Initialize webcam
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

            # Draw face detection results and perform emotion detection
            if face_results.detections:
                for detection in face_results.detections:
                    mp_drawing.draw_detection(image, detection)

                    # Get the bounding box of the face
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    (x, y, w, h) = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Extract face image
                    face_image = image[y:y + h, x:x + w]
                    if face_image.size != 0:
                        emotion = detect_emotion(cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY))
                        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw pose detection results
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the output
            cv2.imshow('Face and Body Detection', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
