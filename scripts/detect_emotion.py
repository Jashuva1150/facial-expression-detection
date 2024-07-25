import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained emotion detection model
emotion_model = load_model('models/emotion_model.h5')

# Labels for the emotion model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(face_image):
    face_image = cv2.resize(face_image, (48, 48))
    face_image = face_image / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    face_image = np.expand_dims(face_image, axis=-1)
    emotion_prediction = emotion_model.predict(face_image)
    max_index = int(np.argmax(emotion_prediction))
    return emotion_labels[max_index]
