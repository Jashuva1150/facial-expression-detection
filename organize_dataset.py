import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# Load the dataset
data = pd.read_csv('path/to/fer2013.csv')

# Create the necessary directories
base_dir = 'data/'
os.makedirs(base_dir + 'train', exist_ok=True)
os.makedirs(base_dir + 'validation', exist_ok=True)
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
for emotion in emotions:
    os.makedirs(base_dir + 'train/' + emotion, exist_ok=True)
    os.makedirs(base_dir + 'validation/' + emotion, exist_ok=True)

# Helper function to save images
def save_images(data, usage, base_dir):
    for index, row in data.iterrows():
        emotion = emotions[row['emotion']]
        img = np.array(row['pixels'].split(), dtype='uint8').reshape((48, 48))
        img = cv2.resize(img, (48, 48))
        img_path = os.path.join(base_dir, usage, emotion, f'{index}.jpg')
        cv2.imwrite(img_path, img)

# Split the data into train and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['emotion'], random_state=42)

# Save training images
save_images(train_data, 'train', base_dir)

# Save validation images
save_images(val_data, 'validation', base_dir)

print("Dataset organized successfully")
