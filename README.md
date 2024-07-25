# Face and Body Emotion Detection

This project detects emotions from facial expressions using a Convolutional Neural Network (CNN). The model is trained on a dataset of facial expressions and can predict emotions from input images and Videos. The project also includes modules for face and body movement detection and sentiment analysis.

We went through several stages, including dataset preparation, model training, and integration for emotion detection.


File Structure 

```
face_body_emotion_detection/
│
├── data/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── surprise/
│   │   ├── neutral/
│   ├── validation/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── surprise/
│   │   ├── neutral/
│
├── models/
│   └── emotion_model.h5          # This will be created after running the training script
│
├── scripts/
│   ├── detect_emotion.py         # Script for emotion detection
│   ├── detect_face_body.py       # Script for face and body movement detection
│   ├── sentiment_analysis.py     # Script for sentiment analysis
│   ├── main.py                   # Main script to integrate all components
│   └── train_emotion_model.py    # Script to train the emotion detection model
│
├── requirements.txt              # Required Python libraries
└── README.md                     # Project documentation


```


## Setup
 ```
   git clone <repository-url>

 ```

## install requirements

``` 
pip install -r requirements.txt

```

## To create a model (emotional model)

Download the FER-2013 dataset from Kaggle.

Extract the dataset. Typically, you'll get a CSV file or Image file(jpg)

Split the data set into Train and Validation and save it in the /data (folder)

```
python scripts/train_emotion_model.py
```

## To Run the Face and Expression detection

```
python scripts/main.py
```#   f a c i a l - e x p r e s s i o n - d e t e c t i o n  
 #   f a c i a l - e x p r e s s i o n - d e t e c t i o n  
 