import os
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

model = load_model("model_best.h5")
cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
path = "emotional_faces_test_data/"
for i in range(7):
    count=0
    for j in range(60):
        img = cv2.imread(f"emotional_faces_test_data/man_sub{j+1}/{emotion_dict[i]}.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img,verbose=None)
            if(emotion_dict[int(np.argmax(prediction))]==emotion_dict[i]):
                count+=1
    for j in range(60):
        img = cv2.imread(f"emotional_faces_test_data/woman_sub{j+1}/{emotion_dict[i]}.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img,verbose=None)
            if(emotion_dict[int(np.argmax(prediction))]==emotion_dict[i]):
                count+=1
    print(count/120)
