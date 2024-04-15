import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pandas as pd
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import cv2
import math
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
import numpy as np
import time

#load model
model = load_model("model.h5")
model_test = load_model("gesture_model.h5")
cv2.ocl.setUseOpenCL(False)

#class dict
emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
dict = {0:'crossing_arms',1:'crossing_fingers',2:'netural&others',3:'touching_faces',4:'touching_jaw',5:'touching_neck'}

facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
col = []
for i in range(23):
    name = mp_pose.PoseLandmark(i).name
    name_x = name + '_x'
    name_y = name + '_y'
    name_z = name + '_z'
    name_v = name + '_v'
    col.append(name_x)
    col.append(name_y)
    col.append(name_z)
    col.append(name_v)
    
#video
cap = cv2.VideoCapture(0)
startTime_gesture = time.time()
startTime_emotion = time.time()
s_stress=0

def valid_gesture(pose_landmarks_list):
    if pose_landmarks_list[16].y<pose_landmarks_list[14].y or pose_landmarks_list[15].y<pose_landmarks_list[13].y:
        return True
    else:
        return False
        
def gesture(frame,display_frame,q_gesture):
    global startTime_gesture
    fpsLimit = 1
    total_landmark_list=[]   
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    if result.pose_landmarks:
        pose_landmarks_list = result.pose_landmarks.landmark
        max_distance = 0
        center_x = (pose_landmarks_list[11].x +pose_landmarks_list[12].x)/2
        center_y = (pose_landmarks_list[11].y +pose_landmarks_list[12].y)/2
        if(valid_gesture(pose_landmarks_list)):
            normalized_list=[]
            for k in range(23):
                distance = math.sqrt(
                        (pose_landmarks_list[k].x - center_x)**2 + (pose_landmarks_list[k].y - center_y)**2)
                if(distance > max_distance):
                    max_distance = distance
            for k in range(23):
                normalized_list.append((pose_landmarks_list[k].x - center_x)/max_distance)
                normalized_list.append((pose_landmarks_list[k].y - center_y)/max_distance)
                normalized_list.append(pose_landmarks_list[k].z/max_distance)
                normalized_list.append(pose_landmarks_list[k].visibility)
            total_landmark_list.append(normalized_list)
            data = pd.DataFrame(total_landmark_list, columns=col)
            prediction=model_test.predict(data, verbose=0)
            maxindex = int(np.argmax(prediction))
        else:
            maxindex=2
        mp.solutions.drawing_utils.draw_landmarks(display_frame[0],
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())
        cv2.putText(display_frame[0], dict[maxindex], (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2,
                    cv2.LINE_AA)
        nowTime = time.time()
        if (int(nowTime - startTime_gesture)) >= fpsLimit:
            startTime_gesture = time.time() 
            q_gesture.put(maxindex)
            return maxindex

def emotion(frame,display_frame,q_emotion):
    global startTime_emotion
    fpsLimit = 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img,verbose=0)
        maxindex = int(np.argmax(prediction))
        cv2.rectangle(display_frame[0], (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        cv2.putText(display_frame[0], emotion_dict[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2,
                    cv2.LINE_AA)
        nowTime = time.time()
        if (int(nowTime - startTime_emotion)) >= fpsLimit:
            startTime_emotion = time.time() 
            q_emotion.put(maxindex)
            return maxindex

def stress_score(c_emotion,c_gesture,q_score):
    global s_stress
    if c_emotion!=None:
        if c_emotion==0 or c_emotion==1 or c_emotion==2 or c_emotion==5:
            s_stress+=1.2
            if c_gesture!=None:
                if c_gesture!=2:
                    s_stress+=2
        elif c_emotion==3 or c_emotion==6:
            s_stress-=0.3
            if c_gesture!=None:
                if c_gesture!=2:
                    s_stress+=0.75
        elif c_emotion==4:
            if c_gesture!=None:
                if c_gesture!=2:
                    s_stress+=1.5
        if s_stress>100:
            s_stress=100
        elif s_stress<0:
            s_stress=0
        q_score.put(s_stress)
    return s_stress
    
def main(q_score,q_emotion,q_gesture):
    while True:
        ret, frame = cap.read()
        if not ret:
            break;
        copy_frame=frame.copy()
        display_frame=[copy_frame]
        c_emotion=emotion(frame,display_frame,q_emotion,)
        c_gesture=gesture(frame,display_frame,q_gesture,)
        s_score=stress_score(c_emotion,c_gesture,q_score,)
        cv2.putText(display_frame[0], str(round(s_score,1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2,
                    cv2.LINE_AA)
        cv2.imshow('Video', cv2.resize(display_frame[0], (400, 280), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()