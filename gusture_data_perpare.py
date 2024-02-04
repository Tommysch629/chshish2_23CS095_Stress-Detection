import os
import cv2
import mediapipe as mp
import glob
import pandas as pd
import numpy as np
import math
import pandas as pd
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

#call mediapipe model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
total_landmark_list=[]
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
col.append('class')
dict = {0:'crossing_arms',1:'crossing_fingers',2:'netural&others',3:'touching_faces',4:'touching_jaw',5:'touching_neck'}
path = "gesture_data/"
for i in range(6):
    directory = os.fsencode(f"{path}{dict[i]}")
    for sub_directory in os.listdir(directory):
        filename = os.fsdecode(sub_directory)
        file_directory= os.fsencode(f"{path}{dict[i]}/{filename}")
        for file in os.listdir(file_directory):
            image_name = os.fsdecode(file)
            image = cv2.imread(f"{path}{dict[i]}/{filename}/{image_name}")
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(img_rgb)
            pose_landmarks_list = result.pose_landmarks.landmark
            max_distance = 0
            center_x = (pose_landmarks_list[11].x +pose_landmarks_list[12].x)/2
            center_y = (pose_landmarks_list[11].y +pose_landmarks_list[12].y)/2
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
            normalized_list.append(dict[i])
            total_landmark_list.append(normalized_list)
data = pd.DataFrame(total_landmark_list, columns=col)
data.to_csv('gesture_data.csv', encoding='utf-8', index=False)