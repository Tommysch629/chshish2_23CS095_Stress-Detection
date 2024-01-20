import numpy as np
import pandas as pd
from PIL import Image
import os

train_image_path = "data/train/"
val_image_path = "data/val/"
test_image_path = "data/test/"
path = 'fer2013.csv'
fer2013 = pd.read_csv(path)
train_data = fer2013[fer2013['Usage'] == 'Training']
public_test_data = fer2013[fer2013['Usage'] == 'PublicTest']
private_test_data = fer2013[fer2013['Usage'] == 'PrivateTest']
emotion_type = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

for i, row in enumerate(train_data.index):
    image = np.fromstring(train_data.loc[row, 'pixels'], dtype=int, sep=' ')
    image = np.reshape(image, (48, 48))
    new_image = Image.fromarray(np.uint8(image),'L')
    filename = '{}.png'.format(i)
    emotion_label = emotion_type[train_data.loc[row, 'emotion']]
    new_image.save(f"{train_image_path}{emotion_label}/{filename}")

for i, row in enumerate(private_test_data.index):
    image = np.fromstring(private_test_data.loc[row, 'pixels'], dtype=int, sep=' ')
    image = np.reshape(image, (48, 48))
    new_image = Image.fromarray(np.uint8(image),'L')
    filename = '{}.png'.format(i)
    emotion_label = emotion_type[private_test_data.loc[row, 'emotion']]
    new_image.save(f"{val_image_path}{emotion_label}/{filename}")

for i, row in enumerate(public_test_data.index):
    image = np.fromstring(public_test_data.loc[row, 'pixels'], dtype=int, sep=' ')
    image = np.reshape(image, (48, 48))
    new_image = Image.fromarray(np.uint8(image),'L')
    filename = '{}.png'.format(i)
    emotion_label = emotion_type[public_test_data.loc[row, 'emotion']]
    new_image.save(f"{test_image_path}{emotion_label}/{filename}")