import os
import glob
import tqdm
import numpy as np
import cv2
import argparse
import time
import torch
import logging
import tensorflow as tf

from PIL import Image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from retinaface.retina_detector import RetinaDetector

class EmotionDetector:
    def __init__(self, model_path):
        self.net = self.create_net()
        self.net = load_model(model_path)

    def create_net(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        return model

    def predict(self, img: Image.Image):
        gray_img = img.convert('L')

        resized_img = gray_img.resize((48,48))
        inp = np.array(resized_img).reshape((1,48,48,1))
        inp = tf.constant(inp)

        pred = self.net(inp).numpy()
        return np.argmax(pred)


def test_retina_emotion(args):
    cap = cv2.VideoCapture(0)
    emotion_detector = EmotionDetector(args.modelfile)
    face_detector = RetinaDetector(device='cpu')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        key = cv2.waitKey(1) & 0xFF
        bboxes = face_detector.detect_from_image(frame)
        for bbox in bboxes:
            if bbox[-1] >= 0.8:
                face = Image.fromarray(frame).crop(bbox[:4])
                flag = emotion_detector.predict(face)
                if flag==0:
                    emotion='angry'
                elif flag==1:
                    emotion='disgusted'
                elif flag==2:
                    emotion='fearful'
                elif flag==3:
                    emotion='happy'
                elif flag==4:
                    emotion='sad'
                elif flag==5:
                    emotion='surprised'
                elif flag==6:
                    emotion=='neutral'

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,255,0), thickness=2)
                frame = cv2.putText(frame, 
                                    emotion, 
                                    (bbox[0], bbox[1]), 
                                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1,
                                    color=(255,0,0), 
                                    thickness=2, 
                                    lineType=cv2.LINE_AA)

        cv2.imshow("", frame)

        if key == ord("q"):
            break

def arg_parser():
    P = argparse.ArgumentParser(description='emotion')
    P.add_argument('--modelfile', type=str, required=True, help='model file path')
    P.add_argument('--device', type=str, default='cpu', help='used device: cpu or cuda')
    args = P.parse_args()

    return args

if __name__=="__main__":
    args = arg_parser()
    test_retina_emotion(args)