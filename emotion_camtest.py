import os
import glob
import tqdm
import numpy as np
import cv2
import argparse
import time
import torch
import logging

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

class emotion:
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

    def predict(self, img: np.ndarray) -> np.ndarray:
        import ipdb; ipdb.set_trace()


def test_retina_emotion(args):
    cap = cv2.VideoCapture(0)
    model = emotion(args.modelfile)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame = cv2.flip(frame, 0)
        key = cv2.waitKey(1) & 0xFF
        pred = model.predict(frame)

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