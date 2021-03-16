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
import glob
import pandas as pd
import shutil

from PIL import Image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from retinaface.retina_detector import RetinaDetector
import tensorflow_addons as tfa


class EmotionDetectorDrake:
    def __init__(self, model_path):
        self.num_classes = 7
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # global sess
        # # set_session(sess)
        model = Sequential()

        #1st convolution layer
        model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
        model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

        #2nd convolution layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

        #3rd convolution layer
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

        model.add(Flatten())

        #fully connected neural networks
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.load_weights(model_path)
        # model._make_predict_function()
        self.model = model 

    def predict(self, img: Image.Image):
        gray_img = img.convert('L')

        resized_img = gray_img.resize((48,48))
        inp = np.array(resized_img).reshape((1,48,48,1))/255
        inp = tf.constant(inp)

        emotion_predictions = self.model.predict(inp)[0,:]

        return np.argmax(emotion_predictions)


class EmotionDetector:
    def __init__(self, model_path):
        self.net = load_model(model_path, custom_objects={'loss': tfa.losses.SigmoidFocalCrossEntropy()})

    def predict(self, img: Image.Image, get_vector=False):
        gray_img = img.convert('L')

        resized_img = gray_img.resize((48,48))
        inp = np.array(resized_img).reshape((1,48,48,1))/255
        inp = tf.constant(inp)

        pred = self.net(inp).numpy()[0]

        if get_vector:
            return pred[0]
        else:
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
                    emotion='neutral'

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


def inference(args):
    try:
        emotion_detector = EmotionDetector(args.modelfile)
        print("using original emotion model!")
    except:
        emotion_detector = EmotionDetectorDrake(args.modelfile)
        print("using drake emotion model")

    face_detector = RetinaDetector(device='cpu')
    try:
        os.remove("test_result.txt")
    except:
        pass
    root = os.path.join(args.root, "*")
    for folder_path in tqdm.tqdm(glob.glob(root)):
        label = folder_path.split("/")[-1]
        counter = 0
        total = 0
        for idx, img_path in enumerate(glob.glob(os.path.join(folder_path, "*"))):
            img = cv2.imread(img_path)
            bboxes = face_detector.detect_from_image(img)
            for box_idx, bbox in enumerate(bboxes):
                if bbox[-1] >= 0.8:
                    face = Image.fromarray(img).crop(bbox[:4])
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
                        emotion='neutral'

                    total += 1
                    if emotion == label:
                        counter += 1
                    
                    if args.save_result:
                        cv2.imwrite(img, "{}.{}.{}.{}.jpg".format(label, emotion, idx, box_idx))

        with open("test_result.txt", "a") as f:
            f.write("{}: {}\n".format(label, 100*(counter/total)))

def arg_parser():
    P = argparse.ArgumentParser(description='emotion')
    P.add_argument('--job', type=str, default='camtest', help='do 1 of 2 jobs: camera test (camtest), processing images folder (inference)')
    P.add_argument('--root', type=str, required=False, help='rooter path')
    P.add_argument('--modelfile', type=str, required=True, help='model file path')
    P.add_argument('--device', type=str, default='cpu', help='used device: cpu or cuda')
    P.add_argument('--save-result', action='store_true')
    args = P.parse_args()

    return args

if __name__=="__main__":
    args = arg_parser()
    if args.job == 'inference':
        inference(args)
    elif args.job == 'camtest':
        test_retina_emotion(args)
    else:
        raise Exception("Invalid Job.")