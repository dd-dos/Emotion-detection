import os
import glob
import tqdm
import numpy as np
import cv2
import argparse
import time
import torch
import logging

from detection.retinaface.test_widerface import test_retina_face
from PIL import Image
from torchsummary import summary
from inference import FaceAlignment
from uuid import uuid4

logging.getLogger().setLevel(logging.INFO)

def arg_parser():
    P = argparse.ArgumentParser(description='landmark')
    P.add_argument('--face-detector', type=str, required=True, help='face detector model to use')
    P.add_argument('--modelfile', type=str, required=True, help='model file path')
    P.add_argument('--detectmodelfile', type=str, required=True, help='face detect model file')
    P.add_argument('--device', type=str, default='cpu', help='used device: cpu or cuda')
    args = P.parse_args()

    return args
    
def test_retina_landmark(args):
    cap = cv2.VideoCapture(0)
    fa = FaceAlignment(modelfilename=args.modelfile,
                        device=args.device,
                        face_detector=args.face_detector,
                        facedetectmodelfile=args.detectmodelfile)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame = cv2.flip(frame, 0)
        key = cv2.waitKey(1) & 0xFF
        time_0 = time.time()
        frame = fa.draw_landmarks(frame)
        # frame = fa.get_head_pose(frame)
        logging.info("reference time: {}".format(time.time()-time_0))
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

        cv2.imshow("", frame)

        if key == ord("q"):
            break