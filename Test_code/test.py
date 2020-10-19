# from https://github.com/ashishtrivedi16/HyperFace/blob/master/src/preprocess.py

import os
import cv2
import numpy as np
import sqlite3
import dlib
from sklearn.model_selection import train_test_split
from mtcnn import FaceDetector
from PIL import Image, ImageDraw,ImageOps
import matplotlib.pyplot as plt
# Path variables


detector = FaceDetector()

#remove bboxed with size < 100
def detect_bboxes(bboxes):
    new_bboxes = []
    for b in bboxes:
        box_w = b[2] - b[0]
        box_h = b[3] - b[1]

        if box_w > 100 or box_h > 100:
            new_bboxes.append(b)

    return new_bboxes


def get_face_count(image_path):
    image = Image.open(image_path)

    try:
        bboxes, _ = detector.detect(image)
        bboxes = detect_bboxes(bboxes)
        return len(bboxes)
    except Exception:
        return 0

image_path = r"D:\AFLW\aflw\data\flickr\2\image09437.jpg"
print("bboxes num:" + str(get_face_count(image_path)))