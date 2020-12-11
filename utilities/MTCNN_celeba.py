#速度过慢，使用HD-celebA-cropper中的align.py代替

import os
import cv2
import numpy as np
import sqlite3
import dlib
from sklearn.model_selection import train_test_split
from mtcnn import FaceDetector

from PIL import Image, ImageDraw, ImageOps, ImageFile
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

# Path variables
# image_path = "C:/Users/Mingrui/Desktop/GAN/celeba/img_align_celeba/"
image_path = "C:/Users/Mingrui/Desktop/test/"
image_save_path = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/celeba_vggface2/'

os.makedirs(image_save_path, exist_ok=True)

detector = FaceDetector()

# remove bboxes that is too small
def detect_bboxes(bboxes):
    new_bboxes = []
    # the minimal size of bbooxes we want to keep
    min_size = 100
    for b in bboxes:
        box_w = b[2] - b[0]
        box_h = b[3] - b[1]

        if box_w > min_size or box_h > min_size:
            new_bboxes.append(b)
    return new_bboxes

def largest_bbox(bboxes):
    if len(bboxes) <= 1:
        return bboxes[0]

    highest_area = 0
    box = None

    for b in bboxes:
        box_w = b[2] - b[0]
        box_h = b[3] - b[1]
        area = box_h * box_w

        if area > highest_area:
            box = b
            highest_area = area
    return box

def get_face_count(image_path):
    image = Image.open(image_path)

    try:
        bboxes, _ = detector.detect(image)
        bboxes = detect_bboxes(bboxes)
        return len(bboxes)
    except Exception:
        return 0

def expand_image(image, padding):
    padding = tuple(map(lambda v: max(0, v), padding))
    return ImageOps.expand(image, padding)

def get_raw_data(image_path, database_path):
    images = []
    landmarks = []
    visibility = []
    pose = []
    gender = []
    paths = []
    paths_cropped = []

    # Image counter
    counter = 1
    grey_count = 0
    # Open the sqlite database
    image_count = 0

    pose_label = np.load("C:/Users/Mingrui/Desktop/GAN/celeba/img_align_celeba/pose_label_celeba.npz", allow_pickle= True)
    image_paths = pose_label["path"]
    pose_angles = pose_label["pose"]

    for input_path in image_paths:

        print(input_path)
        image = Image.open(input_path)

        if len(image.split()) < 3:

            grey_count = grey_count + 1
            continue

        face_count = get_face_count(input_path)

        if face_count > 1 or face_count == 0:
            print('(skip： mlutiple faces)')
            continue
        print(' (OK)')

        paths.append(input_path)

        # Image dimensions
        image_width, image_height = image.size

        # Run MTCNN on the image
        detector = FaceDetector()
        bounding_boxes, _ = detector.detect(image)
        bounding_boxes = [largest_bbox(bounding_boxes)]

        for b in bounding_boxes:
            b = list(map(lambda v: max(v, 0), b))
            face_x = int(b[0])
            face_y = int(b[1])
            face_w = int(b[2] - b[0])
            face_h = int(b[3] - b[1])

            # Resize the MTCNN facebox by 1.3:
            resize_ratio = 2
            face_w_resized = round(face_w * resize_ratio)
            face_h_resized = round(face_h * resize_ratio)
            face_x_resized = round(face_x - face_w * (resize_ratio - 1) / 2)
            face_y_resized = round(face_y - face_h * (resize_ratio - 1) / 2)
            # face_x_resized = round(face_x - face_w * 0.5)
            # face_y_resized = round(face_y - face_h * 0.5)

            face_left = face_x_resized
            face_upper = face_y_resized
            face_right = face_x_resized + face_w_resized
            face_lower = face_y_resized + face_h_resized

            facebox_crop = image.crop((face_left, face_upper, face_right, face_lower))

            cropped_facebox_aspect_ratio = face_w_resized / face_h_resized  # w/h

            if face_w_resized < face_h_resized:
                crop_width = 256
                crop_height = round(256 * (1 / cropped_facebox_aspect_ratio))
                # detecting whether to use face padding
                if image_width < 256 or crop_height > image_height:
                    if image_width < 256 and crop_height > image_height:
                        padding = (0, crop_height - face_h_resized, 256 - face_w_resized, 0)
                        image = expand_image(image, padding)
                    elif image_width < 256:
                        # padding (left up, right, bottom)
                        padding = (0, 0, 256 - face_w_resized, 0)
                        image = expand_image(image, padding)
                    elif image_height < face_h_resized:
                        padding = (0, crop_height - face_h_resized, 0, 0)
                        image = expand_image(image, padding)
            else:
                crop_height = 256
                crop_width = round(256 * cropped_facebox_aspect_ratio)
                if image_height < 256 or crop_width > image_width:
                    if image_height < 256 and crop_width > image_width:
                        padding = (0, 256 - image_height, crop_width - image_width, 0)
                        image = expand_image(image, padding)
                    elif image_height < 256:
                        # padding (left up, right, bottom)
                        padding = (0, 256 - image_height, 0, 0)
                        image = expand_image(image, padding)
                    elif crop_width > image_width:
                        padding = (0, 0, crop_width - image_width, 0)
                        image = expand_image(image, padding)
            facebox_crop_256 = facebox_crop.resize((crop_width, crop_height))

            # Center crop a 224x224 region:
            crop_obj = transforms.CenterCrop((224, 224))
            facebox_crop_256_centercrop_224 = crop_obj(facebox_crop_256)

            image_name = input_path.split('/')[-1]
            image_name = os.path.splitext(image_name)[0]
            crop_image_path = image_save_path + image_name + '_crop'+ ".jpg"
            facebox_crop_256_centercrop_224.save(crop_image_path)
            image_count += 1

            images.append(facebox_crop_256_centercrop_224)
            paths_cropped.append(crop_image_path)

            print("Counter: " + str(counter))
            counter = counter + 1
        # c.close()
    print("Images list items: ", len(images))
    print("Pose list items: ", len(pose))
    print("Grey image number: " + str(grey_count))
    return paths, paths_cropped, images, landmarks, pose, pose_angles


paths, paths_cropped, images, landmarks, pose, pose_angles = get_raw_data(image_path, image_path)
np.savez("D:/AFLW/numpylist2/pose_data_with_name.npz", path=paths_cropped, pose=pose_angles)









