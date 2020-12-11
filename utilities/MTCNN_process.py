import os
import cv2
import numpy as np
import sqlite3
import dlib
from sklearn.model_selection import train_test_split
from mtcnn import FaceDetector
from PIL import Image, ImageDraw,ImageOps,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from glob import glob
from torchvision import transforms
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


image_directory = r"C:\Users\Mingrui\Desktop\datasets\\"
image_save_path = 'C:/Users/Mingrui/Desktop/datasets/n000106/MTCNN_processed/'


filenames = os.listdir(image_directory)
counter = 0
detector = FaceDetector()
for i in filenames:
    if os.path.splitext(i)[1] == '.jpg':
        i = str(image_directory + i)
        image = Image.open(i)
        image_width, image_height = image.size
        bounding_boxes, _ = detector.detect(image)
        bounding_boxes = [largest_bbox(bounding_boxes)]
        for b in bounding_boxes:
            face_x = int(b[0])
            face_y = int(b[1])
            face_w = int(b[2] - b[0])
            face_h = int(b[3] - b[1])

            # Resize the MTCNN facebox by 2:
            resize_ratio = 2
            face_w_resized = round(face_w * resize_ratio)
            face_h_resized = round(face_h * resize_ratio)
            face_x_resized = round(face_x - face_w * (resize_ratio - 1) / 2)
            face_y_resized = round(face_y - face_h * (resize_ratio - 1) / 2)

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
                        image = ImageOps.expand(image, padding)
                    elif image_width < 256:
                        # padding (left up, right, bottom)
                        padding = (0, 0, 256 - face_w_resized, 0)
                        image = ImageOps.expand(image, padding)
                    elif image_height < face_h_resized:
                        padding = (0, crop_height - face_h_resized, 0, 0)
                        image = ImageOps.expand(image, padding)
            else:
                crop_height = 256
                crop_width = round(256 * cropped_facebox_aspect_ratio)
                if image_height < 256 or crop_width > image_width:
                    if image_height < 256 and crop_width > image_width:
                        padding = (0, 256 - image_height, crop_width - image_width, 0)
                        image = ImageOps.expand(image, padding)
                    elif image_height < 256:
                        # padding (left up, right, bottom)
                        padding = (0, 256 - image_height, 0, 0)
                        image = ImageOps.expand(image, padding)
                    elif crop_width > image_width:
                        padding = (0, 0, crop_width - image_width, 0)
                        image = ImageOps.expand(image, padding)
            facebox_crop_256 = facebox_crop.resize((crop_width, crop_height))


            crop_obj = transforms.CenterCrop((224, 224))

            facebox_crop_256_centercrop_224 = crop_obj(facebox_crop_256)

            facebox_crop_256_centercrop_224.save(image_save_path +'MTCNN_'+str(counter)+ ".jpg")

            counter += 1


