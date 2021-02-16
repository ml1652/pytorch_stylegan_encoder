# tform00 += tform[0][0]
# tform01 += tform[1][2]
# tform02 += tform[1][2]
# tform01 += tform[1][2]
# tform11 += tform[1][2]
# tform12 += tform[1][2]

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys

import argparse
import traceback
from functools import partial
from multiprocessing import Pool
import os
import re

import dlib
import glob


import numpy as np
import tqdm
from numpy import linalg as LA
import torch.nn.functional as F
import torch
def crop_image_forVGGFACE2(croped_image):
    target_size = 224
    croped_image = croped_image[:, :, 210:906,164:860]

    #croped_image = F.interpolate(croped_image, target_size, mode = 'bilinear')

    return croped_image

def align_crop_opencv(img,
                      src_landmarks,
                      standard_landmarks,
                      celeba_standard_landmark,
                      src_celeba_landmark,
                      crop_size=224,
                      face_factor=0.8,
                      align_type='similarity',
                      order=3,
                      mode='edge'):
    """Align and crop a face image by landmarks.

    Arguments:
        img                : Face image to be aligned and cropped.
        src_landmarks      : [[x_1, y_1], ..., [x_n, y_n]].
        standard_landmarks : Standard shape, should be normalized.
        crop_size          : Output image size, should be 1. int for (crop_size, crop_size)
                             or 2. (int, int) for (crop_size_h, crop_size_w).
        face_factor        : The factor of face area relative to the output image.
        align_type         : 'similarity' or 'affine'.
        order              : The order of interpolation. The order has to be in the range 0-5:
                                 - 0: INTER_NEAREST
                                 - 1: INTER_LINEAR
                                 - 2: INTER_AREA
                                 - 3: INTER_CUBIC
                                 - 4: INTER_LANCZOS4
                                 - 5: INTER_LANCZOS4
        mode               : One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap'].
                             Points outside the boundaries of the input are filled according
                             to the given mode.
    """
    # set OpenCV
    import cv2

    inter = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA,
             3: cv2.INTER_CUBIC, 4: cv2.INTER_LANCZOS4, 5: cv2.INTER_LANCZOS4}
    border = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
              'symmetric': cv2.BORDER_REFLECT, 'reflect': cv2.BORDER_REFLECT101,
              'wrap': cv2.BORDER_WRAP}

    # check
    assert align_type in ['affine', 'similarity'], 'Invalid `align_type`! Allowed: %s!' % ['affine', 'similarity']
    assert order in [0, 1, 2, 3, 4, 5], 'Invalid `order`! Allowed: %s!' % [0, 1, 2, 3, 4, 5]
    assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], 'Invalid `mode`! Allowed: %s!' % ['constant', 'edge', 'symmetric', 'reflect', 'wrap']

    # crop size
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        crop_size_h = crop_size[0]
        crop_size_w = crop_size[1]
    elif isinstance(crop_size, int):
        crop_size_h = crop_size_w = crop_size
    else:
        raise Exception('Invalid `crop_size`! `crop_size` should be 1. int for (crop_size, crop_size) or 2. (int, int) for (crop_size_h, crop_size_w)!')



    # estimate transform matrix
    trg_landmarks = standard_landmarks * max(crop_size_h, crop_size_w) * face_factor + np.array([crop_size_w // 2, crop_size_h // 2])

    if align_type == 'affine':
        tform = cv2.estimateAffine2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]
    else:
        tform = cv2.estimateAffinePartial2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]

    #calcaute the scale of tform
    m1 = np.mat('0;0;1')
    m2 = np.mat('1;0;1')
    p1 = tform.dot(m1)
    p2 = tform.dot(m2)
    scale = LA.norm(p2 - p1) # defualt is Frobenius norm

    # change the translations part of the transformation matrix for downwarding vertically
    tform[1][2] = tform[1][2] + 20*scale

    # warp image by given transform
    output_shape = (crop_size_h, crop_size_w)
    img_crop = cv2.warpAffine(img, tform, output_shape[::-1], flags=cv2.WARP_INVERSE_MAP + inter[order], borderMode=border[mode])

    # #center crop
    # center_crop_size = 224
    # mid_x, mid_y = int(crop_size_w / 2), int(crop_size_h / 2)
    # mid_y = mid_y +16
    # cw2, ch2 = int(center_crop_size / 2), int(center_crop_size / 2)
    # img_crop = img_crop[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]




    # get transformed landmarks
    tformed_landmarks = cv2.transform(np.expand_dims(src_landmarks, axis=0), cv2.invertAffineTransform(tform))[0]
    tformed_celeba_landmarks = cv2.transform(np.expand_dims(src_celeba_landmark, axis=0), cv2.invertAffineTransform(tform))[0]

    return img_crop, tformed_landmarks,tformed_celeba_landmarks,tform


# ==============================================================================
# =                                      param                                 =
# ==============================================================================

parser = argparse.ArgumentParser()
# main
parser.add_argument('--img_dir', dest='img_dir', default="C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/InterFaceGAN/dataset_directory")
parser.add_argument('--save_dir', dest='save_dir', default= r"C:\Users\Mingrui\Desktop\datasets\StyleGANimge_corp")
parser.add_argument('--landmark_file', dest='landmark_file', default=r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\landmark.txt")
parser.add_argument('--standard_landmark_file', dest='standard_landmark_file', default=r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\standard_landmark_68pts.txt")
parser.add_argument('--crop_size_h', dest='crop_size_h', type=int, default=224)
parser.add_argument('--crop_size_w', dest='crop_size_w', type=int, default=224)
parser.add_argument('--move_h', dest='move_h', type=float, default=0.25)
parser.add_argument('--move_w', dest='move_w', type=float, default=0.)
parser.add_argument('--save_format', dest='save_format', choices=['jpg', 'png'], default='jpg')
parser.add_argument('--n_worker', dest='n_worker', type=int, default=4)
# others
parser.add_argument('--face_factor', dest='face_factor', type=float, help='The factor of face area relative to the output image.', default=0.8) #default = 0.5
parser.add_argument('--align_type', dest='align_type', choices=['affine', 'similarity'], default='similarity')
parser.add_argument('--order', dest='order', type=int, choices=[0, 1, 2, 3, 4, 5], help='The order of interpolation.', default=3)
parser.add_argument('--mode', dest='mode', choices=['constant', 'edge', 'symmetric', 'reflect', 'wrap'], default='edge')
args = parser.parse_args()

ignore_landmark = True
draw_landmark = False

# ==============================================================================
# =                                opencv first                                =
# ==============================================================================

_DEAFAULT_JPG_QUALITY = 95
import cv2
imread = cv2.imread
imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEAFAULT_JPG_QUALITY])
align_crop = align_crop_opencv
print('Use OpenCV')

# ==============================================================================
# =                                     run                                    =
# ==============================================================================

# count landmarks
with open(args.landmark_file) as f:
    line = f.readline()
n_landmark = len(re.split('[ ]+', line)[1:]) // 2

# read data
img_names = os.listdir(args.img_dir)
landmarks = np.genfromtxt(args.landmark_file, dtype=np.float, usecols=range(1, n_landmark * 2 + 1)).reshape(-1, n_landmark, 2)

standard_landmark = np.genfromtxt(args.standard_landmark_file, dtype=np.float).reshape(n_landmark, 2)
standard_landmark[:, 0] += args.move_w
standard_landmark[:, 1] += args.move_h

# data dir
save_dir = os.path.join(args.save_dir, 'align_size(%d,%d)_move(%.3f,%.3f)_face_factor(%.3f)_%s' % (args.crop_size_h, args.crop_size_w, args.move_h, args.move_w, args.face_factor, args.save_format))
data_dir = os.path.join(save_dir, 'data')
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

#load celeba 5 point landamrk
celeba_standard_landmark = np.loadtxt(r"C:\Users\Mingrui\Desktop\celeba\Anno\standard_landmark_celeba.txt", delimiter=',').reshape(-1, 5, 2)
celeba_landmark = np.genfromtxt(r"C:\Users\Mingrui\Desktop\celeba\Anno\list_landmarks_celeba.txt", dtype=np.float,usecols = range(1, 5 * 2 + 1), skip_header = 2).reshape(-1, 5, 2)


def generate_landmark(img):
    # location of the model (path of the model).
    Model_PATH = r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\models\shape_predictor_68_face_landmarks.dat"

    # now from the dlib we are extracting the method get_frontal_face_detector()
    # and assign that object result to frontalFaceDetector to detect face from the image with
    # the help of the 68_face_landmarks.dat model
    frontalFaceDetector = dlib.get_frontal_face_detector()

    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Now the dlip shape_predictor class will take model and with the help of that, it will show
    faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

    # Now this line will try to detect all faces in an image either 1 or 2 or more faces
    allFaces = frontalFaceDetector(imageRGB, 0)

    # List to store landmarks of all detected faces
    allFacesLandmark = []

    # Below loop we will use to detect all faces one by one and apply landmarks on them

    for k in range(0, max(1, len(allFaces))):
        # dlib rectangle class will detecting face so that landmark can apply inside of that area
        faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()), int(allFaces[k].top()),
                                           int(allFaces[k].right()), int(allFaces[k].bottom()))

        # Now we are running loop on every detected face and putting landmark on that with the help of faceLandmarkDetector
        detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)

        # count number of landmarks we actually detected on image
        # if k == 0:
        #     print("Total number of face landmarks detected ", len(detectedLandmarks.parts()))

        # Svaing the landmark one by one to the output folder
        for point in detectedLandmarks.parts():
            allFacesLandmark.append([point.x, point.y])
            if draw_landmark:
                img = cv2.circle(img, (point.x, point.y), 2, (0, 0, 255), 3)
                cv2.imshow("preview", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


    return np.array(allFacesLandmark)
from PIL import Image

def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist

img_list = (get_img_file(args.img_dir))

sum_tform = np.array([[0, 0, 0], [0, 0, 0]], np.float64 )
count = 0
for img_path in img_names[:100]:
    temp_img = []
    #img = imread(os.path.join(args.img_dir, img_path))
    #raw_image = Image.open(os.path.join(args.img_dir, img_path))
    raw_image = cv2.imread(os.path.join(args.img_dir, img_path))
    img = np.transpose(raw_image, (2, 0, 1))

    img = np.array(img)
    temp_img.append(img)
    temp_img = np.array(temp_img)
    #img_landmark = generate_landmark(img)

    img = torch.from_numpy(temp_img).cuda()
    img = img.detach()

    img_crop = crop_image_forVGGFACE2(img)



    # img_crop, tformed_landmarks, tformed_celeba_landmarks,tform = align_crop(img,
    #                                          img_landmark,
    #                                          standard_landmark,
    #                                          celeba_standard_landmark,
    #                                          celeba_landmark[1],
    #                                          crop_size=(args.crop_size_h, args.crop_size_w),
    #                                          face_factor=args.face_factor,
    #                                          align_type=args.align_type,
    #                                          order=args.order,
    #                                          mode=args.mode)


    name = os.path.splitext(img_path)[0] + '.' + args.save_format
    path = os.path.join(data_dir, name)
    if not os.path.isdir(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])

    img_crop = img_crop.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    import cv2

    img_crop = cv2.resize(img_crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    img_crop = np.uint8(img_crop)
    dst = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, img_crop)

    #sum_tform += tform


    count +=1

    continue

# average_tfrom = sum_tform/count
#
# np.savetxt(data_dir+"average_tform.txt", average_tfrom)
