from models.image_to_latent import LandMarksRegressor
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
import torch
from utilities.images import load_images, images_to_video, save_image
from glob import glob
import numpy as np
import cv2
import os
from functools import partial
from PIL import Image, ImageDraw
import json
import requests
def draw_face_landmark(name, landmarkToPaint):
    # draw face's landmark
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 可以为 0 、4、8
    _DEAFAULT_JPG_QUALITY = 95
    name_landmark_str = name
    img_crop =cv2.imread(name)
    image_draw = img_crop
    # if image_draw.shape[0] != 224 or image_draw.shape[1] != 224:
    #     image_draw = cv2.resize(image_draw, (224,224),interpolation = cv2.INTER_AREA)
    #image_draw = cv2.resize(image_draw, (256, 256), interpolation=cv2.INTER_AREA)
    for landmark in landmarkToPaint:
        for i in range(0, len(landmark), 2):
            point1 = int(landmark[i])
            point2 = int(landmark[i+1])
            point_tuple = (point1, point2)
            image_draw = cv2.circle(image_draw, point_tuple, point_size, point_color, thickness)
            # draw = ImageDraw.Draw(img_crop)
            # draw = draw.point((point1, point2), 'red')5landmark_painted
    landmarks_image_path = r"C:\Users\Mingrui\Desktop\datasets\5landmark_painted"
    landmarks_image_path = r"C:\Users\Mingrui\Desktop\datasets\AFLW_painted"
    landmarks_image_path = r"C:\Users\Mingrui\Desktop\datasets\celeba_validsationset_sample_painted"
    #landmarks_image_path = r"C:\Users\Mingrui\Desktop\datasets\Celeba_landamrkRegressor_Test_Rwingloss\\"
    landmarks_image_path = r"C:\Users\Mingrui\Desktop\datasets\\celeba_validsationset_sample_painted\Muhammad_result\\"
    landmarks_image_path = os.path.join(landmarks_image_path, os.path.basename(name))
    #imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEAFAULT_JPG_QUALITY])
    # # print(landmarks_image_path)
    cv2.imwrite(landmarks_image_path, image_draw)

    # draw.save(landmarks_image_path)


def Muhammd_faceNet(image_file):
    url = 'https://cvssp.org/projects/facer2vm/fverify/px.php?dst=/verify_api'
    img = open(image_file, 'rb')
    file_obj = {}
    file_obj['img1'] = img
    data_obj = {}
    data_obj['img1_data'] = '0,0,0,0'
    data_obj['qfeat_only'] = 'true'
    data_obj['apikey'] = 'KIoonmaslanlebw3XyoZ5Cqke09E'
    x = requests.post(url, files=file_obj, data=data_obj)

    try:
        resp = json.loads(x.text)
    except  Exception:
        print(f'failed: failed to parse resp - {x.text}')
        pass

    if 'status' in resp and resp['status'] == 'Error':
        print(f' failed: {resp["message"]}')
        pass

    features = np.array(resp['query_feat'])

    features = torch.from_numpy(features).cuda()
    features = features.unsqueeze(0).float()
    return features

image_directory = 'C:/Users/Mingrui/Desktop/datasets/MTCNN_processed/'
image_directory = 'C:/Users/Mingrui/Desktop/datasets/celeba_imagesample_nocentercrop/'
#image_directory = r"C:\Users\Mingrui\Desktop\datasets\celeba_validsationset_sample\\"
image_directory = r"C:\Users\Mingrui\Desktop\datasets\celeba_validsationset_sample\\"
#image_directory = r"C:\Users\Mingrui\Desktop\datasets\celeba_imagesample_nocentercrop_profile\\"
#image_directory = 'C:/Users/Mingrui/Desktop/datasets/celeba_imagesample/'
image_directory = r"C:\Users\Mingrui\Desktop\datasets\celeba_validsationset_sample\Mhuammad_test\\"

filenames = glob(image_directory + "*.jpg")
pose_record = []
vgg_processing = VGGFaceProcessing()
vgg_face_dag = resnet50_scratch_dag(r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\resnet50_scratch_dag.pth').cuda().eval()
Train_Muhammad_regressor = True
output_count = 3
#pose_regressor = PoseRegressor(output_count).cuda()
landmarks_num = 68
Train_Muhammad_regressor = True
if Train_Muhammad_regressor == True:
    input_features = 512
else:
    input_features = 2048
landmark_regressor = LandMarksRegressor(landmarks_num,input_features).cuda()
if landmarks_num == 68:
    if Train_Muhammad_regressor == True:
        landmark_regressor.load_state_dict(torch.load(r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\Muhammad_Celeba_Regressor_68_landmarks.pt"))
    else:
        landmark_regressor.load_state_dict(torch.load(
            r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\Celeba_Regressor_68_landmarks.pt"))
elif landmarks_num == 5:
    if Train_Muhammad_regressor == True:
        landmark_regressor.load_state_dict(torch.load(
            r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\Muhammad_celebaregressor_5_landmarks.pt"))
    else:
        landmark_regressor.load_state_dict(torch.load(
        r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\celebaregressor_5_landmarks.pt"))
landmark_regressor.eval()
for filename in filenames:
    if Train_Muhammad_regressor == True:
        try:
            descriptors = Muhammd_faceNet(filename)
        except KeyError:
            continue

    else:
        image = load_images([filename])
        image_torch = torch.from_numpy(image).cuda()

        image = vgg_processing(image_torch)
        descriptors = vgg_face_dag(image).cuda()
    pred_landmarks = landmark_regressor(descriptors)

    if Train_Muhammad_regressor == True:
        image_name  = filename.split('\\')[-1]
        dir = "C:/Users/Mingrui/Desktop/Github/HD-CelebA-Cropper/data/aligned/align_size(224,224)_move(0.250,0.000)_face_factor(0.800)_jpg/data/"
        filename = dir + image_name
    draw_face_landmark(filename, pred_landmarks)
    pred_landmarks = str(pred_landmarks)
    pose_record.append(filename + pred_landmarks)

#np.save(r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\pose_test_record.npy", pose_record)



