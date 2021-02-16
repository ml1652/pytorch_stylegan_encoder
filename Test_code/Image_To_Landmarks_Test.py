from models.image_to_latent import ImageToLandmarks,ImageToLandmarks_batch
from models.vgg_face2 import resnet50_scratch_dag
import torch
from utilities.images import load_images, images_to_video, save_image
from glob import glob
import numpy as np
import cv2
import os
from functools import partial
from PIL import Image, ImageDraw
import torch.nn.functional as F
from torchvision import transforms


image_size = 64
# augments = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

augments = transforms.Compose([
    #transforms.Resize(image_size),
    transforms.ToTensor(),

])


augments1 = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])


augments2 = transforms.Compose([
    transforms.Normalize(mean=[131.0912, 103.8827, 91.4953],std=[1,1,1])
])

# class LoadImages(torch.utils.data.Dataset):
#     def __init__(self, filenames, transforms = None):
#         self.filenames = filenames
#         self.transforms = transforms
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, index):
#         filename = self.filenames[index]
#
#         image = self.load_image(filename)
#         image = Image.fromarray(np.uint8(image))
#
#         image = self.transforms(image)
#
#         return image
#
#     def load_image(self, filename):
#         image = np.asarray(Image.open(filename))
#
#         return image
#
class LoadImages(torch.utils.data.Dataset):
    def __init__(self, filenames, transforms = None):
        self.filenames = filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))

        image = self.transforms(image)
        image = image * 255
        image = torch.round(image)
        return image

    def load_image(self, filename):
        image = np.asarray(Image.open(filename))

        return image

class LoadImages256(torch.utils.data.Dataset):
    def __init__(self, filenames, transforms1 = None , transforms2 = None):
        self.filenames = filenames
        self.transforms1 = transforms1
        self.transforms2 = transforms2
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))

        image = self.transforms1(image)
        image = image * 255
        image = self.transforms2(image)
        image = torch.round(image)
        return image

    def load_image(self, filename):
        image = np.asarray(Image.open(filename))

        return image


class ImageLabelDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, labels, image_size=256, transforms=None):
        self.filenames = filenames
        self.labels = labels
        self.image_size = image_size
        self.transforms1 = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]

        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))

        image = self.transforms1(image)
        image = image * 255
        image = torch.round(image)

        return image, label

    def load_image(self, filename):
        image = np.asarray(Image.open(filename))

        return image


def draw_face_landmark2(name, landmarkToPaint):
    # draw face's landmark
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 可以为 0 、4、8
    image_resize = 224
    name_landmark_str = name
    img_crop =cv2.imread(name)
    img_crop = cv2.resize(img_crop, (image_resize,image_resize), interpolation=cv2.INTER_AREA)
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
    landmarks_image_path = r"C:\Users\Mingrui\Desktop\datasets\Celeba_landamrkRegressor_Test_Rwingloss\\"
    landmarks_image_path = r"C:\Users\Mingrui\Desktop\datasets\Image_to_Landamrks_Test\\"
    landmarks_image_path = os.path.join(landmarks_image_path, os.path.basename(name))
    #imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEAFAULT_JPG_QUALITY])
    # # print(landmarks_image_path)
    cv2.imwrite(landmarks_image_path, image_draw)

    # draw.save(landmarks_image_path)



def draw_face_landmark(name, img, landmarkToPaint):
    # draw face's landmark
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 可以为 0 、4、8
    image_resize = 224
    name_landmark_str = name

    img = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    # gi1 = np.uint8(gi1)

    image_draw = img
    image_draw = np.uint8(image_draw)
    if image_draw.shape[2] != 224:
        image_draw = cv2.resize(image_draw, (image_resize, image_resize), interpolation=cv2.INTER_AREA)

    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)

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
    landmarks_image_path = r"C:\Users\Mingrui\Desktop\datasets\Celeba_landamrkRegressor_Test_Rwingloss\\"
    landmarks_image_path = r"C:\Users\Mingrui\Desktop\datasets\Image_to_Landamrks_Test\\"
    landmarks_image_path = os.path.join(landmarks_image_path, os.path.basename(name))
    #imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEAFAULT_JPG_QUALITY])
    # # print(landmarks_image_path)
    cv2.imwrite(landmarks_image_path, image_draw)

    # draw.save(landmarks_image_path)



image_directory = 'C:/Users/Mingrui/Desktop/datasets/MTCNN_processed/'
image_directory = 'C:/Users/Mingrui/Desktop/datasets/celeba_imagesample_nocentercrop/'
image_directory = "C:/Users/Mingrui/Desktop/datasets/Celeba_landamrkRegressor_Test/"
#image_directory = r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(224,224)_move(0.250,0.000)_face_factor(0.800)_jpg\data"

#image_directory = r"C:\Users\Mingrui\Desktop\test\\"
#image_directory = r"C:\Users\Mingrui\Desktop\datasets\celeba_validsationset_sample\\"

#image_directory = r"C:\Users\Mingrui\Desktop\datasets\celeba_imagesample_nocentercrop_profile\\"
#image_directory = 'C:/Users/Mingrui/Desktop/datasets/celeba_imagesample/'
#######################plot label landmark
# import pandas as pd
# data = pd.read_csv(
#             r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(224,224)_move(0.250,0.000)_face_factor(0.800)_jpg\tformed_landmark_68point.txt",
#             sep=' ',
#             header=None,
#         )
# data = data.to_numpy()
# names = data[:, 0].tolist()
# label_sets = np.stack(data[:,1:].astype('float64'))
# filenames = [f"{image_directory}/{x}" for x in names]
# label_sets = np.stack(data[:,1:].astype('float64'))
#
# train_dataset = ImageLabelDataset(filenames, label_sets, transforms=augments)
# train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=1)
#
# for i, (images, latents) in enumerate(train_generator, 1):
#
#     images, latents = images.cuda(), latents.cuda()
#     draw_face_landmark(filenames[i-1], latents)
###########################################################################


filenames = glob(image_directory + "*.jpg")
pose_record = []
output_count = 3
#pose_regressor = PoseRegressor(output_count).cuda()
landmarks_num = 68
landmark_regressor = ImageToLandmarks_batch(landmark_num = 68).cuda()
if landmarks_num == 68:
    landmark_regressor.load_state_dict(torch.load(r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\Image_to_landmarks_Regressor_batchnorm_lr=0.001.pt"))
elif landmarks_num == 5:
    landmark_regressor.load_state_dict(torch.load(
        r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\celebaregressor_5_landmarks.pt"))
landmark_regressor.eval()
# for filename in filenames:
#
#     image = load_images([filename])
#     # image = loadImage(filename,augments)
#     # image = image.numpy()
#     image = torch.from_numpy(image).cuda()
#
#     image = vgg_processing(image)
#     # image = image / torch.tensor(255).float()
#     image = image.float()
#     pred_landmarks = landmark_regressor(image).cuda()
#
#     draw_face_landmark(filename, pred_landmarks)
#
#
#     pred_landmarks = str(pred_landmarks)
#     pose_record.append(filename + pred_landmarks)

images = LoadImages(filenames, transforms=augments)

#images = LoadImages256(filenames, transforms1=augments1, transforms2=augments2)

train_generator = torch.utils.data.DataLoader(images, batch_size=1)
for i,image in enumerate(train_generator):
    image = image.cuda()

    resized_img = F.interpolate(image, 64, mode='bilinear')

    pred_landmarks = landmark_regressor(resized_img)     #min:-1.16 max =  1.68
    pred_landmarks = pred_landmarks*(224/64)

    draw_face_landmark(filenames[i],image, pred_landmarks)


    pred_landmarks = str(pred_landmarks)
    pose_record.append(filenames[i] + pred_landmarks)

#np.save(r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\pose_test_record.npy", pose_record)