import argparse

import cv2
from tqdm import tqdm
import numpy as np
import torch
from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.latent_optimizer import LatentOptimizer,LatentOptimizerVGGface,LatentOptimizerLandmarkRegressor
from models.image_to_latent import ImageToLatent
from models.image_to_latent import VGGToLatent, VGGLatentDataset,PoseRegressor
from torchvision import transforms
from models.vgg_face_dag import vgg_face_dag

from models.losses import LatentLoss,IdentityLoss
from utilities.hooks import GeneratedImageHook
from utilities.images import load_images, images_to_video, save_image
from utilities.files import validate_path
from models.latent_optimizer import VGGFaceProcessing,LatentOptimizerVGGface_vgg_to_latent
from models.vgg_face2 import resnet50_scratch_dag

from models.image_to_latent import ImageToLandmarks,ImageToLandmarks_batch
from torch.autograd import Variable

from multi_task.min_norm_solvers import MinNormSolver, gradient_normalizers


parser = argparse.ArgumentParser(description="Find the latent space representation of an input image.")
parser.add_argument("image_path", help="Filepath of the image to be encoded.")
parser.add_argument("--learning_rate", default=1, help="Learning rate for SGD.", type= float)
parser.add_argument("--iterations", default=1000, help="Number of optimizations steps.", type=int)
parser.add_argument("--model_type", default="stylegan_ffhq", help="The model to use from InterFaceGAN repo.", type=str)
parser.add_argument("dlatent_path", help="Filepath to save the dlatent (WP) at.")
parser.add_argument("--weight_landmarkloss", default=1 , help="the weight of landamrkloss in loss  function",type= float)
parser.add_argument("--Inserver", default=False, help="Number of optimizations steps.", type=bool)

args, other = parser.parse_known_args()
if  args.Inserver == False:
    last_image_save_path = r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Non_ID_result\last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+'.jpg'
else:
    last_image_save_path = '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/Non_ID_result/last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+'.jpg'
    import matplotlib
    matplotlib.use('Agg')

#####target image################################
reference_image = load_images(
        [args.image_path]
    )

reference_image = torch.from_numpy(reference_image).cuda()
reference_image = reference_image.detach()

#####input lantent###############################
latent_space_dim = 512
latents_to_be_optimized = torch.zeros((1, 18, 512)).cuda().requires_grad_(True)

#####styleGAN generator##########################
class PostSynthesisProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.min_value = -1
        self.max_value = 1

    def forward(self, synthesized_image):
        synthesized_image = (synthesized_image - self.min_value) * torch.tensor(255).float() / (self.max_value - self.min_value)
        synthesized_image = torch.clamp(synthesized_image + 0.5, min=0, max=255)

        return synthesized_image  #the value between 0-255, dim = [1,3,1024,1024]

synthesizer = StyleGANGenerator(args.model_type).model.synthesis
synthesizer = synthesizer.cuda().eval()

def synthesis_image(dlatents):
    generated_image = synthesizer(dlatents)
    return generated_image

def pose_porcessing_outputImg(img):
    post_synthesis_processing = PostSynthesisProcessing()
    generated_image = post_synthesis_processing(img)
    return generated_image

def input_image_into_StyleGAN(latents_to_be_optimized):
    generated_image = synthesis_image(latents_to_be_optimized)
    generated_image = pose_porcessing_outputImg(generated_image)
    return generated_image

####feed the StyleGAN generated image into vggface2 encoder###################################
import torch.nn.functional as F

class VGGFaceProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 224  # vggface

        #self.mean = torch.tensor([129.186279296875, 104.76238250732422, 93.59396362304688], device="cuda").view(-1, 1,1)
        #self.mean = torch.tensor([0.5066128599877451, 0.41083287257774204, 0.3670351514629289], device="cuda").view(-1, 1, 1)
        self.std = torch.tensor([1, 1, 1], device="cuda").view(-1, 1, 1)

        self.mean = torch.tensor([131.0912, 103.8827, 91.4953], device="cuda").view(-1, 1,1)



    def forward(self, image):
        #image = image / torch.tensor(255).float()
        image = image.float()
        if image.shape[0] != 224  or image.shape[1] != 224:
            image = F.adaptive_avg_pool2d(image, self.image_size)

        image = (image - self.mean) / self.std

        return image
if args.Inserver == False:
    vgg_face_dag = resnet50_scratch_dag(
        r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\resnet50_scratch_dag.pth').cuda().eval()
else:
    vgg_face_dag = resnet50_scratch_dag(
       '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/Pretrained_model/resnet50_scratch_dag.pth').cuda().eval()
def feed_into_Vgg(generated_image):

    features = vgg_face_dag(generated_image)
    return features # referece [0 7]

def image_to_Vggencoder(img):
    vgg_processing = VGGFaceProcessing()
    generated_image = vgg_processing(img)  #vgg_processing use PIL load image
    features = feed_into_Vgg(generated_image) #reference iamge[-131 - 163]  gereated_image[-130.09 123.9]
    return features

########feed the image into Image_to_landmarks model#####################
# landmark_model_image_size = 64
# style_gan_transformation = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize(landmark_model_image_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])



landmark_regressor = ImageToLandmarks_batch(landmark_num=68).cuda().eval()
if args.Inserver == False:
    weights_path = r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\Image_to_landmarks_Regressor_batchnorm_lr=0.001.pt"
else:
    weights_path = '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/Pretrained_model/Image_to_landmarks_Regressor_batchnorm_lr=0.001.pt'
landmark_regressor.load_state_dict(torch.load(weights_path))

def feed_into_Image_to_landmarksRegressor(img):
    # out = []
    # for x_ in img.cpu():
    #     out.append(style_gan_transformation(x_))
    # generated_image = torch.stack(out).cuda()
    target_size = 64
    #img = img/256
    img = F.interpolate(img, target_size, mode='bilinear')
    #generated_image =  torch.nn.functional.normalize(img,dim=2)

    pred_landmarks = landmark_regressor(img)



    return pred_landmarks

####feed the image into vggface2 encoder###################################


########feed the vgg feathures into landamarksRegresor#####################
from models.image_to_latent import LandMarksRegressor

landmarks_num = 68
features_to_landmarkRegressor = LandMarksRegressor(landmarks_num).cuda().eval()

if args.Inserver == False:
    features_to_landmarkRegressor.load_state_dict(torch.load(
        r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\Celeba_Regressor_68_landmarks.pt"))
else:
    features_to_landmarkRegressor.load_state_dict(torch.load(
        '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/Pretrained_model/Celeba_Regressor_68_landmarks.pt'))

def feed_vggFeatures_into_LandmarkRegressor(vgg_features):
    pred_landmarks = features_to_landmarkRegressor(vgg_features)
    #pred_landmarks = str(pred_landmarks)

    return pred_landmarks


###########loss###########################################

class ID_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, pred):
        loss = torch.nn.functional.l1_loss(target, pred)
        return loss


class Landmark_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, pred):
        loss = torch.mean(torch.square(target - pred))
        loss = torch.log(loss)
        return loss


class LatentLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.l2_loss = torch.nn.MSELoss()

    def forward(self, real_features, generated_features, average_dlatents=None, dlatents=None):
        # Take a look at:
        # https://github.com/pbaylies/stylegan-encoder/blob/master/encoder/perceptual_model.py
        # For additional losses and practical scaling factors.

        loss = 0
        # Possible TODO: Add more feature based loss functions to create better optimized latents.

        # Modify scaling factors or disable losses to get best result (Image dependent).

        # VGG16 Feature Loss
        # Absolute vs MSE Loss
        # loss += 1 * self.l1_loss(real_features, generated_features)
        loss += 1 * self.l2_loss(real_features, generated_features)

        # Pixel Loss
        #         loss += 1.5 * self.log_cosh_loss(real_image, generated_image)

        # Dlatent Loss - Forces latents to stay near the space the model uses for faces.
        if average_dlatents is not None and dlatents is not None:
            loss += 1 * 512 * self.l1_loss(average_dlatents, dlatents)

        return loss

#################################################################################
import dlib
from numpy import linalg as LA

def generate_landmark(img, draw_landmark=False):
    # location of the model (path of the model).
    if args.Inserver == False:
        Model_PATH = r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\models\shape_predictor_68_face_landmarks.dat"
    else:
        Model_PATH = "/scratch/staff/ml1652/StyleGAN_Reconstuction_server/Pretrained_model/shape_predictor_68_face_landmarks.dat"

    # now from the dlib we are extracting the method get_frontal_face_detector()
    # and assign that object result to frontalFaceDetector to detect face from the image with
    # the help of the 68_face_landmarks.dat model
    frontalFaceDetector = dlib.get_frontal_face_detector()

    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #img should be (1024,1024,3) (3,1024,1024)


    # Now the dlip shape_predictor class will take model and with the help of that, it will show
    faceLandmarkDetector = dlib.shape_predictor(Model_PATH)
    imageRGB = np.uint8(imageRGB).squeeze()
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
                imageRGB = cv2.circle(imageRGB, (point.x, point.y), 2, (0, 0, 255), 3)
                #imageRGB[point.x,point.y] = [0,0,255]
                cv2.imshow("preview", imageRGB)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return np.array(allFacesLandmark)


def align_crop_opencv(img,
                      src_landmarks,
                      standard_landmarks,
                      celeba_standard_landmark,
                      src_celeba_landmark,
                      crop_size=512,
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

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inter = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA,
             3: cv2.INTER_CUBIC, 4: cv2.INTER_LANCZOS4, 5: cv2.INTER_LANCZOS4}
    border = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
              'symmetric': cv2.BORDER_REFLECT, 'reflect': cv2.BORDER_REFLECT101,
              'wrap': cv2.BORDER_WRAP}

    # check
    assert align_type in ['affine', 'similarity'], 'Invalid `align_type`! Allowed: %s!' % ['affine', 'similarity']
    assert order in [0, 1, 2, 3, 4, 5], 'Invalid `order`! Allowed: %s!' % [0, 1, 2, 3, 4, 5]
    assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], 'Invalid `mode`! Allowed: %s!' % ['constant',
                                                                                                           'edge',
                                                                                                           'symmetric',
                                                                                                           'reflect',
                                                                                                           'wrap']

    # crop size
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        crop_size_h = crop_size[0]
        crop_size_w = crop_size[1]
    elif isinstance(crop_size, int):
        crop_size_h = crop_size_w = crop_size
    else:
        raise Exception(
            'Invalid `crop_size`! `crop_size` should be 1. int for (crop_size, crop_size) or 2. (int, int) for (crop_size_h, crop_size_w)!')

    # estimate transform matrix
    trg_landmarks = standard_landmarks * max(crop_size_h, crop_size_w) * face_factor + np.array(
        [crop_size_w // 2, crop_size_h // 2])

    if align_type == 'affine':
        tform = cv2.estimateAffine2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]
    else:
        tform = cv2.estimateAffinePartial2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0] #tform{2,3}

    # calcaute the scale of tform
    m1 = np.mat('0;0;1')
    m2 = np.mat('1;0;1')
    p1 = tform.dot(m1)
    p2 = tform.dot(m2)
    scale = LA.norm(p2 - p1)  # defualt is Frobenius norm

    # change the translations part of the transformation matrix for downwarding vertically
    tform[1][2] = tform[1][2] + 20 * scale


    # tform = np.round(tform)
    # numpy to tensor
    #tform = torch.tensor(tform).cuda()
    # tform = torch.tensor(tform)

    # tform = torch.tensor(tform, dtype=torch.float)
    #
    # grid = F.affine_grid(tform.unsqueeze(0),(1,3,224,224),align_corners = True)
    #
    # grid = grid.type(torch.FloatTensor).cuda()
    # output = F.grid_sample(img/256, grid,mode="bilinear", padding_mode="border",align_corners=True)

    # warp image by given transform
    output_shape = (crop_size_h, crop_size_w)
    img_crop = cv2.warpAffine(img, tform, output_shape[::-1], flags=cv2.WARP_INVERSE_MAP + inter[order],
                              borderMode=border[mode])

    # #center crop
    # center_crop_size = 224
    # mid_x, mid_y = int(crop_size_w / 2), int(crop_size_h / 2)
    # mid_y = mid_y +16
    # cw2, ch2 = int(center_crop_size / 2), int(center_crop_size / 2)
    # img_crop = img_crop[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]

    # get transformed landmarks
    tformed_landmarks = cv2.transform(np.expand_dims(src_landmarks, axis=0), cv2.invertAffineTransform(tform))[0]
    tformed_celeba_landmarks = cv2.transform(np.expand_dims(src_celeba_landmark, axis=0), cv2.invertAffineTransform(tform))[0]

    return img_crop, tformed_landmarks, tformed_celeba_landmarks


def align_image_fromStylegan_to_vgg(img):
    img_landmark = generate_landmark(img)

    n_landmark = 68
    if args.Inserver == False:
        standard_landmark_file = 'C:/Users/Mingrui/Desktop/Github/HD-CelebA-Cropper/data/standard_landmark_68pts.txt'
        celeba_standard_landmark = np.loadtxt(r"C:\Users\Mingrui\Desktop\celeba\Anno\standard_landmark_celeba.txt",
                                              delimiter=',').reshape(-1, 5, 2)
        celeba_landmark = np.genfromtxt(r"C:\Users\Mingrui\Desktop\celeba\Anno\list_landmarks_celeba.txt",
                                        dtype=np.float,
                                        usecols=range(1, 5 * 2 + 1), skip_header=2).reshape(-1, 5, 2)
    else:
        standard_landmark_file = '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/Pretrained_model/standard_landmark_68pts.txt'
        celeba_standard_landmark = np.loadtxt("/scratch/staff/ml1652/StyleGAN_Reconstuction_server/Pretrained_model/standard_landmark_celeba.txt",
                                              delimiter=',').reshape(-1, 5, 2)
        celeba_landmark = np.genfromtxt("/scratch/staff/ml1652/StyleGAN_Reconstuction_server/Pretrained_model/list_landmarks_celeba.txt",dtype=np.float,
                                        usecols=range(1, 5 * 2 + 1), skip_header=2).reshape(-1, 5, 2)

    standard_landmark = np.genfromtxt(standard_landmark_file, dtype=np.float).reshape(n_landmark, 2)
    move_w = 0
    move_h = 0.25
    standard_landmark[:, 0] += move_w
    standard_landmark[:, 1] += move_h



    i = 0

    img_crop, tformed_landmarks, tformed_celeba_landmarks = align_crop_opencv(img,
                                                                              img_landmark,
                                                                              standard_landmark,
                                                                              celeba_standard_landmark,
                                                                              celeba_landmark[i],
                                                                              crop_size=(
                                                                              224, 224),
                                                                              face_factor=0.8,
                                                                              align_type='similarity',
                                                                              order=1,
                                                                              mode='edge')

    return img_crop
#################################################################################

def drawLandmarkPoint(img, target_landmark, pred_landmark):
    point_size = 1
    target_point_color = (0, 255, 0)  # BGR
    pred_point_color = (0, 0, 255)
    thickness = 1  # 可以为 4
    image_resize = 224
    if args.Inserver == False:
        str1 = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/Non_ID_result/LandmarkPlot__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations) + '_'
        str2 = (args.image_path).split('\\')[-1]
    else:
        str1 = '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/learningrate_Test/LandmarkPlot__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations) + '_'
        str2 = (args.image_path).split('/')[-1]
    landmarks_image_save_path = str1 + str2

    img = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    # gi1 = np.uint8(gi1)

    image_draw = img
    image_draw = np.uint8(image_draw)
    if image_draw.shape[2] != image_resize:

        image_draw = cv2.resize(image_draw, (image_resize, image_resize), interpolation=cv2.INTER_AREA)

    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)

    for landmark in target_landmark:
        for i in range(0, len(landmark), 2):
            point1 = int(landmark[i])
            point2 = int(landmark[i+1])
            point_tuple = (point1, point2)
            image_draw = cv2.circle(image_draw, point_tuple, point_size, target_point_color, thickness)

        for landmark in pred_landmark:
            for i in range(0, len(landmark), 2):
                point1 = int(landmark[i])
                point2 = int(landmark[i + 1])
                point_tuple = (point1, point2)
                image_draw = cv2.circle(image_draw, point_tuple, point_size, pred_point_color, thickness)
    cv2.imwrite(landmarks_image_save_path, image_draw)
    # cv2.imshow("preview", image_draw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
def draw_targetLandmarkPoint(img, target_landmark):
    point_size = 1
    target_point_color = (0, 255, 0)  # BGR
    pred_point_color = (0, 0, 255)
    thickness = 4  # 可以为 0 、4、8
    if args.Inserver == False:
        str1 = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/Non_ID_result/TargetLandmarkPlot__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+ '_'
        str2 = (args.image_path).split('\\')[-1]
    else:
        str1 = '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/learningrate_Test/TargetLandmarkPlot__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss) +'_iter='+str(args.iterations)+ '_'
        str2 = (args.image_path).split('/')[-1]
    landmarks_image_save_path = str1 + str2

    img = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    # gi1 = np.uint8(gi1)

    image_draw = img
    image_draw = np.uint8(image_draw)
    image_resize = 224
    image_draw = cv2.resize(image_draw, (image_resize, image_resize), interpolation=cv2.INTER_AREA)
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)

    for landmark in target_landmark:
        for i in range(0, len(landmark), 2):
            point1 = int(landmark[i])
            point2 = int(landmark[i+1])
            point_tuple = (point1, point2)
            image_draw = cv2.circle(image_draw, point_tuple, point_size, target_point_color, thickness)

    cv2.imwrite(landmarks_image_save_path, image_draw)

####crop the image to feed into VGGface2#################
# def crop_image_forVGGFACE2(croped_image):
#     resize_size = 320
#     target_size = 224
#     move_h = 60
#
#     #croped_image = F.adaptive_avg_pool2d(img, resize_size)
#
#     (w, h) = croped_image.size()[2:]
#     target = [680, 680]
#     x_left = int(w / 2 - target[0] / 2)
#     x_right = int(x_left + target[0])
#
#     y_top = int(h / 2 - target[1] / 2 + move_h)
#     y_bottom = int(y_top + target[1])
#
#     #crop to certain rect area
#     croped_image = croped_image[:, :, y_top:y_bottom, x_left:x_right]
#
#     #resize to 224
#     croped_image = F.interpolate(croped_image, target_size, mode = 'bilinear')
#
#     return croped_image

def crop_image_forVGGFACE2(croped_image):
    target_size = 224
    croped_image = croped_image[:, :, 210:906,164:860]

    croped_image = F.interpolate(croped_image, target_size, mode = 'bilinear')

    return croped_image

# import matplotlib.pyplot as plt
# croped_image = croped_image.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
# gi1 = np.uint8(croped_image)
# plt.imshow(gi1)
# plt.show()

class Non_ID_reconstructor(torch.nn.Module):
    def __init__(self,synthesizer):
        super().__init__()

        self.synthesizer = synthesizer.cuda().eval()

    def forward(self, latents_to_be_optimized):
        generated_image = synthesis_image(latents_to_be_optimized)
        generated_image = pose_porcessing_outputImg(generated_image)

        #generated_image = input_image_into_StyleGAN(latents_to_be_optimized)  # [1, 3, 1024, 1024] [0-255]

        croped_image = crop_image_forVGGFACE2(generated_image)

        L_pred = feed_into_Image_to_landmarksRegressor(croped_image)  # first resize input image from 224 to 64
        L_pred = L_pred * (224 / 64)

        X_pred = image_to_Vggencoder(croped_image.cuda())


        return L_pred, X_pred,generated_image


import matplotlib.pyplot as plt
#plt.use('Agg')
def show_image(img):
    gi1 = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    plt.imshow(gi1)
    plt.show()
##########Trainig#################

optimizer = torch.optim.SGD([latents_to_be_optimized], lr=args.learning_rate)
#optimizer = torch.optim.Adam([latents_to_be_optimized], lr=args.learning_rate)

progress_bar = tqdm(range(args.iterations))

criterion1 = torch.nn.L1Loss()
#criterion1 = LatentLoss()
criterion2 = torch.nn.MSELoss()
#criterion2 = Landmark_loss()

image_size = 1024
size = (image_size, image_size)
fps = 25
# if args.Inserver == False:
#     video = cv2.VideoWriter(r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Non_ID_result\vdieo__lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_'+str(args.image_path[-10:])+'_Non_IDReconstruction.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
# else:
#     video = cv2.VideoWriter(
#         '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/Non_ID_result/vdieo__lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_'+str(args.image_path[-10:])+'_Non_IDReconstruction.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
if args.Inserver == False:
    str1 = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/Non_ID_result/video__lr=' + str(
        args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations) + '_'
    str2 = ((args.image_path).split('\\')[-1]).split('.')[0]+'.avi'
else:
    str1 = '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/learningrate_Test/video__lr=' + str(
        args.learning_rate) + '_b=' + str(args.weight_landmarkloss) +'_iter='+str(args.iterations)+ '_'
    str2 = ((args.image_path).split('/')[-1]).split('.')[0]+'.avi'
video_path = str1 + str2
video = cv2.VideoWriter( video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
count = 0



# reference_image = reference_image.detach().cpu().numpy()
# reference_image = reference_image.squeeze()
# reference_image = reference_image.transpose(1, 2, 0)
# reference_image = align_image_fromStylegan_to_vgg(reference_image)
# reference_image = reference_image.transpose(2, 0, 1)
# reference_image = reference_image[np.newaxis, :]
# reference_image = torch.tensor(reference_image) # [1,3,224,224]
# reference_image = reference_image.cuda()
X_target = image_to_Vggencoder(reference_image) #[1, 3, 224, 224],dtype=torch.uint8
L_target = feed_vggFeatures_into_LandmarkRegressor(X_target)

# for param in Non_ID_reconstructormodel.parameters():
#         param.requires_grad_(False)
loss_plot = []
Landmark_loss_plot = []
Id_loss_plot = []

lamb = torch.tensor([0.]).cuda().requires_grad_(True)
latents_to_be_optimized_cpoy = Variable(latents_to_be_optimized.data.clone(), requires_grad=True).cuda()
lamb_list = []
for gradient_step in range(500):
    generated_image1 = input_image_into_StyleGAN(latents_to_be_optimized_cpoy)
    croped_image1 = crop_image_forVGGFACE2(generated_image1)

    X_target1 = image_to_Vggencoder(reference_image)
    L_target1 = feed_vggFeatures_into_LandmarkRegressor(X_target1)

    L_pred1 = feed_into_Image_to_landmarksRegressor(croped_image1)  # first resize input image from 224 to 64
    L_pred1 = L_pred1 * (224 / 64)
    X_pred1 = image_to_Vggencoder(croped_image1.cuda())
    Id_l = criterion1(X_target1, X_pred1)
    Landmark_l = criterion2(L_target1, L_pred1)
    lagrangian = Id_l - lamb * (5 - Landmark_l)

    lamb.retain_grad()
    latents_to_be_optimized_cpoy.retain_grad()
    lagrangian.backward()


    latents_to_be_optimized_cpoy = latents_to_be_optimized_cpoy - 0.01 * latents_to_be_optimized_cpoy.grad
    lamb = lamb + lamb.grad

    lamb_list.append(lamb)
    # latents_to_be_optimized.grad.zero_()
    # lamb.grad.zero_()

    if lamb < 0:
        lamb = 0

plt.plot(lamb_list, color="red", linestyle = '-', label = 'lambda')
plt.title('lambda')
plot_path = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/Non_ID_result/lambda.jpg'
plt.savefig(plot_path)

lamb = lamb.item()
print('lambda= ' +str(lamb) )

for step in progress_bar:
    count += 1

    optimizer.zero_grad()

    # image_np = generated_image.detach().cpu().numpy()
    # image_np = image_np.squeeze()
    # image_np = image_np.transpose(1, 2, 0)
    # image_aligned = align_image_fromStylegan_to_vgg(image_np) #(224, 224, 3)
    # image_aligned = np.round(image_aligned)
    # #imge_to_tensor
    # image_fed_inLandmarkregressor = image_aligned.transpose(2, 0, 1)
    # image_fed_inLandmarkregressor = image_fed_inLandmarkregressor[np.newaxis, :]
    #
    #
    # image_fed_inLandmarkregressor = np.uint8(image_fed_inLandmarkregressor)
    # image_fed_inLandmarkregressor = torch.tensor(image_fed_inLandmarkregressor) # ([1, 3, 224, 224])
    generated_image = input_image_into_StyleGAN(latents_to_be_optimized)  # [1, 3, 1024, 1024] [0-255]

    ###################################################################
    tform = torch.tensor([
        [3.11314244e+00, -1.35652176e-02, 1.64518438e+02],
         [1.35652176e-02, 3.11314244e+00, 2.11537025e+02]
], dtype=torch.float)

    ###################################################################
    # image_np = generated_image.detach().cpu().numpy()
    # image_np = image_np.squeeze()
    # image_np = image_np.transpose(1, 2, 0)
    # image_aligned = align_image_fromStylegan_to_vgg(image_np) #(224, 224, 3)
    croped_image = crop_image_forVGGFACE2(generated_image) #input image: [1,3,1024,1024] output: [1,3,224,224]

    # import matplotlib.pyplot as plt
    # gi1 = croped_image.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    # gi1 = np.uint8(gi1)
    # plt.imshow(gi1)
    # plt.show()

######################


    L_pred = feed_into_Image_to_landmarksRegressor(croped_image) #first resize input image from 224 to 64
    L_pred = L_pred * (224 / 64)

    X_pred = image_to_Vggencoder(croped_image.cuda())

    Id_loss = criterion1(X_target.double(), X_pred.double())
    Id_loss.register_hook(print)
    Landmark_loss = criterion2(L_target.double(), L_pred.double())
    b = args.weight_landmarkloss
    #loss = b*Id_loss+ Landmark_loss
    loss = Id_loss + lamb*Landmark_loss
    loss.backward(retain_graph=True)
#################################################################

#################################################################
    # L_pred = feed_into_Image_to_landmarksRegressor(croped_image)  # first resize input image from 224 to 64
    # L_pred = L_pred * (224 / 64)
    #
    # X_pred = image_to_Vggencoder(croped_image.cuda())
    #
    # Id_loss = criterion1(X_target.double(), X_pred.double())
    # Id_loss.backward()
    # Id_loss_grad = latents_to_be_optimized.grad
    #
    # Landmark_loss = criterion2(L_target.double(), L_pred.double()).requires_grad_()
    # grad_list = []
    #
    #
    #
    #
    #
    # Landmark_loss.register_hook(hook_fn)
    # grads = []
    #
    #
    #
    # def lagrangian(θ, λ, ε):
    #
    #     damp = 10 *Landmark_loss_grad
    #
    #     return Id_loss(θ) - (λ-damp) * (ε - Landmark_loss(θ))
    #
    # ε = 0.7
    # λ = 0.0
    # θ = 0
    #
    # #Id_loss_grad = Variable(Id_loss.grad.data.clone(), requires_grad=False)
    #
    #
    # Landmark_loss_grad = Landmark_loss.grad
    # for gradient_step in range(200):
    #     gradient_θ = Id_loss_grad
    #     gradient_λ = Landmark_loss_grad
    #     θ = θ - 0.02 * gradient_θ
    #     λ = λ + gradient_λ
    #     if λ < 0:
    #         λ = 0
    #
    # i =1
#################################################################
    # #compute scale for the loss
    # L_pred = feed_into_Image_to_landmarksRegressor(croped_image)  # first resize input image from 224 to 64
    # L_pred = L_pred * (224 / 64)
    # X_pred = image_to_Vggencoder(croped_image.cuda())
    #
    # Id_loss = criterion1(X_target, X_pred)
    # Landmark_loss = criterion2(L_target, L_pred)
    #
    # grads = {}
    # scale = {}
    # grads[0] = []
    # grads[1] = []
    # for param in synthesizer.parameters() :
    #     if param.grad is not None:
    #         grads[0].append(Variable(param.grad.data.clone(), requires_grad=False))
    #
    # for param in  landmark_regressor.parameters():
    #     if param.grad is not None:
    #         grads[1].append(Variable(param.grad.data.clone(), requires_grad=False))
    #
    # # Frank-Wolfe iteration to compute scales.
    # try:
    #     sol, min_norm = MinNormSolver.find_min_norm_element([grads[i] for i in range(2)])
    # except Exception as ex:
    #     print(ex)
    #     raise ex
    #
    # scale[0] = float(sol[0])
    # scale[1] = float(sol[1])
    #
    # # Scaled back-propagation
    # optimizer.zero_grad()
    #
    # L_pred = feed_into_Image_to_landmarksRegressor(croped_image)  # first resize input image from 224 to 64
    # L_pred = L_pred * (224 / 64)
    # X_pred = image_to_Vggencoder(croped_image.cuda())
    # Id_loss = criterion1(X_target, X_pred)
    # Landmark_loss = criterion2(L_target, L_pred)
    #
    # loss = scale[0] * Id_loss +  scale[1] * Landmark_loss
    # loss.backward()
    # optimizer.step()
    #

    loss = loss.item()
    Id_loss = Id_loss.item()
    Landmark_loss = Landmark_loss.item()
    optimizer.step()

    progress_bar.set_description("Step: {}, Loss: {}, Id_loss: {}, Landmark_loss: {}".format(step, loss,Id_loss,Landmark_loss ))

    loss_plot.append(loss)
    Landmark_loss_plot.append(Landmark_loss)
    Id_loss_plot.append(Id_loss)
    # drawLandmarkPoint(croped_image,L_target, L_pred)
    # drawLandmarkPoint(reference_image,L_target, L_pred)

    # Channel, width, height -> width, height, channel, then RGB to BGR
    image = generated_image.detach().cpu().numpy()[0]
    #image = image*256
    image = np.transpose(image, (1, 2, 0))
    image = image[:, :, ::-1]

    video.write(np.uint8(image))

    optimized_dlatents = latents_to_be_optimized.detach().cpu().numpy()
    np.save(args.dlatent_path, optimized_dlatents)



    if count == int(args.iterations):
        #str1 = "/scratch/staff/ml1652/StyleGAN_Reconstuction_server/croped_img/"

        if args.Inserver == False:
            str1 = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/Non_ID_result/last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+'_'
            str2 = (args.image_path).split('\\')[-1]
        else:
            str1 = '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/learningrate_Test/last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+'_'
            str2 = (args.image_path).split('/')[-1]
        last_image_save_path = str1+str2
        cv2.imwrite(last_image_save_path, np.uint8(image))
        drawLandmarkPoint(croped_image, L_target, L_pred)
        draw_targetLandmarkPoint(reference_image, L_target)

video.release()
subdiagram = 3
plt.subplot(subdiagram, 1, 1)
plt.plot(loss_plot, color="red", linestyle = '-', label = 'loss')
plt.title('total loss')
plt.subplot(subdiagram, 1, 2)
plt.plot(Id_loss_plot, color="blue", linestyle = '-', label = 'Id_L')
plt.title('Id_loss')
plt.subplot(subdiagram, 1, 3)
plt.plot(Landmark_loss_plot, color="green", linestyle = '-', label = 'landmark_L')
plt.title('landmark_loss')

if args.Inserver == False:
    str1 = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/Non_ID_result/plotloss__lr=' + str(
        args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+'_'
    str2 = (args.image_path).split('\\')[-1]
else:
    str1 = '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/learningrate_Test/plotloss__lr=' + str(
        args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations) + '_'
    str2 = (args.image_path).split('/')[-1]
plot_path = str1+str2
plt.savefig(plot_path)



#reference  = np.array(reference)
#
# a = generated_image.detach().cpu().numpy()
# a = a.squeeze()
# a = a.transpose(1, 2, 0)
# cv2.imshow("preview", a)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import kornia
# output = kornia.warp_affine(generated_image, tform.unsqueeze(0).cuda(), dsize=(224, 224))
def param2theta(param, w, h):
    param = np.linalg.inv(param)
    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * h / w
    theta[0, 2] = param[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = param[1, 0] * w / h
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
    return theta


def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv(W, H):
    """N that maps from normalized to unnormalized coordinates"""
    # TODO: do this analytically maybe?
    N = get_N(W, H)
    return np.linalg.inv(N)


def cvt_MToTheta(M, w, h):
    """convert affine warp matrix `M` compatible with `opencv.warpAffine` to `theta` matrix
    compatible with `torch.F.affine_grid`

    Parameters
    ----------
    M : np.ndarray
        affine warp matrix shaped [2, 3]
    w : int
        width of image
    h : int
        height of image

    Returns
    -------
    np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    """
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]


def cv_to_pytorch(M, w, h):
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = M[0, 0]
    N[0, 1] = M[0, 1]
    N[1, 1] = M[1, 1]
    N[1, 0] = M[1, 0]
    N[0, -1] = (M[0, -1] - ((1 - M[0, 0]) * (w / 2.0) - M[0, 1] * (h / 2.0))) / (w / 2.0)
    N[1, -1] = (M[1, -1] - ((M[0, 1]) * (w / 2.0) + (1 - M[0, 0]) * (h / 2.0))) / (h / 2.0)
    N[-1, -1] = 1.0
    N = np.linalg.inv(N)
    return N[:2, :]

# tform_pytorch = cvt_MToTheta(tform, 1024, 1024)
# tform_pytorch = cv_to_pytorch(tform, 1024, 1024)

# tform_pytorch = torch.tensor(tform_pytorch)

# grid = F.affine_grid(tform_pytorch.unsqueeze(0),(1,3,224,224),align_corners = True)
# grid = grid.type(torch.FloatTensor).cuda()
# output = F.grid_sample(generated_image, grid,mode="bilinear", padding_mode="border",align_corners=True)
# import matplotlib.pyplot as plt
# gi1 = output.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
# gi1 = np.uint8(gi1)
# plt.imshow(gi1)
# plt.show()


#https://blog.csdn.net/Eddy_Wu23/article/details/108797023
#https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/13
#https://github.com/wuneng/WarpAffine2GridSample

