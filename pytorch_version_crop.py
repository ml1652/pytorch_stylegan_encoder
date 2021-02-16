import dlib
from numpy import linalg as LA
import cv2
import numpy as np
from torch.nn import functional as F
import torch

def generate_landmark(img, draw_landmark=False):
    # location of the model (path of the model).
    Model_PATH = r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\models\shape_predictor_68_face_landmarks.dat"

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


def align_image_fromStylegan_to_vgg(img, np_img):
    img_landmark = generate_landmark(np_img)

    n_landmark = 68
    standard_landmark_file = 'C:/Users/Mingrui/Desktop/Github/HD-CelebA-Cropper/data/standard_landmark_68pts.txt'
    standard_landmark = np.genfromtxt(standard_landmark_file, dtype=np.float).reshape(n_landmark, 2)
    move_w = 0
    move_h = 0.25
    standard_landmark[:, 0] += move_w
    standard_landmark[:, 1] += move_h
    celeba_standard_landmark = np.loadtxt(r"C:\Users\Mingrui\Desktop\celeba\Anno\standard_landmark_celeba.txt",
                                          delimiter=',').reshape(-1, 5, 2)
    celeba_landmark = np.genfromtxt(r"C:\Users\Mingrui\Desktop\celeba\Anno\list_landmarks_celeba.txt", dtype=np.float,
                                    usecols=range(1, 5 * 2 + 1), skip_header=2).reshape(-1, 5, 2)

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

    #img_crop = img_crop[...,::-1]
    return img_crop

def align_crop_opencv(img,
                      np_img,
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
        tform = cv2.estimateAffinePartial2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]


    # calcaute the scale of tform
    m1 = np.mat('0;0;1')
    m2 = np.mat('1;0;1')
    p1 = tform.dot(m1)
    p2 = tform.dot(m2)
    scale = LA.norm(p2 - p1)  # defualt is Frobenius norm






    # change the translations part of the transformation matrix for downwarding vertically
    tform[1][2] = tform[1][2] + 20 * scale

    #numpy to tensor
    tform = torch.tensor(tform).cuda()


    grid = F.affine_grid(tform, img.unsqueeze(0).size())
    output = F.grid_sample(img.unsqueeze(0), grid)

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



if __name__ == '__main__':
    from torchvision import transforms
    from PIL import Image
    import matplotlib.pyplot as plt

    img_path = r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(224,224)_move(0.250,0.000)_face_factor(0.800)_jpg\data\000001.jpg"
    img_torch = transforms.ToTensor()(Image.open(img_path))

    from torch.nn import functional as F

    theta = torch.tensor([
        [1, 0, -0.2],
        [0, 1, -0.4]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    plt.imshow(new_img_torch.numpy().transpose(1, 2, 0))
    plt.show()


