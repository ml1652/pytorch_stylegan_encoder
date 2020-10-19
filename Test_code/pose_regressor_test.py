from models.image_to_latent import PoseRegressor, VGGLatentDataset
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
import torch
from utilities.images import load_images, images_to_video, save_image
from glob import glob
import numpy as np
image_directory = 'C:/Users/Mingrui/Desktop/datasets/MTCNN_processed/'

filenames = glob(image_directory + "*.jpg")
pose_record = []
vgg_processing = VGGFaceProcessing()
vgg_face_dag = resnet50_scratch_dag(r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\resnet50_scratch_dag.pth').cuda().eval()

output_count = 3

pose_regressor = PoseRegressor(output_count).cuda()
pose_regressor.load_state_dict(torch.load(r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\pose_regressor_absolutelabel=False_singlegangel=False_flip=False.pt"))
pose_regressor.eval()
for i in filenames:
    image = load_images([i])
    image = torch.from_numpy(image).cuda()
    image = vgg_processing(image)
    vgg_descriptors = vgg_face_dag(image).cuda()
    pred_pose = pose_regressor(vgg_descriptors)
    pred_pose = str(pred_pose)
    pose_record.append( i + pred_pose)

np.save(r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\pose_test_record.npy", pose_record)

