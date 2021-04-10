from models.image_to_latent import LandMarksRegressor
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
import torch
from utilities.images import load_images, images_to_video, save_image
from models.image_to_latent import VGGToLatent, VGGLatentDataset,PoseRegressor,VGGToHist
from glob import glob
import numpy as np
import cv2
import os
from functools import partial
from PIL import Image, ImageDraw
import json
import requests

import matplotlib.pyplot as plt

image_directory = 'C:/Users/Mingrui/Desktop/datasets/MTCNN_processed/'
image_directory = 'C:/Users/Mingrui/Desktop/datasets/celeba_imagesample_nocentercrop/'
#image_directory = r"C:\Users\Mingrui\Desktop\datasets\celeba_validsationset_sample\\"
image_directory = r"C:\Users\Mingrui\Desktop\datasets\celeba_validsationset_sample\\"
#image_directory = r"C:\Users\Mingrui\Desktop\datasets\celeba_imagesample_nocentercrop_profile\\"
#image_directory = 'C:/Users/Mingrui/Desktop/datasets/celeba_imagesample/'
image_directory = 'C:/Users/Mingrui/Desktop/datasets/StyleGANimge_corp/webimage_alignmentTest/'
filenames = glob(image_directory + "*.jpg")
hist_record = []
vgg_processing = VGGFaceProcessing()
vgg_face_dag = resnet50_scratch_dag(r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\resnet50_scratch_dag.pth').cuda().eval()


bins_num = 25
sigma_choice = 1.85
vgg_to_hist_regressor= VGGToHist(bins_num).cuda()
vgg_to_hist_regressor.load_state_dict(torch.load(r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\vgg_to_hist_bins=%s.pt" %str(bins_num)))

def plot_hist():
    subdiagram = 3

    plt.subplot(subdiagram, 1,1)
    plt.title('channel r')
    plt.xlabel('bin_num')
    plt.ylabel('pixel_num')
    plt.plot(r_target, color="green", linestyle='-', label ='histc')
    plt.plot(r_pred, color="red", linestyle='-', label='vgg_to_hist')
    plt.plot(r_soft, color="blue", linestyle='-', label='softhist')
    plt.legend()


    plt.subplot(subdiagram, 1, 2)
    plt.title('channel g')
    plt.xlabel('bin_num')
    plt.ylabel('pixel_num')
    plt.plot(g_target, color="green", linestyle='-')
    plt.plot(g_pred, color="red", linestyle='-')
    plt.plot(g_soft, color="blue", linestyle='-')



    plt.subplot(subdiagram, 1, 3)
    plt.title('channel b')
    plt.xlabel('bin_num')
    plt.ylabel('pixel_num')
    plt.plot(b_target, color="green", linestyle='-')
    plt.plot(b_pred, color="red", linestyle='-')
    plt.plot(b_soft, color="blue", linestyle='-')

    path_ = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/diagram/vggtohist_HistcTest_SofthistCompare_sigma=' + str(sigma_choice)
    img_name = filename.split('\\')[-1]
    plot_path = path_ + img_name
    plt.savefig(plot_path)
    plt.close('all')


class SoftHistogram(torch.nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = torch.nn.Parameter(self.centers, requires_grad=False)
        #self.flatten = torch.nn.Flatten()

    def forward(self, x):

        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        x = x.sum(dim=1)
        return x
softhist = SoftHistogram(bins_num, min=0, max=255, sigma=sigma_choice).cuda()
vgg_to_hist_regressor.eval()


for filename in filenames:
    image = load_images([filename])
    image_torch = torch.from_numpy(image).cuda()
    image = vgg_processing(image_torch)
    descriptors = vgg_face_dag(image).cuda()
    pred_hist = vgg_to_hist_regressor(descriptors)

    r = image_torch[:, 0, :]
    g = image_torch[:, 1, :]
    b = image_torch[:, 2, :]


    bins = 25
    x = torch.rand(1, out=None)*255
    #x = torch.ones(1)*255
    #sigma = 0.04
    # delta = 255 / bins
    # sigma = 0.
    # centers = delta * (torch.arange(bins).float() + 0.5)
    # x = torch.unsqueeze(x, 0) - torch.unsqueeze(centers, 1)
    # fucntion1 = torch.sigmoid(0.01 * (x + delta / 2)) - torch.sigmoid(0.01 * (x - delta / 2))
    # fucntion2 = torch.sigmoid(sigma * (x + delta / 2)) - torch.sigmoid(sigma * (x - delta / 2))
    # fucntion3 = torch.sigmoid(0.1 * (x + delta / 2)) - torch.sigmoid(0.1 * (x - delta / 2))
    # fucntion4 = torch.sigmoid(1 * (x + delta / 2)) - torch.sigmoid(1 * (x - delta / 2))
    # #fucntion = fucntion.sum(dim=1)
    # plt.plot(x, fucntion1.detach().cpu().data.numpy(), color="green", linestyle='-', label='sigma = 0.01')
    # plt.plot(x,fucntion2.detach().cpu().data.numpy(), color="yellow", linestyle='-', label ='sigma = 0.04')
    # plt.plot(x, fucntion3.detach().cpu().data.numpy(), color="red", linestyle='-', label='sigma =0.1')
    # plt.plot(x, fucntion4.detach().cpu().data.numpy(), color="blue", linestyle='-', label='sigma =1')
    #
    # plt.xlim(-200, 200)
    # plt.legend()
    #
    # plt.savefig('C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/Non_ID_result/sigma.jpg')
    # plt.show()
    # plt.cla()

    r_target = torch.histc(r.float(), bins_num, min=0, max=255).cpu().detach().numpy()
    g_target = torch.histc(g.float(),bins_num, min=0, max=255).cpu().detach().numpy()
    b_target = torch.histc(b.float(),bins_num, min=0, max=255).cpu().detach().numpy()
    r_flatten = r.flatten()
    g_flatten = g.flatten()
    b_flatten = b.flatten()
    #
    num_pix = image.shape[2] * image.shape[3]

    r_soft = softhist(r_flatten).cpu().detach().numpy()
    g_soft= softhist(g_flatten).cpu().detach().numpy()
    b_soft = softhist(b_flatten).cpu().detach().numpy()
    r_soft = r_soft/ num_pix
    g_soft = g_soft / num_pix
    b_soft = b_soft / num_pix


    r_target = r_target/ num_pix
    g_target = g_target / num_pix
    b_target = b_target / num_pix

    pred_hist = pred_hist.squeeze()


    r_pred = pred_hist[ 0, :].cpu().detach().numpy()
    g_pred = pred_hist[ 1, :].cpu().detach().numpy()
    b_pred = pred_hist[ 2, :].cpu().detach().numpy()

    plot_hist()
    #pred_hist = str(pred_hist)
    #hist_record.append(filename + pred_hist)

