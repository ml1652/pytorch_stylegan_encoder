# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:44:28 2020

@author: Mingrui
"""

from models.image_to_latent import VGGToLatent, VGGLatentDataset
from torchvision import transforms
import torch
from glob import glob
from tqdm import tqdm_notebook as tqdm
#from tqdm import notebook.tqdm as tqdm
#from tqdm.notebook import tqdm
from tqdm import tqdm
import numpy as np
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
import torch.utils.data as Data
import matplotlib.pyplot as plt
from utilities.images import load_images, images_to_video, save_image
from tensorboardX import SummaryWriter

writer = SummaryWriter('vgg_latent_model')

image_size = 224
num_trainsets = 48000
directory = "C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/InterFaceGAN/dataset_directory/"
filenames = sorted(glob(directory + "*.jpg"))

#train_filenames = filenames[0:num_trainsets]
#validation_filenames = filenames[num_trainsets:]
#validation_filenames = filenames[num_trainsets:]
dlatents = np.load(directory + "WP.npy")

train_dlatents = dlatents[0:num_trainsets]
validation_dlatents = dlatents[num_trainsets:]

# ##################################################
# vgg_face_dag = resnet50_scratch_dag(r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\resnet50_scratch_dag.pth').cuda().eval()
# descriptors = []
# vgg_processing = VGGFaceProcessing()
#
# filenames = filenames[0:50000]
# for i in filenames:
#     image = load_images([i])
#     image = torch.from_numpy(image).cuda()
#     image = vgg_processing(image)  # vgg16: the vlaue between -2.04 - 2.54,dim = [1,3,256,256]
#     feature = vgg_face_dag(image).cpu().detach().numpy()
#
#     descriptors.append(feature) #descriptor[128, 28, 28] pool5_7x7_s1:[2048,1,1]
#
# save_path = (directory + 'descriptors.npy')
# np.save(save_path, np.concatenate(descriptors, axis=0))
# ######################################################


descriptor = np.load(directory + "descriptors.npy")
#descriptor = Data.TensorDataset(descriptor)

train_descriptors = descriptor[0:num_trainsets]
validation_descriptors = descriptor[num_trainsets:]

train_dataset = VGGLatentDataset(train_descriptors, train_dlatents)
validation_dataset = VGGLatentDataset(validation_descriptors, validation_dlatents)

train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=32)
validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=32)

# Instantiate Model
vgg_to_latent = VGGToLatent().cuda()
optimizer = torch.optim.Adam(vgg_to_latent.parameters(),lr=0.001)
#criterion = LogCoshLoss()
criterion = torch.nn.MSELoss()

# Train Model
epochs = 20
validation_loss = 0.0

progress_bar = tqdm(range(epochs))
Loss_list = []
Traning_loss =[]
Validation_loss = []
for epoch in progress_bar:
    running_loss = 0.0

    vgg_to_latent.train()
    #for vgg_descriptors, latents in train_generator:
    for i, (vgg_descriptors, latents) in enumerate(train_generator, 1):
        optimizer.zero_grad()

        vgg_descriptors, latents = vgg_descriptors.cuda(), latents.cuda()
        pred_latents = vgg_to_latent(vgg_descriptors)
        loss = criterion(pred_latents, latents)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_description(
            "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))

    traning_loss = running_loss / i
    Traning_loss.append(traning_loss)
    writer.add_scalar('Test/traning_loss', traning_loss, epoch)
    validation_loss = 0.0

    vgg_to_latent.eval()
    for i, (vgg_descriptors, latents) in enumerate(validation_generator, 1):
        with torch.no_grad():
            vgg_descriptors, latents = vgg_descriptors.cuda(), latents.cuda()
            pred_latents = vgg_to_latent(vgg_descriptors)
            loss = criterion(pred_latents, latents)

            validation_loss += loss.item()


    validation_loss = validation_loss / i
    Validation_loss.append(validation_loss)
    writer.add_scalar('Test/validation_loss', validation_loss, epoch)
    progress_bar.set_description(
        "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))

writer.close()
#plot loss

y1 = Traning_loss
y2 = Validation_loss
plt.subplot(2, 1, 1)
plt.plot( y1, 'o-')
plt.title('training loss vs. epoches')
plt.ylabel('training loss')
plt.subplot(2, 1, 2)
plt.plot( y2, '.-')
plt.xlabel('validation loss vs. epoches')
plt.ylabel('validation loss')
plt.savefig("accuracy_loss.jpg")
plt.show()


# save moodel
torch.save(vgg_to_latent.state_dict(), "C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/vgg_to_latent_Z_01.pt")

# load Model
'''
vgg_to_latent = VGGToLatent().cuda()
vgg_to_latent.load_state_dict(torch.load("vgg_to_latent.pt"))
vgg_to_latent.eval()
directory = "C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/InterFaceGAN/dataset_directory/"

descriptor = np.load(directory + "descriptors.npy")
latents_to_be_optimized = vgg_to_latent(descriptor[0])
latents_to_be_optimized = latents_to_be_optimized.detach().cuda().requires_grad_(True)
'''