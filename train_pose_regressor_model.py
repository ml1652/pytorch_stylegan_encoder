from sklearn.utils import shuffle
from torch.utils.data import SubsetRandomSampler

from models.image_to_latent import PoseRegressor, VGGLatentDataset
from torchvision import transforms
import torch
from glob import glob
from tqdm import tqdm_notebook as tqdm
# from tqdm import notebook.tqdm as tqdm
# from tqdm.notebook import tqdm
from tqdm import tqdm
import numpy as np
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
import torch.utils.data as Data
import matplotlib.pyplot as plt
from utilities.images import load_images, images_to_video, save_image
from tensorboardX import SummaryWriter
import os
from utilities.pytorchtools import EarlyStopping
from natsort import natsorted
import pandas as pd
early_stop = False
absolute_label = False
data_augmentation = False
single_yaw = False
epochs = 5

from PIL import Image

def generate_vgg_descriptors(filenames):
    vgg_face_dag = resnet50_scratch_dag(
        r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\resnet50_scratch_dag.pth').cuda().eval()
    descriptors = []
    vgg_processing = VGGFaceProcessing()

    filenames = filenames[0:]

    for image_file in filenames:
        print(image_file)
        image_file = image_directory + image_file
        image = load_images([image_file])


        image = torch.from_numpy(image).cuda()
        image = vgg_processing(image)  # vgg16: the vlaue between -2.04 - 2.54,dim = [1,3,256,256]
        feature = vgg_face_dag(image).cpu().detach().numpy()

        descriptors.append(feature)  # descriptor[128, 28, 28] pool5_7x7_s1:[2048,1,1]

    # save_path = (image_directory + 'descriptors.npy')
    # np.save(save_path, np.concatenate(descriptors, axis=0))
    return np.concatenate(descriptors, axis=0)


writer = SummaryWriter('tesnsor_board_pose_regressor_model')

if data_augmentation:
    image_directory = "C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/AFLW_vggface2_tightcrop_flip/"
    # pose = [roll, pitch, yaw]
    poses = np.load("D:/AFLW/numpylist2/pose_data_with_name_flip.npz", allow_pickle=True)
else:
    image_directory = r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(512,512)_move(0.250,0.000)_face_factor(0.500)_jpg\data\\"
    #poses = np.load("D:/AFLW/numpylist2/pose_data_with_name.npz", allow_pickle=True)
    poses = pd.read_csv(r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(512,512)_move(0.250,0.000)_face_factor(0.500)_jpg\out_pose_label.txt",sep=' ',
            header=None)

    #poses = pd.read_table(r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(512,512)_move(0.250,0.000)_face_factor(0.500)_jpg\out_pose_label.txt",header=None)

# # [filenames, pose_sets] = poses
# filenames = poses['path']
# pose_sets = np.stack(poses['pose'])

data = poses.to_numpy()
filenames = data[:, 0].tolist()
pose_sets = data[:, 1:].tolist()
pose_sets = np.stack(pose_sets)
# generae the vgg descriptor
#descriptor_file = f"D:/AFLW/numpylist/pose_data_flip={data_augmentation}.npy"
#descriptor_file = image_directory+f"/pose_data_absolute_label={absolute_label}_singlegangel={single_yaw}_flip={data_augmentation}.npy"
descriptor_file = image_directory + 'descriptors.npy'
if os.path.isfile(descriptor_file):
    descriptor = np.load(descriptor_file)
else:
    descriptor = generate_vgg_descriptors(filenames)
    np.save(descriptor_file, descriptor)
#shuffle the datasets for random spilting the traning and validation datasets
filenames, pose_sets, descriptor = shuffle(filenames, pose_sets, descriptor)

# np.savez('testsetsetset.npz', names=filenames, poses=pose_sets, descriptor=descriptor)
# v = np.load('testsetsetset.npz')
# filenames=v['names']
# pose_sets=v['poses']
# descriptor=v['descriptor']

image_size = 224
total_dataset_num = len(filenames)
num_validationsets = 10000
num_trainsets = total_dataset_num - num_validationsets

# setting to learn the single yaw angle alone
output_count = 3  # the size of output of the pose regressor
if single_yaw:
    output_count = 1
    pose_sets = np.atleast_2d(pose_sets[:, -1]).transpose()
    train_pose = pose_sets[0:num_trainsets]
    validation_pose = pose_sets[num_trainsets:]
else:
    train_pose = pose_sets[0:num_trainsets]
    validation_pose = pose_sets[num_trainsets:]

if absolute_label:
    train_pose = abs(train_pose)
    validation_pose = abs(validation_pose)

train_descriptors = descriptor[0:num_trainsets]
validation_descriptors = descriptor[num_trainsets:]

# train_dataset = VGGLatentDataset(train_descriptors, pose_sets)
# validation_dataset = VGGLatentDataset(validation_descriptors, pose_sets)
train_dataset = VGGLatentDataset(train_descriptors, train_pose)
validation_dataset = VGGLatentDataset(validation_descriptors, validation_pose)

train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)

# Instantiate Model
pose_regressor = PoseRegressor(output_count).cuda()
optimizer = torch.optim.Adam(pose_regressor.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Train Model
progress_bar = tqdm(range(epochs))
Loss_list = []
Training_loss_epoch_list = []
Validation_loss_epoch_list = []

all_training_loss = []
all_validation_loss = []

# initialize the early_stopping object
if early_stop == True:
    patience = 40  # How long to wait after last time validation loss improved.
    early_stopping = EarlyStopping(patience=patience, verbose=True)

for epoch in progress_bar:
    running_loss_in_epoch = []
    validation_loss_in_epoch = []
    all_training_loss.append(running_loss_in_epoch)
    all_validation_loss.append(validation_loss_in_epoch)

    pose_regressor.train()

    running_loss = 0.0
    running_count = 0

    for i, (vgg_descriptors, pose) in enumerate(train_generator, 1):
        optimizer.zero_grad()

        vgg_descriptors, pose = vgg_descriptors.cuda(), pose.cuda()
        pred_pose = pose_regressor(vgg_descriptors)
        loss = criterion(pred_pose.double(), pose.double())
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        running_loss_in_epoch.append(loss.item())
        running_count += 1
        if i % 50 == 0:
            training_loss_iteration = loss.item()
            # Traning_loss.append(traning_loss)
            training_interation = (epoch + 1) * num_trainsets + i
            writer.add_scalar('Intreation/traning_loss', training_loss_iteration, training_interation)

    # plot loss vs epoch
    traning_loss_epoch = running_loss / running_count
    Training_loss_epoch_list.append(traning_loss_epoch)
    writer.add_scalar('Epoch/traning_loss', traning_loss_epoch, epoch)

    validation_loss_count = 0
    validation_loss_sum = 0.0
    pose_regressor.eval()
    for i, (vgg_descriptors, pose) in enumerate(validation_generator, 1):
        with torch.no_grad():
            vgg_descriptors, pose = vgg_descriptors.cuda(), pose.cuda()
            pred_pose = pose_regressor(vgg_descriptors)
            loss = criterion(pred_pose.double(), pose)
            validation_loss_iteration = loss.item()
            validation_loss_sum += loss.item()
            validation_loss_count += 1
            validation_loss_in_epoch.append(loss.item())
            # Validation_loss.append(validation_loss)
            validation_interation = (epoch + 1) * num_validationsets + i
            writer.add_scalar('Intreation/validation_loss', validation_loss_iteration, validation_interation)

    # plot loss vs epoch
    validation_loss_epoch = validation_loss_sum / validation_loss_count
    Validation_loss_epoch_list.append(validation_loss_epoch)
    writer.add_scalar('Epoch/validation_loss', validation_loss_epoch, epoch)

    # progress_bar.set_description(
    #     "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, 0))
    progress_bar.set_description(
        "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(epoch, traning_loss_epoch, validation_loss_epoch))

    # early_stopping
    if early_stop == True:
        early_stopping(validation_loss_epoch, pose_regressor)

        if early_stopping.early_stop:
            print("Early stopping")
            break

writer.close()


# plot the loss at certain intervals
def every(count):
    i = 0

    def filter_fn(item):
        nonlocal i
        nonlocal count
        result = i % count == 0
        i += 1
        return result

    return filter_fn


def flatten(items):
    result = []
    for item in items:
        result += item
    return result
    # return [item for sublist in items for item in sublist]


# plot loss_iteration
y1 = list(filter(every(50), flatten(all_training_loss)))
y2 = list(filter(every(5), flatten(all_validation_loss)))
plt.subplot(2, 1, 1)
plt.plot(y1, 'o-')
plt.title('training loss vs. iteration')
plt.ylabel('training loss')
plt.subplot(2, 1, 2)
plt.plot(y2, '.-')
plt.xlabel('validation loss vs. iteration')
plt.ylabel('validation loss')
plt.savefig("accuracy_loss_iteration.jpg")
plt.show()

# plot loss_epoch
y1 = Training_loss_epoch_list
y2 = Validation_loss_epoch_list

plt.subplot(2, 1, 1)
plt.plot(y1, 'o-')
plt.title('training loss vs. epoch')
plt.ylabel('training loss')
plt.subplot(2, 1, 2)
plt.plot(y2, '.-')
plt.xlabel('validation loss vs. epoch')
plt.ylabel('validation loss')
plt.savefig("accuracy_loss_epoch.jpg")
plt.show()

# save moodel
torch.save(pose_regressor.state_dict(),
           f"C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/pose_regressor_absolutelabel={absolute_label}_singlegangel={single_yaw}_flip={data_augmentation}.pt")

# load Model
# tensorboard --logdir "C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/tesnsor_board_pose_regressor_model/"

'''
vgg_to_latent = VGGToLatent().cuda()
vgg_to_latent.load_state_dict(torch.load("vgg_to_latent.pt"))
vgg_to_latent.eval()
directory = "C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/InterFaceGAN/dataset_directory/"

descriptor = np.load(directory + "descriptors.npy")
latents_to_be_optimized = vgg_to_latent(descriptor[0])
latents_to_be_optimized = latents_to_be_optimized.detach().cuda().requires_grad_(True)
'''
