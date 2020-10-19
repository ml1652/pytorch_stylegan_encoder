from sklearn.utils import shuffle
from torch.utils.data import SubsetRandomSampler

from models.image_to_latent import CelebaRegressor, VGGLatentDataset
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
from tensorboardX import SummaryWriter, writer
import os
from utilities.pytorchtools import EarlyStopping
import pandas as pd

def generate_vgg_descriptors(filenames):
    vgg_face_dag = resnet50_scratch_dag(
        r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\resnet50_scratch_dag.pth').cuda().eval()
    descriptors = []
    vgg_processing = VGGFaceProcessing()

    filenames = filenames[0:]

    for image_file in filenames:
        # print(image_file)
        image = load_images([image_file])

        image = torch.from_numpy(image).cuda()
        image = vgg_processing(image)  # vgg16: the vlaue between -2.04 - 2.54,dim = [1,3,256,256]
        feature = vgg_face_dag(image).cpu().detach().numpy()

        descriptors.append(feature)  # descriptor[128, 28, 28] pool5_7x7_s1:[2048,1,1]

    # save_path = (image_directory + 'descriptors.npy')
    # np.save(save_path, np.concatenate(descriptors, axis=0))
    return np.concatenate(descriptors, axis=0)


early_stop = False
absolute_label = False
data_augmentation = False
single_yaw = True
epochs = 40
image_directory = r"C:\Users\Mingrui\Desktop\GAN\celeba\img_align_celeba"
label = 'Eyeglasses'
data = pd.read_csv(r'C:\Users\Mingrui\Desktop\celeba\Anno\list_attr_celeba_name+glass+smiling.csv')

filenames = [f"{image_directory}/{x}" for x in data['File_Name']]
# for i, name in enumerate(data['File_Name']):
#     Eyeglasses[name] = data['Eyeglasses'][i]
#     Smiling[name] = data['Smiling'][i]

# generae the vgg descriptor
descriptor_file = image_directory + 'descriptors.npy'
if os.path.isfile(descriptor_file):
    descriptor = np.load(descriptor_file)
else:
    descriptor = generate_vgg_descriptors(filenames)
    np.save(descriptor_file, descriptor)

if label == 'Eyeglasses':
    label_sets = list(data['Eyeglasses'])
elif label == 'Smiling':
    label_sets = list(data['Smiling'])

descriptor_file = image_directory + 'descriptors.npy'
if os.path.isfile(descriptor_file):
    descriptor = np.load(descriptor_file)
else:
    descriptor = generate_vgg_descriptors(filenames)
    np.save(descriptor_file, descriptor)
#shuffle the datasets for random spilting the traning and validation datasets
filenames, label_sets, descriptor = shuffle(filenames, label_sets, descriptor)


image_size = 224
total_dataset_num = len(filenames)
num_validationsets = 1
num_trainsets = total_dataset_num - num_validationsets


train_label = label_sets[0:num_trainsets]
validation_label = label_sets[num_trainsets:]


train_descriptors = descriptor[0:num_trainsets]
validation_descriptors = descriptor[num_trainsets:]

train_dataset = VGGLatentDataset(train_descriptors, train_label)
validation_dataset = VGGLatentDataset(validation_descriptors, validation_label)

train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)

# Instantiate Model
celeba_regressor = CelebaRegressor().cuda()
optimizer = torch.optim.Adam(celeba_regressor.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

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

    celeba_regressor.train()

    running_loss = 0.0
    running_count = 0

    for i, (vgg_descriptors, celeba_label) in enumerate(train_generator, 1):
        optimizer.zero_grad()

        vgg_descriptors, celeba_label = vgg_descriptors.cuda(), celeba_label.cuda()
        pred_label = celeba_regressor(vgg_descriptors)
        pred_label = pred_label.squeeze()
        #pred_label = pred_label.type_as(celeba_label)
        #loss = criterion(pred_label.double(), celeba_label.double())
        loss = criterion(pred_label.double(), celeba_label.double())

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        running_loss_in_epoch.append(loss.item())
        running_count += 1
        if i % 50 == 0:
            training_loss_iteration = loss.item()
            # Traning_loss.append(traning_loss)
            training_interation = (epoch + 1) * num_trainsets + i

    # plot loss vs epoch
    traning_loss_epoch = running_loss / running_count
    Training_loss_epoch_list.append(traning_loss_epoch)

    validation_loss_count = 0
    validation_loss_sum = 0.0
    celeba_regressor.eval()
    for i, (vgg_descriptors, celeba_label) in enumerate(validation_generator, 1):
        with torch.no_grad():
            vgg_descriptors, celeba_label = vgg_descriptors.cuda(), celeba_label.cuda()
            pred_label = celeba_regressor(vgg_descriptors)
            loss = criterion(pred_label.double(), celeba_label.double())
            validation_loss_iteration = loss.item()
            validation_loss_sum += loss.item()
            validation_loss_count += 1
            validation_loss_in_epoch.append(loss.item())
            # Validation_loss.append(validation_loss)
            validation_interation = (epoch + 1) * num_validationsets + i

    # plot loss vs epoch
    validation_loss_epoch = validation_loss_sum / validation_loss_count
    Validation_loss_epoch_list.append(validation_loss_epoch)

    # progress_bar.set_description(
    #     "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, 0))
    progress_bar.set_description(
        "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(epoch, traning_loss_epoch, validation_loss_epoch))

    # early_stopping
    if early_stop == True:
        early_stopping(validation_loss_epoch, celeba_regressor)

        if early_stopping.early_stop:
            print("Early stopping")
            break



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
torch.save(celeba_regressor.state_dict(),
           f"C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/celebaregressor.pt")
