from torch.utils.data import SubsetRandomSampler
from models.image_to_latent import CelebaRegressor, VGGLatentDataset ,LandMarksRegressor
import torch
from tqdm import tqdm
import numpy as np
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
import matplotlib.pyplot as plt
from utilities.images import load_images, images_to_video, save_image
import os
from utilities.pytorchtools import EarlyStopping
import pandas as pd

epochs = 20
absolute_label = False
data_augmentation = False
single_yaw = True

image_directory = 'C:/Users/Mingrui/Desktop/Github/HD-CelebA-Cropper/data/aligned/align_size(224,224)_move(0.250,0.000)_face_factor(0.800)_jpg/data'
descriptor_file = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/Muhammad_faceenbeding_skipped_img_=6580.npy'
label = 'Smiling'
#label = 'landmark'
landmarks_num = 68

descriptor = np.load(descriptor_file, allow_pickle=True)
descriptor = descriptor.item()
img_names = []
for key in descriptor.keys():
    img_names.append(key)
filenames = [f"{image_directory}/{x}" for x in img_names]
#number of filenames : 196019

if label == 'landmarks':
    if landmarks_num == 68:
        data = pd.read_csv(
            r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(224,224)_move(0.250,0.000)_face_factor(0.800)_jpg\tformed_landmark_68point.txt",
            sep=' ',
            header=None,
        )

    elif landmarks_num == 5:
        data = pd.read_csv(
            r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(224,224)_move(0.250,0.000)_face_factor(0.800)_jpg\tformed_landmark_5point.txt",
            sep=' ',
            header=None,
        )
    data = data.to_numpy()
    names = data[:, 0].tolist()

    for i in names:
        if i not in img_names:
            data.drop([i])
    np.save(r"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(224,224)_move(0.250,0.000)_face_factor(0.800)_jpg\Muhammad_tformed_landmark_5point.txt", data)
else:
    # data = pd.read_csv(r'C:\Users\Mingrui\Desktop\celeba\Anno\list_attr_celeba_name+glass+smiling.csv')
    # names = data['File_Name']
    data = pd.read_table(r"C:\Users\Mingrui\WebstormProjects\db_filter\processed.txt", skiprows=1, header=0, sep=r"\s+")
    names = data.index

    # i = 0
    # to_drop = []
    # for (name, row) in data.iterrows():
    #     if name not in img_names:
    #         to_drop.append(name)
    #         # print('%s: %s' % (name, row))
    #         # i += 1
    #         # if i > 10: break
    # data.drop(to_drop)
    # for i in names:
    #     if i not in img_names:
    #         data.drop([i])
    # np.save('C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/Muhammada_descriptors_afterdrop.npy',data)

filenames = [f"{image_directory}/{x}" for x in names]

if label == 'landmarks':
    label_sets = np.stack(data[:,1:].astype('float64'))
else:
    #label_sets = list(data['Eyeglasses'])
    label_sets = data[label]
    label_sets = (label_sets + 1) / 2
    label_sets = label_sets.astype(int)

total_dataset_num = len(filenames)
num_validationsets = 10000
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
if label == "landmarks":
    celeba_regressor = LandMarksRegressor(landmarks_num).cuda()
    criterion = torch.nn.MSELoss()
    #criterion = WingLoss()
else:
    celeba_regressor = CelebaRegressor().cuda()
    criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(celeba_regressor.parameters(), lr=0.001)

# Train Model
progress_bar = tqdm(range(epochs))
Loss_list = []
Training_loss_epoch_list = []
Validation_loss_epoch_list = []
Training_correct_percent_list = []
Validation_correct_percent_list = []
all_training_loss = []
all_validation_loss = []

for epoch in progress_bar:
    running_loss_in_epoch = []
    validation_loss_in_epoch = []
    running_correct_precent_in_epoch = []
    validation_correct_precent_in_epoch = []
    all_training_loss.append(running_loss_in_epoch)
    all_validation_loss.append(validation_loss_in_epoch)

    celeba_regressor.train()

    running_loss = 0.0
    running_count = 0

    postive = 0
    negetive = 0

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

        if label != "landmarks":
            #corrct precent
            pred_choice = (pred_label > 0.5).int()
            pred_choice = torch.squeeze(pred_choice)
            correct_sum  = sum(celeba_label == pred_choice).cpu().numpy()
            correct_percent_interation = (correct_sum / len(celeba_label)) * 100
            running_correct_precent_in_epoch.append(correct_percent_interation)

    #plot correct precent vs epoch
    if label != "landmarks":
        correct_percent = sum(running_correct_precent_in_epoch)/running_count
        Training_correct_percent_list.append(correct_percent)


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


            if label != "landmarks":
                # corrct precent
                pred_choice = (pred_label > 0.5).int()
                pred_choice = torch.squeeze(pred_choice)
                correct_sum = sum(celeba_label == pred_choice).cpu().numpy()
                correct_percent_interation = (correct_sum / len(celeba_label)) * 100
                validation_correct_precent_in_epoch.append(correct_percent_interation)

    if label != "landmarks":
        #plot correct precent vs epoch
        correct_percent = sum(validation_correct_precent_in_epoch) / validation_loss_count
        Validation_correct_percent_list.append(correct_percent)
    # plot loss vs epoch
    validation_loss_epoch = validation_loss_sum / validation_loss_count
    Validation_loss_epoch_list.append(validation_loss_epoch)

    # progress_bar.set_description(
    #     "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, 0))
    progress_bar.set_description(
        "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(epoch, traning_loss_epoch, validation_loss_epoch))


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
# y1 = y1[1:]
# y2 = y2[1:]
plt.subplot(2, 1, 1)
plt.yscale("log")
plt.plot(y1, 'o-')
plt.title('training loss vs. iteration')
plt.ylabel('training loss')
plt.subplot(2, 1, 2)
plt.plot(y2, '.-')
plt.xlabel('validation loss vs. iteration')
plt.ylabel('validation loss')
plt.savefig("./diagram/Muhammad_accuracy_loss_iteration.jpg")
plt.show()

# plot loss_epoch
subdiagram = 2
y1 = Training_loss_epoch_list
y2 = Validation_loss_epoch_list
y1 = y1[1:]
y2 = y2[1:]
if label != "landmarks":
    subdiagram = 4
    y3 = Training_correct_percent_list
    y4 = Validation_correct_percent_list
    y3 = y3[1:]
    y4 = y4[1:]

#plt.subplot(2, 1, 1)
plt.subplot(subdiagram, 1, 1)
plt.plot(y1, 'o-')
plt.title('training loss vs. epoch')
plt.ylabel('training loss')
plt.yscale("log")
plt.subplot(subdiagram, 1, 2)
plt.plot(y2, '.-')
plt.xlabel('validation loss vs. epoch')
plt.ylabel('validation loss')

if label != "landmarks":
    plt.subplot(subdiagram, 1, 3)
    plt.plot(y3, '.-')
    plt.xlabel('Training_correct_percent vs. epoch')
    plt.ylabel('Training_correct')
    plt.subplot(subdiagram, 1, 4)
    plt.plot(y4, '.-')
    plt.xlabel('Validation_correct_percent vs. epoch')
    plt.ylabel('validation_correct')

plt.savefig("./diagram/Muhammad_accuracy_loss_epoch_" + label+ ".jpg")

plt.show()

# save moodel
if label == 'landmarks':
    torch.save(celeba_regressor.state_dict(),
               f"C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/Trained_model/Muhammad_Celeba_Regressor_" + str(landmarks_num)+ "_landmarks"+ ".pt")
else:
    torch.save(celeba_regressor.state_dict(),
               f"C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/Trained_model/Muhammad_Celeba_Regressor_" +label +".pt")
