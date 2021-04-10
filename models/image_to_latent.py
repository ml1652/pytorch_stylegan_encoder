import torch
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import torch.nn.functional as F


class ImageToLatent(torch.nn.Module):
    def __init__(self, image_size=256):
        super().__init__()
        
        self.image_size = image_size
        self.activation = torch.nn.ELU()
        
        self.resnet = list(resnet50(pretrained=True).children())[:-2]
        self.resnet = torch.nn.Sequential(*self.resnet)
        self.conv2d = torch.nn.Conv2d(2048, 256, kernel_size=1)
        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(16384, 256)
        self.dense2 = torch.nn.Linear(256, (18 * 512))


    def forward(self, image):
        x = self.resnet(image) #[1, 2048, 8, 8]
        x = self.conv2d(x) #[1, 256, 8, 8]
        x = self.flatten(x) #[1, 16384] 16384 = 256x8x8
        x = self.dense1(x) #[1, 256]
        x = self.dense2(x) #[1, 9216]
        x = x.view((-1, 18, 512)) #[1, 18, 512]


        return x





class VGGToLatent(torch.nn.Module):
    def __init__(self):
        super(VGGToLatent,self).__init__()
        '''
        self.flatten = torch.nn.Flatten()
        #self.dense1 = torch.nn.Linear(100352, 224) # 100352 = 128X28X28
        self.dense1 = torch.nn.Linear(2048, 2048)  # pool layer2048 = 2048X1X1
        #self.dense2 = torch.nn.Linear(224, (18 * 512))
        self.dense2 = torch.nn.Linear(2048,  2048)
        self.dense3 = torch.nn.Linear(2048, 512)
        '''
        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(2048, 2048)  # pool layer2048 = 2048X1X1
        self.dense2 = torch.nn.Linear(2048, 2048)
        #self.dense3 = torch.nn.Linear(2048, 512)
        self.dense3 = torch.nn.Linear(2048, (18*512))

    def forward(self, latent):
        x = self.flatten(latent)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = x.view((-1, 18, 512))
        #x = x.view((-1, 512))
        return x

class PoseRegressor(torch.nn.Module):
    def __init__(self, output_count = 3):
        super(PoseRegressor,self).__init__()
        '''
        self.flatten = torch.nn.Flatten()
        #self.dense1 = torch.nn.Linear(100352, 224) # 100352 = 128X28X28
        self.dense1 = torch.nn.Linear(2048, 2048)  # pool layer2048 = 2048X1X1
        #self.dense2 = torch.nn.Linear(224, (18 * 512))
        self.dense2 = torch.nn.Linear(2048,  2048)
        self.dense3 = torch.nn.Linear(2048, 512)
        '''
        self.output_count = output_count
        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(2048,1)  # pool layer2048 = 2048X1X1
        self.bn1 = torch.nn.BatchNorm1d(num_features=1)
        self.dense2 = torch.nn.Linear(1, 1)
        self.bn2 = torch.nn.BatchNorm1d(num_features=1)
        # self.dense3 = torch.nn.Linear(512, 128)
        # self.bn3 = torch.nn.BatchNorm1d(num_features=128)
        self.dense4 = torch.nn.Linear(1, output_count)

    def forward(self, latent):
        x = self.flatten(latent)
        x = self.dense1(x)
        x = F.relu(self.bn1(x))
        # x = self.dense2(x)
        # x = F.relu(self.bn2(x))
        # x = self.dense3(x)
        # x = F.relu(self.bn3(x))
        x = self.dense4(x)
        #x = x.view((-1, 18, 512))
        x = x.view((-1, self.output_count))
        return x

class VGGToHist(torch.nn.Module):
    def __init__(self, bins_num = 20):
        super(VGGToHist,self).__init__()
        '''
        self.flatten = torch.nn.Flatten()
        #self.dense1 = torch.nn.Linear(100352, 224) # 100352 = 128X28X28
        self.dense1 = torch.nn.Linear(2048, 2048)  # pool layer2048 = 2048X1X1
        #self.dense2 = torch.nn.Linear(224, (18 * 512))
        self.dense2 = torch.nn.Linear(2048,  2048)
        self.dense3 = torch.nn.Linear(2048, 512)
        '''
        self.bins_num = bins_num
        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(2048, 2048)  # pool layer2048 = 2048X1X1
        self.relu1 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(2048, 2048)
        self.relu2 = torch.nn.ReLU()
        #self.dense3 = torch.nn.Linear(2048, 512)
        self.dense3 = torch.nn.Linear(2048, (3*bins_num))
        self.relu3 = torch.nn.ReLU()
        self.softMax = torch.nn.Softmax(dim = 2)
    def forward(self, latent):
        x = self.flatten(latent)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        #x = self.relu3(x)
        batch_num = x.shape[0]
        x = x.view(batch_num, 3, self.bins_num)
        x = self.softMax(x) # dim = [32,60]
        #x = x.view((-1, 512))
        return x

class CelebaRegressor(torch.nn.Module):
    def __init__(self,num_input_features = 2048):
        super(CelebaRegressor,self).__init__()
        '''
        self.flatten = torch.nn.Flatten()
        #self.dense1 = torch.nn.Linear(100352, 224) # 100352 = 128X28X28
        self.dense1 = torch.nn.Linear(2048, 2048)  # pool layer2048 = 2048X1X1
        #self.dense2 = torch.nn.Linear(224, (18 * 512))
        self.dense2 = torch.nn.Linear(2048,  2048)
        self.dense3 = torch.nn.Linear(2048, 512)
        '''
        self.flatten = torch.nn.Flatten()
        self.num_input_features = num_input_features
        self.dense1 = torch.nn.Linear(num_input_features,256)  # pool layer2048 = 2048X1X1
        self.bn1 = torch.nn.BatchNorm1d(num_features=256)
        self.dense2 = torch.nn.Linear(256, 256)
        self.bn2 = torch.nn.BatchNorm1d(num_features=256)
        self.dense3 = torch.nn.Linear(256, 1)
        self.bn3 = torch.nn.BatchNorm1d(num_features=1)
        #self.dense4 = torch.nn.Linear(1)

    def forward(self, latent):
        x = self.flatten(latent)
        x = self.dense1(x)
        x = F.relu(self.bn1(x))
        x = self.dense2(x)
        x = F.relu(self.bn2(x))
        x = self.dense3(x)
        x = F.sigmoid(self.bn3(x))
        #x = self.dense4(x)
        #x = x.view((-1, 18, 512))
        return x



class LandMarksRegressor(torch.nn.Module):
    def __init__(self, landmark_num = 68, num_input_features = 2048):
        super(LandMarksRegressor,self).__init__()
        '''
        self.flatten = torch.nn.Flatten()
        #self.dense1 = torch.nn.Linear(100352, 224) # 100352 = 128X28X28
        self.dense1 = torch.nn.Linear(2048, 2048)  # pool layer2048 = 2048X1X1
        #self.dense2 = torch.nn.Linear(224, (18 * 512))
        self.dense2 = torch.nn.Linear(2048,  2048)
        self.dense3 = torch.nn.Linear(2048, 512)
        '''
        self.landmark_num = landmark_num*2
        self.num_input_features = num_input_features
        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(num_input_features,256)  # pool layer2048 = 2048X1X1
        self.bn1 = torch.nn.BatchNorm1d(num_features=256)
        self.dense2 = torch.nn.Linear(256, 256)
        self.bn2 = torch.nn.BatchNorm1d(num_features=256)
        self.dense3 = torch.nn.Linear(256, self.landmark_num)
        # self.bn3 = torch.nn.BatchNorm1d(num_features=self.landmark_num)
        #self.dense4 = torch.nn.Linear(1)

    def forward(self, latent):
        x = self.flatten(latent)
        x = self.dense1(x)
        x = F.relu(self.bn1(x))
        x = self.dense2(x)
        x = F.relu(self.bn2(x))
        x = self.dense3(x)
        # x = F.relu(self.bn3(x))
        #x = self.dense4(x)
        #x = x.view((-1, 18, 512))
        return x

class VGGToLatent2(torch.nn.Module):
    def __init__(self):
        super(VGGToLatent, self).__init__()

        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(100352, 224) # 100352 = 128X28X28
        self.dense2 = torch.nn.Linear(224,  512)


    def forward(self, latent):
        x = self.flatten(latent)
        x = self.dense1(x)
        x = self.dense2(x)
        # x = x.view((-1, 18, 512))
        x = x.view((-1, 512))
        return x


class ImageLatentDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, dlatents, image_size=256, transforms = None):
        self.filenames = filenames
        self.dlatents = dlatents
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        dlatent = self.dlatents[index]

        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))

        if self.transforms:
            image = self.transforms(image)

        return image, dlatent

    def load_image(self, filename):
        image = np.asarray(Image.open(filename))

        return image

class ImageLabelDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, labels, image_size=256, transforms = None):
        self.filenames = filenames
        self.labels = labels
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]

        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))


        image = self.transforms(image)

        return image, label

    def load_image(self, filename):
        image = np.asarray(Image.open(filename))

        return image


class ImageToLandmarks_batch(torch.nn.Module):
    def __init__(self, landmark_num = 68):
        super().__init__()

        self.landmark_num = landmark_num*2
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3,stride=1, padding = 1)
        self.conv1_bn = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2,
                     stride=2)

        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = torch.nn.BatchNorm2d(128)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2,
                                        stride=2)

        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = torch.nn.BatchNorm2d(256)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2,
                                        stride=2)

        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = torch.nn.BatchNorm2d(512)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2,
                                        stride=2)
        self.conv5 = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5_bn = torch.nn.BatchNorm2d(1024)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2,
                                        stride=2)

        self.dense1 = torch.nn.Linear(4096, 1024)
        #self.dense1 = torch.nn.Linear(4096 * 1024, 1024)
        self.dense2 = torch.nn.Linear(1024, self.landmark_num)


    def forward(self, image):
        # x = image.view(image.size(0), -1)
        x = self.conv1(image)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = F.relu(x)
        x = self.pool5(x)

        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)

        return x

class ImageToLandmarks(torch.nn.Module):
    def __init__(self, landmark_num = 68):
        super().__init__()

        self.landmark_num = landmark_num*2
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3,stride=1, padding = 1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2,
                     stride=2)

        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2,
                                        stride=2)

        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2,
                                        stride=2)

        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2,
                                        stride=2)
        self.conv5 = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2,
                                        stride=2)

        self.dense1 = torch.nn.Linear(4096, 1024)
        #self.dense1 = torch.nn.Linear(4096 * 1024, 1024)
        self.dense2 = torch.nn.Linear(1024, self.landmark_num)


    def forward(self, image):
        # x = image.view(image.size(0), -1)
        x = self.conv1(image)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool5(x)

        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)

        return x




class VGGLatentDataset(torch.utils.data.Dataset):
    def __init__(self, descriptors, dlatents):
        self.descriptors = descriptors
        self.dlatents = dlatents

    def __len__(self):
        return len(self.descriptors)

    def __getitem__(self, index):
        descriptors = self.descriptors[index]
        dlatent = self.dlatents[index]

        return descriptors, dlatent






