import torch.nn.functional as F
from torchvision.models import vgg16
from models.vgg_face_dag import vgg_face_dag
from models.Vgg_m_face_bn_dag import vgg_m_face_bn_dag
from models.vgg_face2 import resnet50_scratch_dag
import numpy as np
from torchvision import transforms
from PIL import Image
from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.image_to_latent import ImageToLandmarks


import torch

class PostSynthesisProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.min_value = -1
        self.max_value = 1

    def forward(self, synthesized_image):
        synthesized_image = (synthesized_image - self.min_value) * torch.tensor(255).float() / (self.max_value - self.min_value)
        synthesized_image = torch.clamp(synthesized_image + 0.5, min=0, max=255)

        return synthesized_image  #the value between 0-255, dim = [1,3,1024,1024]

class VGGProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.image_size = 256   #vggpre

        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(-1, 1, 1)

    def forward(self, image):

        image = image / torch.tensor(255).float()
        image = image.float()
        image = F.adaptive_avg_pool2d(image, self.image_size)

        image = (image - self.mean) / self.std #value from: vgg[-2.0172 - 2.2489] vggface[-129 - -92]

        return image
'''
def cropimage(image):
    image = image.detach().cpu().numpy()
    image = Image.fromarray(image[0].astype('uint8'), 'RGB')

    #image = np.transpose(PIL_image, (1, 2, 0)).astype(np.uint8)
    scale = transforms.Compose([transforms.Scale((256, 256))])
    image = scale(image)

    crop_obj = transforms.CenterCrop((224, 224))
    image = crop_obj(image)
    image = np.array(image)
    #image = np.transpose(image, (2, 0, 1)).astype(np.uint8)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)

    image = torch.from_numpy(image).cuda()
    image = image.unsqueeze(0).requires_grad_(True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = torch.tensor(image, device=device).float() #[224,224,3]
    image =image.unsqueeze(0).requires_grad_(True)
    #image = transforms.ToTensor()

    return image
'''



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
        if image.shape[2] != 224  or image.shape[3] != 224:
            image = F.adaptive_avg_pool2d(image, self.image_size)

        image = (image - self.mean) / self.std  # value from: vgg[-2.0172 - 2.2489] vggface[-129 - -92]

        return image

class LatentOptimizerVGGface(torch.nn.Module):
    def __init__(self, synthesizer, layer=12):
        super().__init__()

        self.synthesizer = synthesizer.cuda().eval()

        self.post_synthesis_processing = PostSynthesisProcessing()
        self.vgg_processing = VGGFaceProcessing()
        self.vgg_face_dag = resnet50_scratch_dag(r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\resnet50_scratch_dag.pth').cuda().eval()



    def forward(self, dlatents):
        generated_image = self.synthesizer(dlatents) #dim [1,3,1024,1024]
        generated_image = self.post_synthesis_processing(generated_image)
        generated_image = self.vgg_processing(generated_image)  #value between -118 and 159 , dim = [1,3,224,224]
        features = self.vgg_face_dag(generated_image)   # value between 0 and182 , dim = [1,3,224,224]


        return features

class LatentOptimizerLandmarkRegressor(torch.nn.Module):
    def __init__(self, synthesizer, layer=12):
        super().__init__()

        self.synthesizer = synthesizer.cuda().eval()

        self.post_synthesis_processing = PostSynthesisProcessing()
        self.vgg_processing = VGGFaceProcessing()
        self.vgg_face_dag = resnet50_scratch_dag('./resnet50_scratch_dag.pth').cuda().eval()
        self.weights_path = "./Trained_model/Image_to_landmarks_Regressor.pt"
        self.landmark_regressor = ImageToLandmarks(landmark_num = 68).cuda().eval()
    def forward(self, dlatents):
        image_size = 64
        style_gan_transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        generated_image = self.synthesizer(dlatents) #dim [1,3,1024,1024]
        generated_image = self.post_synthesis_processing(generated_image)
        self.landmark_regressor.load_state_dict(torch.load(self.weights_path))

        out = []
        for x_ in generated_image.cpu():
            out.append(style_gan_transformation(x_))
        generated_image = torch.stack(out).cuda()

        # generated_image = to_pil(generated_image)
        # # generated_image = transforms.ToTensor()(style_gan_transformation(transforms.ToPILImage()(generated_image)))
        # generated_image = style_gan_transformation(generated_image)
        features = self.landmark_regressor(generated_image).requires_grad_()

        return features



class LatentOptimizerVGGface2(torch.nn.Module):
    def __init__(self, mappinglayer,truncationlayer,synthesizer, layer=12):
        super().__init__()

        self.mappinglayers = mappinglayer.cuda().eval()
        self.truncationlayers = truncationlayer.cuda().eval()
        self.synthesislayers = synthesizer.cuda().eval()

        self.post_synthesis_processing = PostSynthesisProcessing()
        self.vgg_processing = VGGFaceProcessing()
        self.vgg_face_dag = resnet50_scratch_dag('./resnet50_scratch_dag.pth').cuda().eval()



    def forward(self, dlatents):
        w = self.mappinglayers(dlatents) #dim [1,3,1024,1024]
        wp = self.truncationlayers(w)
        generated_image = self.synthesislayers(wp)
        generated_image = self.post_synthesis_processing(generated_image)
        generated_image = self.vgg_processing(generated_image)  #value between -118 and 159 , dim = [1,3,224,224]
        features = self.vgg_face_dag(generated_image)   # value between 0 and182 , dim = [1,3,224,224]


        return features

class LatentOptimizerVGGface3(torch.nn.Module):
    def __init__(self, synthesizer, layer=12):
        super().__init__()

        self.synthesizer = synthesizer

        self.post_synthesis_processing = PostSynthesisProcessing()
        self.vgg_processing = VGGFaceProcessing()
        self.vgg_face_dag = resnet50_scratch_dag('./resnet50_scratch_dag.pth').cuda().eval()



    def forward(self, dlatents):
        output = self.synthesizer(dlatents) #dim [1,3,1024,1024]
        generated_image = output['image']
        #generated_image = generated_image.transpose(0, 3, 1, 2)
        generated_image = torch.from_numpy(generated_image).type(torch.FloatTensor).cuda()

        generated_image = self.vgg_processing(generated_image)  #value between -118 and 159 , dim = [1,3,224,224]
        features = self.vgg_face_dag(generated_image).requires_grad_(True)  # value between 0 and182 , dim = [1,3,224,224]


        return features

def LatentOptimizerVGGface_vgg_to_latent2(dlatents):


    post_synthesis_processing = PostSynthesisProcessing()
    vgg_processing = VGGFaceProcessing()
    vgg_face_dag = resnet50_scratch_dag('./resnet50_scratch_dag.pth').cuda().eval()

    kwargs = {'latent_space_type': 'W'}
    model = StyleGANGenerator('stylegan_ffhq')
    dlatents = dlatents.detach().cpu().numpy()
    latent_codes = model.preprocess(dlatents, **kwargs) #normlaze the latens
    output = model.easy_synthesize(latent_codes,
                                            **kwargs)

    generated_image = output['image']
    #generated_image = generated_image.transpose(0, 2, 3, 1) [1,3,1024,1024]
    generated_image = generated_image.transpose(0, 3, 1, 2)
    generated_image = torch.from_numpy(generated_image).type(torch.FloatTensor).cuda()  #dim (1,1024,1024,3)
    generated_image = vgg_processing(generated_image)  #value between -118 and 159 , dim = [1,3,224,224]
    features = vgg_face_dag(generated_image)   # value between 0 and182 , dim = [1,3,224,224]


    return features

class torch_to_numpy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, latents_to_be_optimized):
        latents_to_be_optimized = latents_to_be_optimized.detach().cpu().numpy()

        return latents_to_be_optimized  #the value between 0-255, dim = [1,3,1024,1024]


class LatentOptimizerVGGface_vgg_to_latent(torch.nn.Module):
    def __init__(self, synthesizer, layer=12):
        super().__init__()

        self.synthesizer = synthesizer
        self.post_synthesis_processing = PostSynthesisProcessing()
        self.vgg_processing = VGGFaceProcessing()
        self.vgg_face_dag = resnet50_scratch_dag('./resnet50_scratch_dag.pth').cuda().eval()
        self.kwargs = {'latent_space_type': 'W'}


    def forward(self, dlatents):

        output = self.synthesizer.easy_synthesize(dlatents,
                                      **self.kwargs) #dim [1,3,1024,1024]
        generated_image = output['image']
        generated_image = generated_image.transpose(0, 3, 1, 2)
        generated_image = torch.from_numpy(generated_image).type(torch.FloatTensor).cuda()  # dim (1,1024,1024,3)
        generated_image = self.vgg_processing(generated_image)  #value between -118 and 159 , dim = [1,3,224,224]
        features = self.vgg_face_dag(generated_image).requires_grad_(True)   # value between 0 and182 , dim = [1,3,224,224]


        return features #dim [1,2048,1,1]

class LatentOptimizerVGGface2(torch.nn.Module):
    def __init__(self, synthesizer, layer=12):
        super().__init__()

        self.synthesizer = synthesizer.cuda().eval()
        self.post_synthesis_processing = PostSynthesisProcessing()
        self.vgg_processing = VGGFaceProcessing()
        self.vgg_face_dag = vgg_face_dag('./vgg_face_dag.pth').cuda().eval()



    def forward(self, dlatents):
        generated_image = self.synthesizer(dlatents)
        generated_image = self.post_synthesis_processing(generated_image)
        generated_image = self.vgg_processing(generated_image)  #value between -118 and 159 , dim = [1,3,224,224]
        features = self.vgg_face_dag(generated_image)   # value between 0 and182 , dim = [1,3,224,224]

        return features

class LatentOptimizer(torch.nn.Module):
    def __init__(self, synthesizer, layer=12):
        super().__init__()

        self.synthesizer = synthesizer.cuda().eval()
        self.post_synthesis_processing = PostSynthesisProcessing()
        self.vgg_processing = VGGProcessing()
        self.vgg16 = vgg16(pretrained=True).features[:layer].cuda().eval()


    def forward(self, dlatents):
        #epsilon = 1e-8
        #dlatents = dlatents / torch.sqrt(torch.mean(dlatents ** 2, dim=1, keepdim=True) + epsilon)
        generated_image = self.synthesizer(dlatents)
        generated_image = self.post_synthesis_processing(generated_image) #the value between 0-255, dim = [1,3,1024,1024]
        generated_image = self.VGGProcessing()(generated_image)  #value between 2.5344 and -2.1179 , dim = [1,3,256,256]
        features = self.vgg16(generated_image) #value beteween 0-72.4174,   dim = [1,256,64.64]

        return features


class LatentOptimizerPixelLoss(torch.nn.Module):
    def __init__(self, synthesizer, layer=12):
        super().__init__()

        self.synthesizer = synthesizer.cuda().eval()
        self.post_synthesis_processing = PostSynthesisProcessing()
        self.vgg_processing = VGGProcessing()
        self.vgg16 = vgg16(pretrained=True).features[:layer].cuda().eval()


    def forward(self, dlatents):
        generated_image = self.synthesizer(dlatents)
        generated_image = self.post_synthesis_processing(generated_image)
        generated_image = self.vgg_processing(generated_image)

        return generated_image