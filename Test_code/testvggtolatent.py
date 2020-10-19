from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.latent_optimizer import PostSynthesisProcessing
from models.image_to_latent import VGGToLatent, VGGLatentDataset
from models.losses import LogCoshLoss
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from glob import glob
from tqdm import tqdm_notebook as tqdm
# from tqdm.notebook import tqdm
import numpy as np
from utilities.images import load_images, images_to_video, save_image
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
import torch.utils.data as Data
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag

from PIL import Image

latent_space_dim = 512
run_device = 'cuda'

# load Model
vgg_to_latent = VGGToLatent().cuda()
vgg_to_latent.load_state_dict(torch.load("vgg_to_latent_Z.pt"))
vgg_to_latent.eval()

directory = "C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/InterFaceGAN/dataset_directory/"
descriptor = np.load(directory + "descriptors.npy")

vgg_face_dag = resnet50_scratch_dag(r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\resnet50_scratch_dag.pth').cuda().eval()

vgg_processing = VGGFaceProcessing()

image_name = 'test_01.png'
image = load_images([image_name])
image = torch.from_numpy(image).cuda()
image = vgg_processing(image)  # vgg16: the vlaue between -2.04 - 2.54,dim = [1,3,256,256]
feature = vgg_face_dag(image)

latents_to_be_optimized = vgg_to_latent(feature).cpu().detach().numpy()

'''
norm = np.linalg.norm(latents_to_be_optimized, axis=1, keepdims=True)
latents_to_be_optimized = latents_to_be_optimized / norm * np.sqrt(latent_space_dim)
latents_to_be_optimized.astype(np.float32)
latents_to_be_optimized = torch.from_numpy(latents_to_be_optimized).type(torch.FloatTensor)
'''

from InterFaceGAN.utils.logger import setup_logger
import os.path
import cv2
from collections import defaultdict

#logger = setup_logger('vgg_latent_result_image', logger_name='generate_data')
kwargs = {'latent_space_type': 'Z'}
#model = StyleGANGenerator('stylegan_ffhq', logger)
model = StyleGANGenerator('stylegan_ffhq')

output_dir = 'vgg_latent_result_image'
latent_codes = model.preprocess(latents_to_be_optimized, **kwargs)
outputs = model.easy_synthesize(latent_codes,
                                      **kwargs)

results = defaultdict(list)
for key, val in outputs.items():  # key: WP val:[4,18,512]
    if key == 'image':
        for image in val:
            #save_path = os.path.join(output_dir + image_name)
            cv2.imwrite(output_dir + image_name, image[:, :, ::-1])

    else:
        results[key].append(val)

'''
#logger.info(f'Saving results.')
for key, val in results.items():
    save_path = os.path.join(output_dir, f'{key}.npy')
    np.save(save_path, np.concatenate(val, axis=0))
'''


'''
truncation = StyleGANGenerator("stylegan_ffhq").model.truncation
mapping = StyleGANGenerator("stylegan_ffhq").model.mapping
synthesizer = StyleGANGenerator("stylegan_ffhq").model.synthesis
mapping = mapping.cuda().eval()
truncation = truncation.cuda().eval()
synthesizer = synthesizer.cuda().eval()
latents_to_be_optimized = latents_to_be_optimized.to(run_device)

latents_to_be_optimized = mapping(latents_to_be_optimized)
#latents_to_be_optimized = truncation(latents_to_be_optimized.unsqueeze(0)).to(run_device).detach().requires_grad_(True)
latents_to_be_optimized = truncation(latents_to_be_optimized.unsqueeze(0))
image = synthesizer(latents_to_be_optimized).to(run_device).detach()

min_value = -1
max_value = 1
synthesized_image = (image - min_value) * torch.tensor(255).float() / (max_value - min_value)
synthesized_image = torch.clamp(synthesized_image + 0.5, min=0, max=255)

synthesized_image = synthesized_image[0].cpu().detach().numpy()
synthesized_image = np.transpose(synthesized_image, (1, 2, 0)).astype(np.uint8)
synthesized_image = Image.fromarray(synthesized_image)
synthesized_image.save("testvgg_to_latent.jpg")
'''