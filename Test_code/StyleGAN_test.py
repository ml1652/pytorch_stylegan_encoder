from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
import numpy as np
import torch
from PIL import Image


num_latents = 100
latents = []
synthesizer = StyleGANGenerator("stylegan_ffhq").model.synthesis
truncation = StyleGANGenerator("stylegan_ffhq").model.truncation
mapping = StyleGANGenerator("stylegan_ffhq").model.mapping


def convertto255(synthesized_image):
    min_value = -1
    max_value = 1
    synthesized_image = (synthesized_image - min_value) * torch.tensor(255).float() / (max_value - min_value)
    synthesized_image = torch.clamp(synthesized_image + 0.5, min=0, max=255)
    return synthesized_image

def save_image_forEasyexample(image, save_path):
    img = np.array(image)
    # img = img[0]
    # img = ((img + 1)/2)*255  # Todo: The min/max is actually smaller/larger than +-1 - need to take care of this properly
    # Use instead convertto255

    image = np.transpose(img[0], (1, 2, 0)).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_path + "%d.jpg" % (i))

save_path = "./zeros image/image"
resolution = 1024
w_space_dim = 512
num_layers = (int(np.log2(resolution)) - 1) * 2
num = 1
latent_space_dim = 512
run_device = 'cuda'
for i in range(num_latents):

    '''
    model = StyleGANGenerator("stylegan_ffhq")
    latent_type = 'Z'

    sample_code = model.easy_sample(1, latent_type)  # sample has zero mean, std=1
    model_out = model.synthesize(sample_code, latent_type, generate_style=False, generate_image=True)
    # Generate & save 10 images with latent_space_type 'Z', 10 with 'W', and 10 with 'WP'
    # model_out contains ['z'], ['w'], ['wp'], ['image']
    img = model_out['image']
    img = torch.from_numpy(img)
    img = convertto255(img)
    save_image_forEasyexample(img,save_path)
    '''
    #latent_z = torch.randn((1, 18, 512)).cuda()
    latent_z = np.random.randn(num, latent_space_dim)   #output dim: [1,512]
    latent_z = latent_z.reshape(-1, latent_space_dim)
    norm = np.linalg.norm(latent_z, axis=1, keepdims=True)
    latent_z = latent_z / norm * np.sqrt(latent_space_dim)
    latent_z.astype(np.float32)
    latent_z = torch.from_numpy(latent_z).type(torch.FloatTensor)

    synthesizer = synthesizer.cuda().eval()
    mapping = mapping.cuda().eval()

    latent_z = latent_z.to(run_device)

    output = mapping(latent_z)  # inpput【1，18，512】 output = [1,3,1024,1024]
    output = truncation(output.unsqueeze(0))
    output = synthesizer(output)
    output = convertto255(output)
    output = output.detach().cpu().numpy()[0]  # output(3, 1024, 1024)

    # toRGB
    image = np.transpose(output, (1, 2, 0)).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_path + "%d.jpg" % (i))
