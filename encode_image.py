import argparse
from tqdm import tqdm
import numpy as np
import torch
from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.latent_optimizer import LatentOptimizer,LatentOptimizerVGGface
from models.image_to_latent import ImageToLatent
from models.image_to_latent import VGGToLatent, VGGLatentDataset,PoseRegressor

from models.losses import LatentLoss,IdentityLoss
from utilities.hooks import GeneratedImageHook
from utilities.images import load_images, images_to_video, save_image
from utilities.files import validate_path
from models.latent_optimizer import VGGFaceProcessing,LatentOptimizerVGGface_vgg_to_latent
from models.vgg_face2 import resnet50_scratch_dag

parser = argparse.ArgumentParser(description="Find the latent space representation of an input image.")
parser.add_argument("image_path", help="Filepath of the image to be encoded.")
parser.add_argument("dlatent_path", help="Filepath to save the dlatent (WP) at.")

parser.add_argument("--save_optimized_image", default=False, help="Whether or not to save the image created with the optimized latents.", type=bool)
parser.add_argument("--optimized_image_path", default="optimized.png", help="The path to save the image created with the optimized latents.", type=str)
parser.add_argument("--video", default=False, help="Whether or not to save a video of the encoding process.", type=bool)
parser.add_argument("--video_path", default="video.avi", help="Where to save the video at.", type=str)
parser.add_argument("--save_frequency", default=10, help="How often to save the images to video. Smaller = Faster.", type=int)
parser.add_argument("--iterations", default=1000, help="Number of optimizations steps.", type=int)
parser.add_argument("--model_type", default="stylegan_ffhq", help="The model to use from InterFaceGAN repo.", type=str)
parser.add_argument("--learning_rate", default=1, help="Learning rate for SGD.", type=int)
parser.add_argument("--vgg_layer", default=12, help="The VGG network layer number to extract features from.", type=int)
parser.add_argument("--use_latent_finder", default=False, help="Whether or not to use a latent finder to find the starting latents to optimize from.", type=bool)
parser.add_argument("--image_to_latent_path", default="image_to_latent.pt", help="The path to the .pt (Pytorch) latent finder model.", type=str)
parser.add_argument("--get_firstimage", default="first_image.png", help="The path to save the image before optimastion.", type=str)
parser.add_argument("--use_vggfacelatent_finder", default=False, help="Whether or not to use a vggface latent finder to find the starting latents to optimize from.", type=bool)
parser.add_argument("--vggface_to_latent_path", default="vgg_to_latent.pt", help="The path to the vggface latent finder model.", type=str)



args, other = parser.parse_known_args()

vggface = True  #use vggface pretrained model
vgg_identityLoss = True# use identitylkoss to replace perceptual loss
CompleteStyleGAN = False # Use the complete styleGAn network
vgg_to_latent_model = True #'vgg_to_latent' and 'use_latent_finder' are mutex
pose_regressor_model = False
latent_type = 'WP'

def optimize_latents():

    print("Optimizing Latents.")

    #image_path = args.image_path

    if vggface:
        '''
        mappinglayer = StyleGANGenerator(args.model_type).model.mapping
        truncationlayer = StyleGANGenerator(args.model_type).model.truncation
        synthesizer = StyleGANGenerator(args.model_type).model.synthesis

        latent_optimizer = LatentOptimizerVGGface(mappinglayer,truncationlayer,synthesizer, args.vgg_layer)
        '''
        '''
        if vgg_to_latent_model:
            synthesizer = StyleGANGenerator(args.model_type)
            latent_optimizer = LatentOptimizerVGGface(synthesizer, args.vgg_layer)
            
        '''

        synthesizer = StyleGANGenerator(args.model_type).model.synthesis
        latent_optimizer = LatentOptimizerVGGface(synthesizer, args.vgg_layer)

    else:
        synthesizer = StyleGANGenerator(args.model_type).model.synthesis
        latent_optimizer = LatentOptimizer(synthesizer, args.vgg_layer)

    # Optimize only the dlatents.
    for param in latent_optimizer.parameters():
        param.requires_grad_(False)
    
    if args.video or args.save_optimized_image:
        # Hook, saves an image during optimization to be used to create video.
        generated_image_hook = GeneratedImageHook(latent_optimizer.post_synthesis_processing, args.save_frequency)

    reference_image = load_images([args.image_path])
    reference_image = torch.from_numpy(reference_image).cuda()         #reference_image pixel value: 0-255


    #reference_image = latent_optimizer.vgg_processing(reference_image) #vgg16: the vlaue between -2.04 - 2.54,dim = [1,3,256,256]
                                                                        #vggface: the value between -0.48889 - 0.6086 dim = [1,3,224,224]
    
    if vggface:
        reference_image = latent_optimizer.vgg_processing(reference_image)
        reference_features = latent_optimizer.vgg_face_dag(reference_image).detach()  # vggface:x12 the value between 0-172.6192  dim = [1,256,56,56]
        # pool5_7x7_s1 dim = [1, 2048, 1, 1]
    else:
        reference_image = latent_optimizer.vgg_processing(reference_image)  # vgg16: the vlaue between -2.04 - 2.54,dim = [1,3,256,256]
        reference_features = latent_optimizer.vgg16(reference_image).detach()         #vgg16: the value between 0-60.6167  dim = [1,256,64,64]

    reference_image = reference_image.detach()

    if args.use_latent_finder:
        image_to_latent = ImageToLatent().cuda()
        image_to_latent.load_state_dict(torch.load(args.image_to_latent_path))
        image_to_latent.eval()

        latents_to_be_optimized = image_to_latent(reference_image)
        latents_to_be_optimized = latents_to_be_optimized.detach().cuda().requires_grad_(True)
    elif vgg_to_latent_model:
        if latent_type == 'Z':
            vgg_to_latent = VGGToLatent().cuda()
            vgg_to_latent.load_state_dict(torch.load("vgg_to_latent_Z.pt"))
            vgg_to_latent.eval()
            vgg_face_dag = resnet50_scratch_dag(
                r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\resnet50_scratch_dag.pth').cuda().eval()
            vgg_processing = VGGFaceProcessing()
            image = load_images([args.image_path])
            image = torch.from_numpy(image).cuda()
            image = vgg_processing(image)  # vgg16: the vlaue between -2.04 - 2.54,dim = [1,3,256,256]
            feature = vgg_face_dag(image)
            #latents_to_be_optimized = vgg_to_latent(feature).detach().cuda().requires_grad_(True)


            run_device = 'cuda'
            latent_space_dim = 512

            truncation = StyleGANGenerator("stylegan_ffhq").model.truncation
            mapping = StyleGANGenerator("stylegan_ffhq").model.mapping

            latents_to_be_optimized = vgg_to_latent(feature).detach().cpu().numpy()
            latents_to_be_optimized = latents_to_be_optimized.reshape(-1, latent_space_dim)
            # normalization Z vector
            norm = np.linalg.norm(latents_to_be_optimized, axis=1, keepdims=True)
            latents_to_be_optimized = latents_to_be_optimized / norm * np.sqrt(latent_space_dim)
            latents_to_be_optimized.astype(np.float32)
            latents_to_be_optimized = torch.from_numpy(latents_to_be_optimized).type(torch.FloatTensor)

            mapping = mapping.cuda().eval()
            truncation = truncation.cuda().eval()
            latents_to_be_optimized = latents_to_be_optimized.detach().to(run_device).requires_grad_(True)

            latents_to_be_optimized = mapping(latents_to_be_optimized)
            latents_to_be_optimized = truncation(latents_to_be_optimized.unsqueeze(0))

        elif latent_type == 'WP':
            vgg_to_latent = VGGToLatent().cuda()
            vgg_to_latent.load_state_dict(torch.load("vgg_to_latent_WP.pt"))
            vgg_to_latent.eval()
            vgg_face_dag = resnet50_scratch_dag(
                r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\resnet50_scratch_dag.pth').cuda().eval()
            vgg_processing = VGGFaceProcessing()
            image = load_images([args.image_path])
            image = torch.from_numpy(image).cuda()
            image = vgg_processing(image)  # vgg16: the vlaue between -2.04 - 2.54,dim = [1,3,256,256]
            feature = vgg_face_dag(image)
            # latents_to_be_optimized = vgg_to_latent(feature).detach().cuda().requires_grad_(True)

            latents_to_be_optimized = vgg_to_latent(feature).detach().cpu().numpy()
            latents_to_be_optimized = torch.from_numpy(latents_to_be_optimized).type(torch.FloatTensor).detach().cuda().requires_grad_(True)

            '''
            run_device = 'cuda'
            latent_space_dim = 512
            num = 1
            w_space_dim = 512
            num_layers = 18
    
            truncation = StyleGANGenerator("stylegan_ffhq").model.truncation
            mapping = StyleGANGenerator("stylegan_ffhq").model.mapping
            
            latents_to_be_optimized = vgg_to_latent(feature).detach().cpu().numpy()
            latents_to_be_optimized = latents_to_be_optimized.reshape(-1, latent_space_dim)
            norm = np.linalg.norm(latents_to_be_optimized, axis=1, keepdims=True)
            latents_to_be_optimized = latents_to_be_optimized / norm * np.sqrt(latent_space_dim)
            latents_to_be_optimized.astype(np.float32)
            latents_to_be_optimized = torch.from_numpy(latents_to_be_optimized).type(torch.FloatTensor)
    
            mapping = mapping.cuda().eval()
            truncation = truncation.cuda().eval()
            latents_to_be_optimized = latents_to_be_optimized.to(run_device).requires_grad_(True)
    
            latents_to_be_optimized = mapping(latents_to_be_optimized)
            latents_to_be_optimized = truncation(latents_to_be_optimized.unsqueeze(0))
            #latents_to_be_optimized = truncation(latents_to_be_optimized.unsqueeze(0)).to(run_device).requires_grad_(True)
            '''

    elif pose_regressor_model:

        pose_regressor = PoseRegressor().cuda()
        pose_regressor.load_state_dict(torch.load("pose_regressor.pt"))
        pose_regressor.eval()
        vgg_face_dag = resnet50_scratch_dag(
            r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\resnet50_scratch_dag.pth').cuda().eval()
        vgg_processing = VGGFaceProcessing()
        image = load_images([args.image_path])
        image = torch.from_numpy(image).cuda()
        image = vgg_processing(image)  # vgg16: the vlaue between -2.04 - 2.54,dim = [1,3,256,256]
        feature = vgg_face_dag(image)
        # latents_to_be_optimized = vgg_to_latent(feature).detach().cuda().requires_grad_(True)

        run_device = 'cuda'
        latent_space_dim = 512

        truncation = StyleGANGenerator("stylegan_ffhq").model.truncation
        mapping = StyleGANGenerator("stylegan_ffhq").model.mapping

        latents_to_be_optimized = pose_regressor(feature).detach().cpu().numpy()
        latents_to_be_optimized = latents_to_be_optimized.reshape(-1, latent_space_dim)
        # normalization Z vector
        norm = np.linalg.norm(latents_to_be_optimized, axis=1, keepdims=True)
        latents_to_be_optimized = latents_to_be_optimized / norm * np.sqrt(latent_space_dim)
        latents_to_be_optimized.astype(np.float32)
        latents_to_be_optimized = torch.from_numpy(latents_to_be_optimized).type(torch.FloatTensor)

        mapping = mapping.cuda().eval()
        truncation = truncation.cuda().eval()
        latents_to_be_optimized = latents_to_be_optimized.detach().to(run_device).requires_grad_(True)

        latents_to_be_optimized = mapping(latents_to_be_optimized)
        latents_to_be_optimized = truncation(latents_to_be_optimized.unsqueeze(0))

    else:
        run_device = 'cuda'
        latent_space_dim = 512
        num = 1
        w_space_dim = 512
        num_layers = 18

        if CompleteStyleGAN:
            '''
            truncation = StyleGANGenerator("stylegan_ffhq").model.truncation
            mapping = StyleGANGenerator("stylegan_ffhq").model.mapping
            if not (args.use_vggfacelatent_finder or vgg_to_latent_model):
                latents_to_be_optimized = np.random.randn(num, latent_space_dim)  # output dim: [1,512]
            else: None

            latents_to_be_optimized = latents_to_be_optimized.reshape(-1, latent_space_dim)
            norm = np.linalg.norm(latents_to_be_optimized, axis=1, keepdims=True)
            latents_to_be_optimized = latents_to_be_optimized / norm * np.sqrt(latent_space_dim)
            latents_to_be_optimized.astype(np.float32)
            latents_to_be_optimized = torch.from_numpy(latents_to_be_optimized).type(torch.FloatTensor)

            mapping = mapping.cuda().eval()
            truncation = truncation.cuda().eval()
            latents_to_be_optimized = latents_to_be_optimized.to(run_device)

            latents_to_be_optimized = mapping(latents_to_be_optimized)
            latents_to_be_optimized = truncation(latents_to_be_optimized.unsqueeze(0)).to(run_device).detach().requires_grad_(True)
            '''

            latents_to_be_optimized = np.random.randn(num, latent_space_dim)
            latents_to_be_optimized = latents_to_be_optimized.reshape(-1, latent_space_dim)
            norm = np.linalg.norm(latents_to_be_optimized, axis=1, keepdims=True)
            latents_to_be_optimized = latents_to_be_optimized / norm * np.sqrt(latent_space_dim)
            latents_to_be_optimized.astype(np.float32)
            latents_to_be_optimized = torch.from_numpy(latents_to_be_optimized).type(torch.FloatTensor).cuda().detach().requires_grad_(True)
            # latents_to_be_optimized = torch.from_numpy(latents_to_be_optimized).type(torch.FloatTensor).

        else:
            latents_to_be_optimized = torch.zeros((1, 18, 512)).cuda().requires_grad_(True)

    if vgg_identityLoss:
        criterion= IdentityLoss()
    else:
        criterion = LatentLoss()
    optimizer = torch.optim.SGD([latents_to_be_optimized], lr=args.learning_rate)

    progress_bar = tqdm(range(args.iterations))
    for step in progress_bar:
        optimizer.zero_grad()


        generated_image_features = latent_optimizer(latents_to_be_optimized) #vgg16: the value 0- 53.79: dim = [1,256,64,64]
                                                                             #vggface: the value 0- 36833.3398: dim = [1,256,56,56]
        loss = criterion(generated_image_features, reference_features)
        loss.backward()
        loss = loss.item()

        optimizer.step()
        progress_bar.set_description("Step: {}, Loss: {}".format(step, loss))
    
    optimized_dlatents = latents_to_be_optimized.detach().cpu().numpy()
    np.save(args.dlatent_path, optimized_dlatents)

    if args.video:
        images_to_video(generated_image_hook.get_images(), args.video_path)
    if args.save_optimized_image:
        save_image(generated_image_hook.last_image, args.optimized_image_path)
        #save_image(generated_image_hook.last_image, "rconstuctuion"+image_path) #拼图用
    if args.get_firstimage:
        save_image(generated_image_hook.first_image, "first_image.png")  #vggface:image mean:105.36 pretrained:image_mean:122.248

def main():
    assert(validate_path(args.image_path, "r"))
    assert(validate_path(args.dlatent_path, "w"))
    assert(1 <= args.vgg_layer <= 16)
    if args.video: assert(validate_path(args.video_path, "w"))
    if args.save_optimized_image: assert(validate_path(args.optimized_image_path, "w"))
    if args.use_latent_finder: assert(validate_path(args.image_to_latent_path, "r"))
    
    optimize_latents()

if __name__ == "__main__":
    main()

# import PIL.Image as Image
# import os
# IMAGES_FORMAT = ['.jpg', '.JPG']
# IMAGES_PATH = r'C:\Users\Mingrui\Desktop\datasets'
# if __name__ == "__main__":
#     image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
#                    os.path.splitext(name)[1] == item]
#
#     for i in image_names:
#         image_path = i
#         optimize_latents(image_path)
#     main()
