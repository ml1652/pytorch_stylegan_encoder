
import PIL.Image as Image
from torchvision import transforms


imagepath = r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\test_01.png"

image =Image.open(imagepath)

scale = transforms.Compose([transforms.Scale((256,256))])
image = scale(image)

crop_obj = transforms.CenterCrop((224,224))
image = crop_obj(image)

image.save("crop_01.jpg",format = 'JPEG')