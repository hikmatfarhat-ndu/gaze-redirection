import torch,torchvision
import torchvision.transforms as transforms
from PIL import Image
vgg=torchvision.models.vgg16(weights=torchvision.models.vgg.VGG16_Weights.DEFAULT)
transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((64,64),antialias=True)])
img=Image.open('/home/user/Downloads/dataset/0P/0012_2m_0P_0V_-15H_R.jpg')
img=transform(img).unsqueeze(0)
output=vgg(img)