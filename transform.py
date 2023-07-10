import torch
import torch.nn as nn
import argparse

from torchvision import transforms as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def recover_image(img):
    img=img.cpu().numpy().transpose(0, 2, 3, 1)*255
    return img.astype(np.uint8)
def save_images(imgs):
    height=recover_image(imgs[0])[0].shape[0]
    width=recover_image(imgs[0])[0].shape[1]
    total_width=width*len(imgs)
    
    new_im = Image.new('RGB', (total_width+len(imgs), height))
    for i,img in enumerate(imgs):
        result = Image.fromarray(recover_image(img)[0])
        new_im.paste(result, (i*width+i,0))
    return new_im

parser=argparse.ArgumentParser()
parser.add_argument('--generator',default='generator.pth',type=str)
parser.add_argument('--input',type=str,required=True)
parser.add_argument('--angles',nargs=2,type=str)
args=parser.parse_args()
print(args.angles,args.generator,args.input)
transform=tf.Compose([tf.ToTensor(),tf.Resize((64,64),antialias=True)])
img=Image.open(args.input)
img=transform(img).unsqueeze(0)

gen=torch.load('generator.pth').to('cpu')
gen.eval()
res=[img]
if args.angles is not  None:
    angles=list(map(float,args.angles))
    angles=torch.tensor(angles).unsqueeze(0)
    a=gen(img,angles)
    res.append(a.detach())
else:
    
    for h in [-15,-10,-5,0,5,10,15]:
        angles=torch.tensor([h/15,0.]).unsqueeze(0)
        a=gen(img,angles)
        res.append(a.detach())
from torchvision.utils import  make_grid
make_grid(res,nrow=1)
#output=save_images(res)
plt.imshow(output)
plt.show()
