#!/usr/bin/env python
# coding: utf-8

import os
from typing import Any
import torch
import torch.nn as nn
import torchvision.transforms as tf
from PIL import Image
import torch.autograd as autograd
from networks import Discriminator,Generator2
from loss_network import LossNetwork
from loss import content_style_loss,adv_loss_d,adv_loss_g,gaze_loss_d,gaze_loss_g,reconstruction_loss
from PIL import Image
import numpy as np
import lightning as L
import yaml
from munch import DefaultMunch
# The images files have the form "ID_2m_0P_xV_yH_z.jpg" where ID is the ID of the person, 2m is fixed, 0P means head pose of 0 degrees (only head pose used in this notebook)
# x is the vertical orientation, y is the horizontal orientation and z is either L for left or R for right eye (note that the right eye patch was flipped horizontally).
# In training the images are grouped as follows:
# For a given person and a given eye (R or L) all orientations are grouped together. One element of the data set is of the form
# imgs_r,angles_r,labels,imgs_t,angles_g where imgs_r is considered the "real" image with orientation angles_r, or x_r in the paper,
# imgs_t with orientation angles_g is the image of the same person with different orientation (could be the same image since we go through a double loop) and the label is the ID of the person


with open('config.yaml') as f:
    config = yaml.safe_load(f)
config = DefaultMunch.fromDict(config)
for k, v in config.items():
    print(k, v)



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, transform=None):
        super().__init__()
        self.transform = transform
        self.ids=50
        self.data_path = dir_path
        self.file_names = [f for f in os.listdir(self.data_path)
                      if f.endswith('.jpg')]
        self.file_dict = dict()
        for f_name in self.file_names:
            fields = f_name.split('.')[0].split('_')
            identity = fields[0]
            head_pose = fields[2]
            side = fields[-1]
            key = '_'.join([identity, head_pose, side])
            if key not in self.file_dict.keys():
                self.file_dict[key] = []
                self.file_dict[key].append(f_name)
            else:
                self.file_dict[key].append(f_name)
        self.train_images = []
        self.train_angles_r = []
        self.train_labels = []
        self.train_images_t = []
        self.train_angles_g = []

        self.test_images = []
        self.test_angles_r = []
        self.test_labels = []
        self.test_images_t = []
        self.test_angles_g = []
        self.preprocess()
    def preprocess(self):

        for key in self.file_dict.keys():

            if len(self.file_dict[key]) == 1:
                continue

            idx = int(key.split('_')[0])
            flip = 1
            if key.split('_')[-1] == 'R':
                flip = -1

            for f_r in self.file_dict[key]:

                file_path = os.path.join(self.data_path, f_r)

                h_angle_r = flip * float(
                    f_r.split('_')[-2].split('H')[0]) / 15.0
                    
                v_angle_r = float(
                    f_r.split('_')[-3].split('V')[0]) / 10.0
                    

                for f_g in self.file_dict[key]:

                    file_path_t = os.path.join(self.data_path, f_g)

                    h_angle_g = flip * float(
                        f_g.split('_')[-2].split('H')[0]) / 15.0
                        
                    v_angle_g = float(
                        f_g.split('_')[-3].split('V')[0]) / 10.0
                        

                    if idx <= self.ids:
                        self.train_images.append(file_path)
                        self.train_angles_r.append([h_angle_r, v_angle_r])
                        self.train_labels.append(idx - 1)
                        self.train_images_t.append(file_path_t)
                        self.train_angles_g.append([h_angle_g, v_angle_g])
                    else:
                        self.test_images.append(file_path)
                        self.test_angles_r.append([h_angle_r, v_angle_r])
                        self.test_labels.append(idx - 1)
                        self.test_images_t.append(file_path_t)
                        self.test_angles_g.append([h_angle_g, v_angle_g])

    def __getitem__(self, index):
        return (
            self.transform(Image.open(self.train_images[index])),
                torch.tensor(self.train_angles_r[index]),
                self.train_labels[index],
            self.transform(Image.open(self.train_images_t[index])),
                torch.tensor(self.train_angles_g[index]))
        
    def __len__(self):
        return len(self.train_images)

transform=tf.Compose([tf.ToTensor(),tf.Resize((64,64),antialias=True)])
dataset=MyDataset(dir_path=config.data_path,transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
device='cuda' if torch.cuda.is_available() else 'cpu'


if os.path.isfile('discriminator.pth'):
    discriminator=torch.load('discriminator.pth')
    print('loaded discriminator')
else:
    discriminator=Discriminator()
    print('created discriminator')
if os.path.isfile('generator.pth'):
    generator=torch.load('generator.pth')
    print('loaded generator')
else:
    generator=Generator2()
    print('created generator')

generator=generator.to(device)
discriminator=discriminator.to(device)
# LR = 5e-5
# beta1=0.5
# beta2=0.999
LR=config.lr
beta1=config.beta1
beta2=config.beta2
optimizer_g = torch.optim.Adam(generator.parameters(), LR,betas=(beta1, beta2))
optimizer_d = torch.optim.Adam(discriminator.parameters(), LR,betas=(beta1, beta2))

loss_network=LossNetwork()
loss_network=loss_network.to(device)


def generator_step(generator,discriminator,loss_network,batch):
    imgs_r, angles_r, _, imgs_t, angles_g=batch
    optimizer_g.zero_grad()
    generator.train()
    discriminator.eval()
    x_g=generator(imgs_r,angles_g)
    x_recon=generator(x_g,angles_r)
    loss_adv=adv_loss_g(discriminator,imgs_r,x_g)
    loss2=content_style_loss(loss_network,x_g,imgs_t)
    loss_p=loss2[0]+loss2[1]
    loss_gg=gaze_loss_g(discriminator,x_g,angles_g)
    loss_recon=reconstruction_loss(generator,imgs_r,x_recon)
    loss=loss_adv+config.lambda_p*loss_p+config.lambda_gaze*loss_gg+config.lambda_recon*loss_recon
    loss.backward()
    optimizer_g.step()
    return loss.item()


def discriminator_step(generator,discriminator,batch):
    imgs_r, angles_r, _, _, angles_g=batch
    optimizer_d.zero_grad()
    generator.eval()
    discriminator.train()
    x_g=generator(imgs_r,angles_g)
    loss1=adv_loss_d(discriminator,imgs_r,x_g)
    loss2=gaze_loss_d(discriminator,imgs_r,angles_r)
    loss=loss1+config.lambda_gaze*loss2
    loss.backward()
    optimizer_d.step()
    return loss.item()



def recover_image(img):
    img=img.cpu().numpy().transpose(0, 2, 3, 1)*255
    return img.astype(np.uint8)
def save_images(imgs, filename):
    height=recover_image(imgs[0])[0].shape[0]
    width=recover_image(imgs[0])[0].shape[1]
    total_width=width*len(imgs)
    
    new_im = Image.new('RGB', (total_width+len(imgs), height))
    for i,img in enumerate(imgs):
        result = Image.fromarray(recover_image(img)[0])
        new_im.paste(result, (i*width+i,0))
    new_im.save(filename)


for epoch in range(config.epochs):
    count=0
  
    # for imgs_r, angles_r, labels, imgs_t, angles_g in train_loader:
    for batch in train_loader:
        count+=1
        batch=[x.to(device) for x in batch]
        l_d=discriminator_step(generator,discriminator,batch)
        if count%config.critic_iter_per_gen==0:
            l_g=generator_step(generator,discriminator,loss_network,batch)
        if count%config.image_save_freq==0:
            imgs=[batch[0]]
            for h in [-15,-10,-5,0,5,10,15]:
                    a=torch.tile(torch.tensor([h/15.,0.]),[32,1])
                    a=a.to(device)
                    y=generator(batch[0],a)
                    imgs.append(y.detach())
            save_images(imgs, "./debug/{}_{}.png".format(epoch,count))
    print(l_d,l_g)
    if epoch%config.model_save_freq==0:
        torch.save(generator, './generator.pth')
        torch.save(discriminator, './discriminator.pth')
     