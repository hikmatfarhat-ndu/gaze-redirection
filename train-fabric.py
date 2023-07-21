#!/usr/bin/env python
# coding: utf-8

import os
import comet_ml
import torch
import torch.nn as nn
import torchvision.transforms as tf
from torchvision.utils import  make_grid,save_image
from PIL import Image
import torch.autograd as autograd
from networks import Discriminator,Generator2
from loss_network import LossNetwork,LossNetwork2

from loss import content_style_loss,adv_loss_d,adv_loss_g,gaze_loss_d,gaze_loss_g,reconstruction_loss
from PIL import Image
import numpy as np
#import lightning as L
import yaml
from munch import DefaultMunch
from tqdm import tqdm
from comet_ml.integration.pytorch import log_model
from random import randint
import socket
from data import PreProcessData,TrainDataset,TestDataset
from images import recover_image,save_images,prepare_images
# The images files have the form "ID_2m_0P_xV_yH_z.jpg" where ID is the ID of the person, 2m is fixed, 0P means head pose of 0 degrees (only head pose used in this notebook)
# x is the vertical orientation, y is the horizontal orientation and z is either L for left or R for right eye (note that the right eye patch was flipped horizontally).
# In training the images are grouped as follows:
# For a given person and a given eye (R or L) all orientations are grouped together. One element of the data set is of the form
# imgs_r,angles_r,labels,imgs_t,angles_g where imgs_r is considered the "real" image with orientation angles_r, or x_r in the paper,
# imgs_t with orientation angles_g is the image of the same person with different orientation (could be the same image since we go through a double loop) and the label is the ID of the person

torch.set_float32_matmul_precision('medium')

with open('config.yaml') as f:
    config = yaml.safe_load(f)
config = DefaultMunch.fromDict(config)
print(config)
try:
    os.makedirs(os.getcwd()+"/"+config.debug_path, exist_ok=True)
except:
    print("cannot create directory ("+os.getcwd()+"/"+config.debug_path+")")
    exit()
if config.use_comet == 'online':
    try:
        socket.gethostbyname('google.com')
    except:
        config.use_comet='offline'
if config.use_comet is not None:
    if config.use_comet=='offline':
        print("using comet offline")
        experiment = comet_ml.OfflineExperiment(project_name=config.comet_project, workspace=config.comet_workspace,
                                                auto_metric_logging=False, auto_output_logging=False)
    else:
        experiment = comet_ml.Experiment(project_name=config.comet_project, workspace=config.comet_workspace,
                                         auto_metric_logging=False, auto_output_logging=False)
    experiment.log_parameters(config)
else:
    print("don't use comet")

data=PreProcessData(config.data_path)
transform=tf.Compose([tf.ToTensor(),tf.Resize((64,64),antialias=True)])
train_dataset=TrainDataset(*data.training_data(),transform=transform)
test_dataset=TestDataset(*data.testing_data(),transform=transform)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
device='cuda' if torch.cuda.is_available() else 'cpu'


if os.path.isfile('discriminator.pth') and config.resume_training:
    discriminator=torch.load('discriminator.pth')
    print('loaded discriminator')
else:
    discriminator=Discriminator()
    print('created discriminator')
if os.path.isfile('generator.pth') and config.resume_training:
    generator=torch.load('generator.pth')
    print('loaded generator')
else:
    generator=Generator2()
    print('created generator')


LR=config.lr
beta1=config.beta1
beta2=config.beta2
optimizer_g = torch.optim.Adam(generator.parameters(), LR,betas=(beta1, beta2))
optimizer_d = torch.optim.Adam(discriminator.parameters(), LR,betas=(beta1, beta2))

loss_network=LossNetwork2()
#loss_network=LossNetwork()
#loss_network=loss_network.to(device)
from lightning.fabric import Fabric
fabric=Fabric(accelerator='cuda',devices=1,precision="bf16-mixed")
fabric.launch()
generator,optimizer_g=fabric.setup(generator,optimizer_g)
discriminator,optimizer_d=fabric.setup(discriminator,optimizer_d)
loss_network=fabric.setup(loss_network)
def generator_step(generator,discriminator,loss_network,batch):
    imgs_r, angles_r, _, imgs_t, angles_g=batch
    optimizer_g.zero_grad()
    generator.train()
    discriminator.eval()
    x_g=generator(imgs_r,angles_g)
    x_recon=generator(x_g,angles_r)
    with fabric.autocast():
        loss_adv=adv_loss_g(discriminator,imgs_r,x_g)
        loss2=content_style_loss(loss_network,x_g,imgs_t)
        loss_p=loss2[0]+loss2[1]
        loss_gg=gaze_loss_g(discriminator,x_g,angles_g)
        loss_recon=reconstruction_loss(generator,imgs_r,x_recon)
        loss=loss_adv+config.lambda_p*loss_p+config.lambda_gaze*loss_gg+config.lambda_recon*loss_recon
    #loss.backward()
    fabric.backward(loss)
    optimizer_g.step()
    return loss.item()



def discriminator_step(generator,discriminator,batch):
    imgs_r, angles_r, _, _, angles_g=batch
    optimizer_d.zero_grad()
    generator.eval()
    discriminator.train()
    x_g=generator(imgs_r,angles_g)
    with fabric.autocast():
        loss1=adv_loss_d(discriminator,imgs_r,x_g)
        loss2=gaze_loss_d(discriminator,imgs_r,angles_r)
        loss=loss1+config.lambda_gaze*loss2
    #loss.backward()
    fabric.backward(loss)
    optimizer_d.step()
    return loss.item()

loop=tqdm(range(config.epochs))

#for epoch in loop:
for epoch in range(config.epochs):
    #loop.set_description(f"Epoch [{epoch+1}/{config.epochs}]")
    batch_count=0
    loss_d,loss_g=0.,0.
    for batch in tqdm(train_loader):
        batch_count+=1
        batch=[x.to(device) for x in batch]
        l_d=discriminator_step(generator,discriminator,batch)
        loss_d=0.9*loss_d+0.1*l_d
        if batch_count%config.critic_iter_per_gen==0:
            l_g=generator_step(generator,discriminator,loss_network,batch)
            loss_g=0.9*loss_g+0.1*l_g
    if epoch%config.image_save_freq==0:
        try:
            orig=next(test_iter)[0].to(device)
        except:
            test_iter=iter(test_loader)
            orig=next(test_iter)[0].to(device)# extract the images
        imgs=[torch.unbind(orig,0)]
        for h in [-15,-10,-5,0,5,10,15]:
                a=torch.tile(torch.tensor([h/15.,0.]),[config.test_batch_size,1])
                a=a.to(device)
                y=generator(orig,a).detach()
                y=torch.unbind(y,0)
                imgs.append(y)
        imgs=prepare_images(imgs,config.test_batch_size)
        filename=config.debug_path+"/{}_{}.png".format(epoch,batch_count)
        img=make_grid(imgs,nrow=8)#nrow specifies the number of columns !!!
        img=recover_image(img)
        img=Image.fromarray(img)
        img.save(filename)
        #save_image(imgs,filename,nrow=8)# nrow specifies the number of columns !!!
        if config.use_comet is not None:
            experiment.log_image(img)
    #loop.set_postfix(loss_d=loss_d,loss_g=loss_g)
    if config.use_comet is not None:
        metrics={'loss_d':l_d,'loss_g':l_g}
        experiment.log_metrics(metrics, epoch=epoch)
    if epoch%config.model_save_freq==0:
        if config.use_comet is not None:
            log_model(experiment,generator,"generator")
            log_model(experiment,discriminator,"discriminator")
        # torch.save(generator, './generator.pth')
        # torch.save(discriminator, './discriminator.pth')
        torch.save(generator.state_dict(), './generator.pth')
        torch.save(discriminator.state_dict(), './discriminator.pth')
        
        