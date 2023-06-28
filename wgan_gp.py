import torch
import torch.nn as nn
from torch.optim import Adam
from torch import autograd
from pathlib import Path

import os
import torchvision.transforms as tf

from net import NetG, NetD
from utils import init_weight
from layers import LayerNorm2d, PixelNorm
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import random
from tqdm import tqdm, trange
class WGAN_GP():
    """
    WGAN_GP Wasserstein GANs with Gradient Penalty.
    Gets rid of gradient clipping from WGAN and uses
    gradient clipping instead to enforce 1-Lipschitz
    continuity. Everything else is the same as WGAN.
    """

    def __init__(self):
        super().__init__()
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        seed = 42
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        tf_list = [
            tf.ToTensor(),
            tf.Normalize(0.5, 0.5),
            tf.Resize((64, 64),antialias=True),
        ]

        transforms = tf.Compose(tf_list)

        self.dataset = ImageFolder(
            root="/home/user/PyTorch-Lightning-GAN/GAN/celeba",
              transform=transforms
        )
        self.epochs=6
        self.batch_size=256
        self.z_dim=100
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

        

        self.current_ep = 0
        self.netG = NetG(
            z_dim=self.z_dim,
            out_ch=3,
            norm_layer=LayerNorm2d,
            final_activation=torch.tanh,
        )
        self.netD = NetD(3, norm_layer=LayerNorm2d)

        self.netG.apply(init_weight)
        self.netD.apply(init_weight)

        self.n_critic = 5
        self.fixed_noise = self.sample_noise()
        self._update_model_optimizers()

    def _update_model_optimizers(self):
        self.netG = self.netG.to(self.device)
        self.netD = self.netD.to(self.device)

        self.optG = Adam(self.netG.parameters(), lr=1.0e-4)
        self.optD = Adam(self.netD.parameters(), lr=1.0e-4)

    def generator_step(self, data):
        self.netG.train()
        self.netD.eval()

        self.optG.zero_grad()

        noise = self.sample_noise()

        fake_images = self.netG(noise)

        fake_logits = self.netD(fake_images)

        loss = -fake_logits.mean().view(-1)

        loss.backward()
        self.optG.step()
        return loss.item()
    def sample_noise(self):
        return torch.randn(self.batch_size, self.z_dim, 1, 1).to(
            self.device
        )
    def critic_step(self, data):
        self.netG.eval()
        self.netD.train()

        self.optD.zero_grad()

        real_images = data[0].float().to(self.device)

        noise = self.sample_noise()
        fake_images = self.netG(noise)

        real_logits = self.netD(real_images)
        fake_logits = self.netD(fake_images)
        w_gp=10
        gradient_penalty = w_gp * self._compute_gp(
            real_images, fake_images
        )

        loss_c = fake_logits.mean() - real_logits.mean()

        loss = loss_c + gradient_penalty

        loss.backward()
        self.optD.step()

        return loss.item()
        #,gradient_penalty.item()
        

    def _compute_gp(self, real_data, fake_data):
        batch_size = real_data.size(0)
        eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        eps = eps.expand_as(real_data)
        interpolation = eps * real_data + (1 - eps) * fake_data

        interp_logits = self.netD(interpolation)
        grad_outputs = torch.ones_like(interp_logits)

        gradients = autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)
    def train_epoch(self, dataloader):
        

        loop = tqdm(dataloader, desc="Iteration: ", ncols=75, leave=False)
        d_loss=0
        g_loss=0
        for ix, data in enumerate(loop):
            d_loss+=self.critic_step(data)
            if ix % self.n_critic == 0:
                g_loss+=self.generator_step(data)
        return d_loss/len(dataloader),g_loss/len(dataloader)

    def train(self):
        inter_results=[]
        loop = trange(self.epochs, desc="Epoch: ", ncols=75)
        for ep in enumerate(loop):          
            d_loss,g_loss=self.train_epoch(self.dataloader)
            print(f"Epoch: {ep} D_loss: {d_loss} G_loss: {g_loss}")
            if ep[0]%2==0:
                inter_results.append(self.generate_images(16,self.fixed_noise))
        p=Path('./debug').mkdir(parents=True, exist_ok=True)
        for i,img in enumerate(inter_results):
            plt.imsave('./debug/'+str(i)+'.png', img)
    

    def generate_images(self, n_samples,noise=None ):
        self.netG.eval()
        with torch.no_grad():
            if noise==None:
                noise = self.sample_noise()[:n_samples]
            fake_images = self.netG(noise)
            grid = make_grid(fake_images[:n_samples], nrow=4, normalize=True)
        return np.transpose(grid.cpu().numpy(),(1,2,0) ) 
    
