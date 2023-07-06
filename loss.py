import torch
import torch.autograd as autograd
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def compute_gp(discriminator,real_data, fake_data):
        batch_size = real_data.size(0)
        eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        eps = eps.expand_as(real_data)
        interpolation = eps * real_data + (1 - eps) * fake_data

        interp_logits = discriminator(interpolation)[0]
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
def adv_loss(discriminator,real,fake):
    real_logits = discriminator(real)[0]
    fake_logits =discriminator(fake)[0]
    w_gp=10
    gradient_penalty = w_gp * compute_gp(discriminator,real,fake)

    loss_c = fake_logits.mean() - real_logits.mean()
    loss = loss_c + gradient_penalty
    #return loss, gradient_penalty,loss_c
    return loss
import numpy as np

def gaze_loss_d(discriminator,x_r,angles_r):
    loss=torch.nn.MSELoss()
    y=discriminator(x_r)[1].squeeze()
    return loss(y,angles_r)
def gaze_loss_g(discriminator,x_g,angles_g):
    loss=torch.nn.MSELoss()
    y=discriminator(x_g)[1].squeeze()
    return loss(y,angles_g)
    
def reconstruction_loss(generator,x,y):
    loss=torch.nn.L1Loss()
    return loss(y,x)
    
def content_style_loss(loss_network,x,y):
    mse_loss=torch.nn.MSELoss()
    with torch.no_grad():
        xc = x.detach()
        features_y = loss_network(y)
        features_xc = loss_network(xc)
        f_xc_c = features_xc[2].detach()
        content_loss = mse_loss(features_y[2], f_xc_c)
    height, width,channels = y.size()[2],y.size()[3],y.size()[1]
    content_loss /= (height * width * channels)

    style_loss = 0.
    #for l, weight in enumerate(STYLE_WEIGHTS):
    for l in range(len(features_y)):
        gram_x = gram_matrix(features_xc[l])
        gram_y = gram_matrix(features_y[l])
        style_loss +=  mse_loss(gram_x, gram_y)
    return content_loss,style_loss