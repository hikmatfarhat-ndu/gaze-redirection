from collections import namedtuple

import torch
import torchvision.models.vgg as vgg

LossOutput = namedtuple(
    "LossOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])


class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(weights=vgg.VGG19_Weights.DEFAULT).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)
class LossNetwork2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.all_layers=vgg.vgg16(weights=vgg.VGG16_Weights.DEFAULT).features
        self.model=vgg.vgg16(weights=vgg.VGG16_Weights.DEFAULT)
        self.all_layers=[]
        mods=self.model.modules()
        for idx,m in enumerate(mods):
            if idx!=0 and not isinstance(m,torch.nn.Sequential) and not isinstance(m,torch.nn.AdaptiveAvgPool2d):
                self.all_layers.append(m)
        self.needed_layers=[3,8,15,22]
    def forward(self,x):
        output=[]
        for idx,layer in enumerate(self.all_layers):
            x=layer(x)
            if idx in self.needed_layers:
                output.append(x)
            if idx==22:
                break
        return output