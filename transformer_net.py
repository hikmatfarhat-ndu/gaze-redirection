import torch
import torch.nn as nn

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y
class Discriminator(torch.nn.Module):
    def __init__(
        self, in_ch=3, norm_layer=nn.BatchNorm2d, final_activation=None
    ):
        super().__init__()
        self.in_ch = in_ch
        self.final_activation = final_activation
        self.slope=0.2
        self.slope=0.01
        self.backbone = nn.Sequential(
            # * 64x64
            nn.Conv2d(self.in_ch, 64,kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(self.slope),
            # * 32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
           #norm_layer(128, affine=True),
            nn.LeakyReLU(self.slope),
            # * 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            #norm_layer(256, affine=True),
            nn.LeakyReLU(self.slope),
            # * 8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            #norm_layer(512, affine=True),
            nn.LeakyReLU(self.slope),
            # * 4x4
            #nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Conv2d(512,1024,kernel_size=4,stride=2,padding=1,bias=False),
        )
        self.angle=nn.Conv2d(1024,2,kernel_size=2,stride=1,padding=0,bias=False)
        self.disc=nn.Conv2d(1024,1,kernel_size=2,stride=1,padding=1,bias=False)

    def forward(self, x):
        x = self.backbone(x)
        # return (
        #     x if self.final_activation is None else self.final_activation(x)
        # )   
        return self.disc(x),self.angle(x) 
class Generator2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(5,64,kernel_size=7,stride=1,padding=3,bias=False)
        self.in1 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv2=nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=False)
        self.in2 = torch.nn.InstanceNorm2d(128, affine=True)
        self.conv3=nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=False)
        self.in3 = torch.nn.InstanceNorm2d(256, affine=True)
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(256)
        self.res5 = ResidualBlock(256)
        self.res6 = ResidualBlock(256)
        self.deconv1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.in4 = torch.nn.InstanceNorm2d(128, affine=True)
        self.deconv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.in5 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv4=torch.nn.Conv2d(64,3,kernel_size=7,stride=1,padding=3,bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self,X,angles):
        angles_reshaped=angles.reshape(-1,2,1,1)
        angles_tiled = torch.tile(angles_reshaped, [1,1, X.shape[2],
                                             X.shape[3]])
        X=torch.cat((X,angles_tiled),dim=1)
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.conv4(y)
        return torch.tanh(y)
    
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(5, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X,angles):
        angles_reshaped=angles.reshape(-1,2,1,1)
        angles_tiled = torch.tile(angles_reshaped, [1,1, X.shape[2],
                                             X.shape[3]])
        X=torch.cat((X,angles_tiled),dim=1)
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return torch.tanh(y)

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
