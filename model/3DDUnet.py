import sys
sys.path.append('')


import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from thop import profile
from dcn.modules.deform_conv import *


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))
        
        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):

        x = self.conv(x) + self.conv_residual(x)
        return x



class conv_block_3D(nn.Module):
    def __init__(self, in_ch, out_ch, frames):
        super(conv_block_3D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))
        self.conv_residual = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.conv(x) + self.conv_residual(x)
        return x


class De_conv_block_3D(nn.Module):
    def __init__(self, in_ch, out_ch, frames):
        super(De_conv_block_3D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            DeformConvPack_d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, dimension='HW'),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.conv_residual = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.conv(x) + self.conv_residual(x)
        return x



class Unet_3DD(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, frames=16):
        super(Unet_3DD, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]


        self.Down1 = nn.Conv2d(filters[0], filters[0], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down2 = nn.Conv2d(filters[1], filters[1], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down3 = nn.Conv2d(filters[2], filters[2], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down4 = nn.Conv2d(filters[3], filters[3], kernel_size=4, stride=2, padding=1, bias=True)


        self.Conv1 = conv_block_3D(in_ch, filters[0], frames=frames)
        self.Conv2 = De_conv_block_3D(filters[0], filters[1], frames=frames)
        self.Conv3 = De_conv_block_3D(filters[1], filters[2], frames=frames)
        self.Conv4 = De_conv_block_3D(filters[2], filters[3], frames=frames)
        self.Conv5 = De_conv_block_3D(filters[3], filters[4], frames=frames)



        self.Up5 = nn.ConvTranspose2d(filters[4], filters[3],2,2)
        self.Up_conv5 = De_conv_block_3D(filters[4], filters[3], frames=frames)

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2],2,2)
        self.Up_conv4 = De_conv_block_3D(filters[3], filters[2], frames=frames)

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1],2,2)
        self.Up_conv3 = De_conv_block_3D(filters[2], filters[1], frames=frames)

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0],2,2)
        self.Up_conv2 = conv_block_3D(filters[1], filters[0], frames=frames)

        self.Conv = nn.Conv3d(filters[0], 12, kernel_size=1, stride=1,padding=0)
        self.conv_global_residual = nn.Conv3d(in_ch, 12, kernel_size=1, stride=1, bias=True)
        self.depth_to_space=nn.PixelShuffle(2)


    def forward(self, x):
        e1 = self.Conv1(x)   #[1, 4, 16, 128, 128] ->  [1, 32, 16, 128, 128]

        e2=self.pool_2d(self.Down1,e1) # [1, 32, 16, 128, 128] -> [1, 32, 16, 64, 64]
        e2 = self.Conv2(e2)

        e3=self.pool_2d(self.Down2,e2)
        e3 = self.Conv3(e3)

        e4=self.pool_2d(self.Down3,e3)
        e4 = self.Conv4(e4)

        e5=self.pool_2d(self.Down4,e4)
        e5 = self.Conv5(e5)

        d5 = self.up_2d(self.Up5, e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.up_2d(self.Up4, d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.up_2d(self.Up3, d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.up_2d(self.Up2, d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)+self.conv_global_residual(x)
        out = self.Conv(d2)

        b, c, t, h, w = out.shape
        out = out.permute(0, 2, 1, 3, 4)
        out = self.depth_to_space(out.reshape(b*t, c, h, w)).reshape(b, t, c//4, h*2, w*2).permute(0, 2, 1, 3, 4)

        return out
    

    def pool_2d(self, opt, x):                           #Conv2d(x)  
        b, c, t, h, w = x.shape                          #x.shape=[1, 32, 16, 128, 128]
        x = x.permute(0, 2, 1, 3, 4) # b t c h w
        out = opt(x.reshape(b*t, c, h, w)).reshape(b, t, c, h//2, w//2)             #out.shape=[16, 32, 64, 64]    out.reshape(b, t, c, h//2, w//2).shape=[1, 16, 32, 64, 64]
        return out.permute(0, 2, 1, 3, 4)                #out.permute(0, 2, 1, 3, 4).shape=[1, 32, 16, 64, 64]
    

    def up_2d(self, opt, x):
        b, c, t, h, w = x.shape                         #x.shape=[1, 512, 16, 8, 8]
        x = x.permute(0, 2, 1, 3, 4) # b t c h w        #x.shape=[1, 16, 512, 8, 8]
        out = opt(x.reshape(b*t, c, h, w)).reshape(b, t, c//2, h*2, w*2)                #x.reshape(b*t, c, h, w).shape=[16, 512, 8, 8]     out.shape=[16, 256, 16, 16]
        return out.permute(0, 2, 1, 3, 4)


from torch.nn.init import kaiming_normal_, constant_

class Model(nn.Module):
    """
    set the model
    """
    def __init__(self, para):
        super(Model, self).__init__()
        assert para.frames == 16 , "Unet takes 16 frames as input"
        self.net = Unet_3DD(in_ch=4,out_ch=3,frames=16 )  

    def forward(self, x, profile_flag=False):
        out = self.net(x)

        return out


def feed(model, iter_samples):
    inputs = iter_samples[0]
    outputs = model(inputs)
    return outputs


def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, 4, seq_length, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops, params
