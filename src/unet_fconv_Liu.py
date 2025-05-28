## Amended by Hanze Liu on 2023.12.23

import torch
import torch.nn as nn
import FCNN_plus as fn

class UNet(nn.Module):


    def __init__(self, in_channels=3, out_channels=3):


        super(UNet, self).__init__()


        self._block1 = nn.Sequential(
            fn.Fconv_PCA(3, in_channels, 12, 4, inP=3, ifIni=1, padding=1),
            nn.ReLU(inplace=True),
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))


        self._block2 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        

        self._block3 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        

        self._block4 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        

        self._block5 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        

        self._block6 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))


        self._block7 = nn.Sequential(
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))


        self._block8 = nn.Sequential(
            fn.Fconv_PCA(3, 36, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        

        self._block9 = nn.Sequential(
            fn.Fconv_PCA(3, 36, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        

        self._block10 = nn.Sequential(
            fn.Fconv_PCA(3, 36, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))


        self._block11 = nn.Sequential(
            fn.Fconv_PCA(3, 96 + in_channels, 16, 4, inP=3, ifIni=1, padding=1),
            nn.ReLU(inplace=True),
            fn.Fconv_PCA(3, 16, 8, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            fn.Fconv_PCA_out(3, 8, out_channels, 4, inP=3, padding=1))


    def forward(self, x):

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block3(pool2)
        pool4 = self._block4(pool3)
        pool5 = self._block5(pool4)

        # Decoder
        upsample5 = self._block6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block7(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block8(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block9(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block10(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block11(concat1)
