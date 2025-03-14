import torch
import torch.nn as nn
import FCNN_plus as fn


class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv2, pool2
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv3, pool3
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv4, pool4
        self._block4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv5, pool5
        self._block5 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv6, upsample5
        self._block6 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block7 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv4a, dec_deconv4b, upsample3
        self._block8 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        # Layers: dec_deconv3a, dec_deconv3b, upsample2
        self._block9 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        # Layers: dec_deconv2a, dec_deconv2b, upsample1
        self._block10 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block11 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1))

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

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
      
class UNet_F(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet_F, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            fn.Fconv_PCA(3, in_channels, 12, 4, inP=3, ifIni=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1), 
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv2, pool2
        self._block2 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv3, pool3
        self._block3 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv4, pool4
        self._block4 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv5, pool5
        self._block5 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv6, upsample5
        self._block6 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block7 = nn.Sequential(
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv4a, dec_deconv4b, upsample3
        self._block8 = nn.Sequential(
            fn.Fconv_PCA(3, 36, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        # Layers: dec_deconv3a, dec_deconv3b, upsample2
        self._block9 = nn.Sequential(
            fn.Fconv_PCA(3, 36, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        # Layers: dec_deconv2a, dec_deconv2b, upsample1
        self._block10 = nn.Sequential(
            fn.Fconv_PCA(3, 36, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block11 = nn.Sequential(
            fn.Fconv_PCA(3, 96 + in_channels, 16, 4, inP=3, ifIni=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA(3, 16, 8, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA_out(3, 8, out_channels, 4, inP=3, padding=1))
        

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

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
    

class MaskNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaskNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活确保值在0到1之间

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        mask = self.sigmoid(x) 

        return mask


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)                 
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, out_channels, 3, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x + residual 
        return x



class AdaReNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(AdaReNet, self).__init__()

        self.unet = UNet(in_channels, out_channels)
        self.unet_f = UNet_F(in_channels, out_channels)

        self.mask_network = MaskNetwork(in_channels, out_channels=1)  # 根据需要调整输出通道数
        self.resnet = ResNetBlock(in_channels, out_channels)                                      

    def forward(self, x):
        output_unet = self.unet(x)
        output_unet_f = self.unet_f(x)

        mask = self.mask_network(x)

        combined_output = output_unet * mask + output_unet_f * (1 - mask)

        corrected_output = self.resnet(combined_output)                                     
        
        return corrected_output                                                          