import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=10):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.center = conv_block(256, 512)

        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec6 = conv_block(512, 256)

        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec7 = conv_block(256, 128)

        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec8 = conv_block(128, 64)

        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec9 = conv_block(64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        c = self.center(self.pool4(e4))

        d6 = self.dec6(torch.cat([self.up6(c), e4], dim=1))
        d7 = self.dec7(torch.cat([self.up7(d6), e3], dim=1))
        d8 = self.dec8(torch.cat([self.up8(d7), e2], dim=1))
        d9 = self.dec9(torch.cat([self.up9(d8), e1], dim=1))

        return torch.sigmoid(self.final(d9))


# Example instantiation
# model = UNet(in_channels=8, out_channels=10)

