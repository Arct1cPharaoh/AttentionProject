import torch
import torch.nn as nn
import torchvision.models as models

# Don't mind the weird comments here
class SaliencyModel(nn.Module):
    def __init__(self):
        super(SaliencyModel, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # 256
        self.encoder3 = resnet.layer2                                          # 512
        self.encoder4 = resnet.layer3                                          # 1024
        self.encoder5 = resnet.layer4                                          # 2048

        self.up1 = self.up_block(2048, 1024)         # decoder step
        self.up2 = self.up_block(1024 + 1024, 512)
        self.up3 = self.up_block(512 + 512, 256)
        self.up4 = self.up_block(256 + 256, 128)
        self.up5 = self.up_block(128 + 64, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)  # 64
        e2 = self.encoder2(e1)  # 256
        e3 = self.encoder3(e2)  # 512
        e4 = self.encoder4(e3)  # 1024
        e5 = self.encoder5(e4)  # 2048

        d1 = self.up1(e5)                   # → 14x14
        d1 = torch.cat([d1, e4], dim=1)     # 1024 + 1024
        d2 = self.up2(d1)                   # → 28x28
        d2 = torch.cat([d2, e3], dim=1)     # 512 + 512
        d3 = self.up3(d2)                   # → 56x56
        d3 = torch.cat([d3, e2], dim=1)     # 256 + 256
        d4 = self.up4(d3)                   # → 112x112
        d4 = torch.cat([d4, e1], dim=1)     # 128 + 64
        d5 = self.up5(d4)                   # → 224x224

        out = self.final(d5)                # → 1x224x224
        return out
