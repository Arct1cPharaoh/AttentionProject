import torch
import torch.nn as nn
import torchvision.models as models

class SaliencyModel(nn.Module):
    def __init__(self):
        super(SaliencyModel, self).__init__()

        # ResNet-50 encoder (pretrained, no classifier)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Output: [B, 2048, 7, 7]

        # Decoder: upsample back to 224x224
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 14x14

            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 28x28

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 56x56

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 112x112

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 224x224

            nn.Conv2d(64, 1, kernel_size=1)  # Output: saliency map [B, 1, 224, 224]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
