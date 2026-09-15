import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, out_channels=64, downsample=16):
        super().__init__()
        self.downsample = downsample

        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        # Use layers up to layer3 (downsample by 16x)
        self.stem = nn.Sequential(
            resnet.conv1,   # [B, 64, 240, 320]
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # [B, 64, 120, 160]
        )
        self.layer1 = resnet.layer1     # [B, 64, 120, 160]
        self.layer2 = resnet.layer2     # [B, 128, 60, 80]
        self.layer3 = resnet.layer3     # [B, 256, 30, 40]
        # Channel projection
        self.conv_out = nn.Conv2d(256, out_channels, kernel_size=1)

    def get_features(self, x):
        # x: [B, 3, 480, 640]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)          # [B, 256, 30, 40]
        x = self.conv_out(x)        # [B, 64, 30, 40]
        return x

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        # get feature map
        x = x.to(torch.float32)
        x = self.get_features(x)                      # [B, C, H, W]

        # Downsample depth map to match feature map resolution
        depth_map = F.interpolate(d[0], scale_factor=1 / self.downsample, mode='bilinear', align_corners=False)  # [B, 1, H, W]

        return x, depth_map