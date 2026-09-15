import torch
import torch.nn as nn

from bev_perception.models.humanpose_attention import SpatialPoseTransformer, get_valid_person


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class BevEncoder(nn.Module):
    def __init__(self, features_dim=256):
        super().__init__()
        block = Bottleneck
        layers = [2, 1, 1]
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(64, self.inplanes, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256),
        )
        self.downsample2 = nn.Sequential(nn.Conv2d(128, 256, 1, stride=2), nn.BatchNorm2d(256))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1), nn.BatchNorm2d(512),
        )
        self.downsample3 = nn.Sequential(nn.Conv2d(64, 512, 1, stride=4), nn.BatchNorm2d(512))
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.state_embedding = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 64))
        self.linear_fc = nn.Sequential(
            nn.Linear(256 * block.expansion + 64, features_dim),
            nn.ReLU(),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
        for module in self.modules():
            if isinstance(module, Bottleneck):
                nn.init.constant_(module.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, 1, self._norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, norm_layer=self._norm_layer))
        return nn.Sequential(*layers)

    def encode_bev(self, x):
        x = self.maxpool(self.relu(self.bn1(x)))
        identity3 = self.downsample3(x)
        x = self.layer1(x)
        identity2 = self.downsample2(x)
        x = self.relu2(self.conv2_2(self.layer2(x)) + identity2)
        x = self.relu3(self.conv3_2(self.layer3(x)) + identity3)
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x, state):
        bev_out = self.encode_bev(x)
        state_out = self.state_embedding(state.reshape(-1, 5))
        return self.linear_fc(torch.cat((bev_out, state_out), dim=1))


class SocialBevEncoder(BevEncoder):
    def __init__(self, features_dim=256):
        super().__init__(features_dim)
        self.humanpose_encoder = SpatialPoseTransformer()
        self.linear_fc = nn.Sequential(
            nn.Linear(256 * Bottleneck.expansion + 128 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, x, state, human_pose):
        bev_out = self.encode_bev(x)
        pose_out = self.humanpose_encoder(human_pose, state, get_valid_person(human_pose))
        state_out = self.state_embedding(state.reshape(-1, 5))
        return self.linear_fc(torch.cat((bev_out, pose_out, state_out), dim=1))
