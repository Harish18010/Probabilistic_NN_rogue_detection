import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MultiTaskResNet(nn.Module):
    def __init__(self, num_devices=30):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = ResNetBlock(32, 32, stride=1)
        self.layer2 = ResNetBlock(32, 64, stride=2)
        self.layer3 = ResNetBlock(64, 64, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.device_head = nn.Linear(64, num_devices)
        self.rogue_head = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(-1, 1, 128, 128)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.pool(x)
        features = x.view(x.size(0), -1)
        
        out_device = self.device_head(features)
        out_rogue = self.rogue_head(features)
        
        return out_device, out_rogue

if __name__ == "__main__":
    model = MultiTaskResNet(num_devices=30)
    dummy_input = torch.randn(16, 16384)
    out_dev, out_rog = model(dummy_input)
    print(out_dev.shape)
    print(out_rog.shape)