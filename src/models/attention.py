import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_base import ResNetBlock

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        q = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, width * height)
        
        energy = torch.bmm(q, k)
        attention = F.softmax(energy, dim=-1)
        
        v = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        return self.gamma * out + x

class AttentionMultiTaskResNet(nn.Module):
    def __init__(self, num_devices=30):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = ResNetBlock(32, 32, stride=1)
        self.layer2 = ResNetBlock(32, 64, stride=2)
        self.layer3 = ResNetBlock(64, 64, stride=2)
        
        self.attention = SelfAttention(64)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.device_head = nn.Linear(64, num_devices)
        self.rogue_head = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(-1, 1, 128, 128)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.attention(x)
        
        x = self.pool(x)
        features = x.view(x.size(0), -1)
        
        out_device = self.device_head(features)
        out_rogue = self.rogue_head(features)
        
        return out_device, out_rogue

if __name__ == "__main__":
    model = AttentionMultiTaskResNet(num_devices=30)
    dummy_input = torch.randn(16, 16384)
    out_dev, out_rog = model(dummy_input)
    print(out_dev.shape)
    print(out_rog.shape)