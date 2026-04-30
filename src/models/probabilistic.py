import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_base import ResNetBlock
from models.attention import SelfAttention

class ProbabilisticMultiTaskResNet(nn.Module):
    def __init__(self, num_devices=30, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = ResNetBlock(32, 32, stride=1)
        self.layer2 = ResNetBlock(32, 64, stride=2)
        self.layer3 = ResNetBlock(64, 64, stride=2)
        
        self.attention = SelfAttention(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
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
        
        features_dropped = self.dropout(features)
        
        out_device = self.device_head(features_dropped)
        out_rogue = self.rogue_head(features_dropped)
        
        return out_device, out_rogue

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()