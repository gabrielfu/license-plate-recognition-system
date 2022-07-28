'''
ResNet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNet(nn.Module):
    def __init__(self, out_dim, size=34):
        super().__init__()
        self.out_dim = out_dim
        if size == 18:
            self.resnet = torchvision.models.resnet18(pretrained=False)
        elif size == 34:
            self.resnet = torchvision.models.resnet34(pretrained=False)
        elif size == 50:
            self.resnet = torchvision.models.resnet50(pretrained=False)
        elif size == 101:
            self.resnet = torchvision.models.resnet101(pretrained=False)
        elif size == 152:
            self.resnet = torchvision.models.resnet152(pretrained=False)
        else:
            raise NotImplementedError(f'Wrong model: ResNet-{size} is not implemented')
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.out_dim)
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, imgs):
        out = self.resnet(imgs)
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = nn.Dropout(0.5)(out)
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = nn.Dropout(0.25)(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        
        return out