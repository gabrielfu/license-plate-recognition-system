'''
ResNeXt-50-32x4d
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNeXt(nn.Module):
    def __init__(self, out_dim, size=50):
        super().__init__()
        self.out_dim = out_dim
        if size == 50:
            self.resnext = torchvision.models.resnext50_32x4d(pretrained=False)
        else:
            raise NotImplementedError(f'Wrong model: ResNeXt-{size} is not implemented')
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.out_dim)
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, imgs):
        out = self.resnext(imgs)
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = nn.Dropout(0.5)(out)
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = nn.Dropout(0.25)(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        
        return out