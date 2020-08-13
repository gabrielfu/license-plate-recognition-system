import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', activation=None, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        '''
        activation: torch.nn object
        '''
        super().__init__()
        self.kernel_size = kernel_size
        if isinstance(stride, int):  # For symmetric stride
            stride = (stride,stride)
        self.stride = stride
        self.padding = padding
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)
        if activation is not None:
            self.activation = activation()

    def forward(self, imgs):
        '''
        imgs - (N,C,H,W)
        '''
        if self.padding == 'same':  # same padding
            num_pad_h = (self.stride[0]*(imgs.shape[2]-1)+self.kernel_size[0]-imgs.shape[2])
            num_pad_w = (self.stride[1]*(imgs.shape[2]-1)+self.kernel_size[1]-imgs.shape[2])
            out = F.pad(imgs, pad=(int(np.ceil(num_pad_w/2)),int(np.floor(num_pad_w/2)),int(np.ceil(num_pad_h/2)),int(np.floor(num_pad_h/2))))
        if isinstance(self.padding, tuple):  # valid padding
            assert len(self.padding) == 2 or len(self.padding) == 4, 'Invalid padding dimension, either symmetric (Height,Width) or asymmetric (Left,Right,Top,Bottom) is valid'
            if len(self.padding) == 2:
                out = F.pad(imgs, pad=(self.padding[0], self.padding[0], self.padding[1], self.padding[1]))
            if len(self.padding) == 4:
                out = F.pad(imgs, pad=(self.padding[0], self.padding[1], self.padding[2], self.padding[3]))
        out = self.conv2d(out)
        if self.activation:
            out = self.activation(out)
        return out
    
class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Charnet(nn.Module):
    def __init__(self, input_size, out_dim):
        '''
        input_size: (C,H,W)
        Takes in RGB image
        '''
        super().__init__()
        self.input_size = input_size
        self.out_dim = out_dim
        self.net = nn.Sequential(
            Conv2d(3,32,(3,3),padding='same',activation=nn.SELU),
            Conv2d(32,32,(3,3),padding='same',activation=nn.SELU),
            Conv2d(32,32,(3,3),padding='same',activation=nn.SELU),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.25),
            Conv2d(32,64,(3,3),padding='same',activation=nn.SELU),
            Conv2d(64,64,(3,3),padding='same',activation=nn.SELU),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.25),
            Flatten(),
            nn.Linear(1024,256),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.Dropout(0.25),
            nn.Linear(128,self.out_dim)
        )
    def forward(self, imgs):
        return self.net(imgs)

if __name__ == '__main__':
    from torchsummary import summary
    model = CharRecognizer((3,16,16),33)
    print(summary(model, input_size=(3,16,16)))