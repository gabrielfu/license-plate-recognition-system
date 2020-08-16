import numpy as np
import torch
import torch.nn.functional as F
import cv2

def to_tensor(img):
    '''
    Tranform np array(W,H,C) to torch tensor (C,W,H)
    '''
    max_pixel_value = np.max(img)
    if max_pixel_value > 1.:
        img = torch.tensor(img, dtype=torch.float32)/255.
        img = img.permute(2,0,1)
    else:
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2,0,1)
    return img

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def pad_to_square(img, pad_value):
    '''
    img: (c,h,w) torch tensor
    '''
    _, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # pad = torch.from_numpy(np.array(pad))  # modified for bug fixing
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

# Our codes
def cv_resize(img, shape):
    # May change interpolation to nearest neighbour later, because darknet-53 uses that in training
    img = cv2.resize(img,(shape[0], shape[1]), interpolation = cv2.INTER_CUBIC)
    return img

# Our codes
def cv_preprocess(img):
    '''
    (1920*n) * (1080*n) this shape of rectangle might be better
    Accepts image of shape (width, height, num_channels) or (width, height)
    '''
    width = img.shape[1]
    height = img.shape[0]

    diff = width - height
    pad = int(diff/2)
    if diff > 0:
        img = cv2.copyMakeBorder(img, pad, diff-pad, 0, 0, cv2.BORDER_CONSTANT, value=(128,128,128))
    elif diff < 0:
        img = cv2.copyMakeBorder(img, 0, 0, -pad, pad-diff, cv2.BORDER_CONSTANT, value=(128,128,128))

    return img, pad

def prepare_raw_imgs(imgs_list, mode, img_size):
    '''
    Resize, pad to square & make it torch.tensor() for model input

    Inputs
        imgs_list: list of imgs (each img is a BGR np array read from openCV)
    '''
    imgs_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs_list]
        
    if mode == 'torch':
        # Torch model preprocess pipeline
        imgs = [to_tensor(img) for img in imgs_list]
        imgs_shapes = [(img.shape[1],img.shape[2]) for img in imgs]
        imgs = [resize(pad_to_square(img, pad_value=128/255)[0],img_size) for img in imgs]
    elif mode == 'cv2':
        # OpenCV model preprocess pipeline
        imgs_shapes = [(img.shape[0],img.shape[1]) for img in imgs_list]
        imgs = [cv_resize(cv_preprocess(img)[0],(img_size, img_size)) for img in imgs_list]
        imgs = [to_tensor(img) for img in imgs]

    return torch.stack(imgs), imgs_shapes
