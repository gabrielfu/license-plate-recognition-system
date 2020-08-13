from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Rewritten
class UpsampleLayer(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return x

# Rewritten
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

# Rewritten
def create_darknet(darknet_cfg):
    hyperparams = darknet_cfg.pop(0)
    output_channels = [int(hyperparams["channels"])]
    net = nn.ModuleList()
    for layer_idx, layer_block_cfg in enumerate(darknet_cfg):
        module = nn.Sequential()

        if layer_block_cfg["type"] == "upsample":
            upsample_layer = UpsampleLayer(scale_factor=int(layer_block_cfg["stride"]))
            module.add_module(f"{layer_idx}_upsample", upsample_layer)

        elif layer_block_cfg["type"] == "shortcut":
            out_channel = output_channels[-1]
            module.add_module(f"{layer_idx}_shortcut", EmptyLayer())

        elif layer_block_cfg["type"] == "route":
            route_layer_indices = list(map(int, layer_block_cfg['layers'].split(',')))
            # Concatenation(2 layers) or new branch(1 layer)
            out_channel = sum([output_channels[1:][i] for i in route_layer_indices])
            module.add_module(f"{layer_idx}_route", EmptyLayer())

        elif layer_block_cfg["type"] == "yolo":
            anchor_indices = list(map(int, layer_block_cfg["mask"].split(',')))
            anchors = list(map(int, layer_block_cfg["anchors"].split(',')))
            anchors = [(anchors[i*2],anchors[i*2+1]) for i in anchor_indices]
            num_classes = int(layer_block_cfg["classes"])
            img_size = int(hyperparams["height"])
            yolo_layer = YoloLayer(anchors, num_classes, img_size)
            module.add_module(f"{layer_idx}_yolo", yolo_layer)

        elif layer_block_cfg["type"] == "convolutional":
            use_bn = bool(layer_block_cfg["batch_normalize"])
            out_channel = int(layer_block_cfg["filters"])
            kernel_size = int(layer_block_cfg["size"])

            module.add_module(
                f"{layer_idx}_conv2d",
                nn.Conv2d(
                    in_channels=output_channels[-1],
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=int(layer_block_cfg["stride"]),
                    padding=(kernel_size // 2),
                    bias= not use_bn,
                ),
            )
            if use_bn:
                module.add_module(
                    f"{layer_idx}_batchnorm2d", 
                    nn.BatchNorm2d(out_channel, momentum=float(hyperparams['momentum']))
                )
            if layer_block_cfg["activation"] == "leaky":
                module.add_module(
                    f"{layer_idx}_leakyrelu", 
                    nn.LeakyReLU(negative_slope=0.1)
                )

        net.append(module)
        output_channels.append(out_channel)

    return hyperparams, net

# Rewritten
def parse_yolo_config(path):
    '''Parses yolo layer config'''
    with open(path,'r') as f:
        lines = f.read()
    lines = lines.split('\n')
    lines = [x.strip() for x in lines if x and not x.startswith('#')]

    block = {}
    blocks = []
    for line in lines:
        # new block
        if line.startswith('['):
            if block:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].strip()
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        # same block
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)

    return blocks


# Rewritten
class YoloLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim

    def forward(self, prediction, img_dim=None):

        FloatTensor = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor

        self.img_dim = img_dim
        num_samples = prediction.size(0)
        grid_size = prediction.size(2)

        prediction = (
            prediction.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        bbox_x = torch.sigmoid(prediction[..., 0])  # Center x
        bbox_y = torch.sigmoid(prediction[..., 1])  # Center y
        bbox_w = prediction[..., 2]  # Width
        bbox_h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        
        # Compute x, y centre offsets and w, h anchor offsets for each grid
        stride = self.img_dim / grid_size
        grid_y, grid_x = torch.meshgrid(torch.arange(grid_size),torch.arange(grid_size))
        offset_x = grid_x.view([1, 1, grid_size, grid_size]).type(FloatTensor)
        offset_y = grid_y.view([1, 1, grid_size, grid_size]).type(FloatTensor)
        
        scaled_anchors = FloatTensor(self.anchors) / float(stride)
        anchor_w = scaled_anchors[:, 0].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = bbox_x.data + offset_x
        pred_boxes[..., 1] = bbox_y.data + offset_y
        pred_boxes[..., 2] = torch.exp(bbox_w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(bbox_h.data) * anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        return output

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super().__init__()
        self.darknet_cfg = parse_yolo_config(config_path)
        self.hyperparams, self.net = create_darknet(self.darknet_cfg)
        self.img_size = img_size

    def forward(self, img):
        img_channel = img.shape[2]
        layer_outputs, yolo_outputs = [], []

        for i, (layer_block_cfg, layer_block) in enumerate(zip(self.darknet_cfg, self.net)):
            if layer_block_cfg['type'] in ['convolutional', 'upsample']:
                img = layer_block(img)
            elif layer_block_cfg['type'] == 'route':
                route_layer_indices = list(map(int, layer_block_cfg['layers'].split(',')))
                outputs = [layer_outputs[int(layer_idx)] for layer_idx in route_layer_indices]
                img = torch.cat(outputs, 1)
            elif layer_block_cfg['type'] == 'shortcut':
                layer_idx = int(layer_block_cfg['from'])
                img = layer_outputs[-1] + layer_outputs[layer_idx]
            elif layer_block_cfg['type'] == 'yolo':
                img = layer_block[0](img, img_channel)
                yolo_outputs.append(img)

            layer_outputs.append(img)
        final_outputs = torch.cat(yolo_outputs, 1).detach().cpu()
        return final_outputs

    # Rewritten
    def load_darknet_weights(self, weights_path):
        with open(weights_path, "rb") as f:
            weights = np.fromfile(f, dtype=np.float32)[5:]

        for i, (layer_block_cfg, layer_block) in enumerate(zip(self.darknet_cfg, self.net)):
            if layer_block_cfg["type"] == "convolutional":
                conv_layer = layer_block[0]
                if bool(layer_block_cfg["batch_normalize"]):
                    # BN2D bias, weights, running mean and running variance (must be in order)
                    bn_layer = layer_block[1]
                    out_channel = bn_layer.weight.numel()
                    bn_params = [bn_layer.bias, bn_layer.weight, bn_layer.running_mean, bn_layer.running_var]

                    for param in bn_params:
                        param_data = torch.from_numpy(weights[:out_channel]).view_as(param)
                        param.data.copy_(param_data)
                        weights = weights[out_channel:]

                else:
                    # Conv layer bias used when there is no BN
                    out_channel = conv_layer.bias.numel()
                    conv_bias_data = torch.from_numpy(weights[:out_channel]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_bias_data)
                    weights = weights[out_channel:]

                # Conv layer weights
                conv_num_weights = conv_layer.weight.numel()
                conv_weight_data = torch.from_numpy(weights[:conv_num_weights]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_weight_data)
                weights = weights[conv_num_weights:]
