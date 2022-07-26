import torch
import cv2

from .modules.darknet import Darknet
from ..utils.image_preprocess import to_tensor, prepare_raw_imgs
from ..utils.utils import load_classes, get_correct_path
from ..utils.bbox import non_max_suppression, rescale_boxes_with_pad

class PlateDetector:
    def __init__(self, cfg):
        class_path = get_correct_path(cfg['class_path'])
        weights_path = get_correct_path(cfg['weights_path'])
        model_cfg_path = get_correct_path(cfg['model_cfg'])
        self.img_size = cfg['img_size']
        self.n_cpu = cfg['n_cpu']
        self.conf_thres = cfg['conf_thres']
        self.nms_thres = cfg['nms_thres']
        self.classes = load_classes(class_path)
        self.pred_mode = cfg['pred_mode']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set up model
        self.model = Darknet(model_cfg_path, img_size=cfg['img_size']).to(self.device)
        if cfg['weights_path'].endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()  # Set in evaluation mode
    
    def predict(self, img_lst):
        '''
        Inputs
            img_lst: list of np.array(h,w,c)
                Can be empty
                Cannot have any None elements
        Outputs
            list of np.arrays
            # for each frame
            [
                # np.array if have n >= 1 plates (x1, y1, x2, y2, conf, cls_conf, cls_pred,)
                # None if no plates
                np.array(n, 7),
                None
            ]
        '''
        if not img_lst: # Empty imgs list
            return []

        # Prepare input
        input_imgs, imgs_shapes = prepare_raw_imgs(img_lst, self.pred_mode, self.img_size)
        input_imgs = input_imgs.to(self.device)

        # Yolo prediction
        with torch.no_grad():
            img_detections = self.model(input_imgs)
            img_detections = non_max_suppression(img_detections, self.conf_thres, self.nms_thres)

        for i, (detection, img_shape) in enumerate(zip(img_detections, imgs_shapes)):
            if detection is not None:
                # Rescale boxes to original image
                img_detections[i] = rescale_boxes_with_pad(detection, self.img_size, img_shape).numpy()

        return img_detections