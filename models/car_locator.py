import torch
import cv2

from .modules.darknet import Darknet
from .utils.utils import to_tensor, prepare_raw_imgs, load_classes, get_correct_path, non_max_suppression, rescale_boxes, diff_cls_nms

class CarLocator():
    def __init__(self, cfg):
        # Yolov3 stuff
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

        # Define car detection classes
        self.target_classes = ['car', 'bus', 'truck']
        self.idx2targetcls = {idx:cls_name for idx, cls_name in enumerate(self.classes) if cls_name in self.target_classes}
    
    def predict(self, imgs_list, sort_by='conf'):
        '''
        *** Can be empty imgs_list but cannot be a list with any None inside ***
        Support arbitrary Batchsize prediction, be careful of device memory usage
        output: (x1, y1, x2, y2, conf, cls_conf, cls_pred) for each tensor in a list
        '''
        ### Yolo prediction
        # Configure input
        if not imgs_list: # Empty imgs list
            return []

        input_imgs, imgs_shapes = prepare_raw_imgs(imgs_list, self.pred_mode, self.img_size)
        input_imgs = input_imgs.to(self.device)

        # Get detections
        with torch.no_grad():
            imgs_detections = self.model(input_imgs)
            imgs_detections = non_max_suppression(imgs_detections, self.conf_thres, self.nms_thres)

        for i, (detection, img_shape) in enumerate(zip(imgs_detections, imgs_shapes)): # for each image
            if detection is not None:
                # Rescale boxes to original image
                detection = rescale_boxes(detection, self.img_size, img_shape).numpy()

                # Filter out wanted classes and perform diff class NMS      
                detection = [det for det in detection if int(det[-1]) in self.idx2targetcls]
                detection = diff_cls_nms(detection, self.nms_thres, sort_by=sort_by)
                imgs_detections[i] = detection

        return imgs_detections
