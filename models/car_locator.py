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
    
    def predict(self, img_lst, sort_by='conf'):
        '''
        Inputs
            img_lst: list of np.array(h,w,c)
                Can be empty
                Cannot have any None elements

        output:
            [   
                # For each frame (empty list if no car in the frame)
                [
                    # For each detected car
                    {
                        'coords': (x1,y1,x2,y2),
                        'confidence': 0.99
                    }
                ]
            ]
        '''
        if not img_lst: # Empty imgs list
            return []

        # Prepare input
        input_imgs, imgs_shapes = prepare_raw_imgs(img_lst, self.pred_mode, self.img_size)
        input_imgs = input_imgs.to(self.device)

        # Yolo prediction
        with torch.no_grad():
            imgs_detections = self.model(input_imgs)
            imgs_detections = non_max_suppression(imgs_detections, self.conf_thres, self.nms_thres)

        # if no car in the frame, output empty list
        output = [[] for _ in range(len(imgs_detections))]

        for i, (img_detection, img_shape) in enumerate(zip(imgs_detections, imgs_shapes)): # for each image
            if img_detection is not None:
                # Rescale boxes to original image
                img_detection = rescale_boxes(img_detection, self.img_size, img_shape).numpy()

                # Filter out wanted classes and perform diff class NMS      
                img_detection = [det for det in img_detection if int(det[-1]) in self.idx2targetcls]
                img_detection = diff_cls_nms(img_detection, self.nms_thres, sort_by=sort_by)

                '''
                img_detection:
                [
                    np.array([x1,y1,x2,y2,conf,cls_conf,cls]),
                    ...
                ]

                now make dict for output
                '''
                img_detection = [{
                    'coords': tuple(det[:4]),
                    'confidence': det[4]    
                } for det in img_detection]

                output[i] = img_detection

        return output