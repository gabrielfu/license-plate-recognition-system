import torch
import cv2

from .modules.darknet import Darknet
from .utils.utils import to_tensor, load_classes, get_correct_path, non_max_suppression, rescale_boxes, diff_cls_nms
from .utils.preprocess import resize, pad_to_square, cv_resize, cv_preprocess

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
        
    def prepare_raw_imgs(self, imgs_list, mode):
        '''
        imgs_list: list of imgs (each img is a BGR np array read from openCV)
        '''
        imgs_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs_list]

        if mode == 'torch':
            # Torch model preprocess pipeline
            imgs = [to_tensor(img) for img in imgs_list]
            imgs_shapes = [(img.shape[1],img.shape[2]) for img in imgs]
            imgs = [resize(pad_to_square(img, pad_value=128/255)[0],self.img_size) for img in imgs]
        elif mode == 'cv2':
            # OpenCV model preprocess pipeline
            imgs_shapes = [(img.shape[0],img.shape[1]) for img in imgs_list]
            imgs = [cv_resize(cv_preprocess(img)[0],(self.img_size, self.img_size)) for img in imgs_list]
            imgs = [to_tensor(img) for img in imgs]

        return torch.stack(imgs), imgs_shapes
    
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

        input_imgs, imgs_shapes = self.prepare_raw_imgs(imgs_list, self.pred_mode)
        input_imgs = input_imgs.to(self.device)

        # Get detections
        with torch.no_grad():
            imgs_detections = self.model(input_imgs)
            imgs_detections = non_max_suppression(imgs_detections, self.conf_thres, self.nms_thres)

        for i, (detection, img_shape) in enumerate(zip(imgs_detections, imgs_shapes)):
            if detection is not None:
                # Rescale boxes to original image
                imgs_detections[i] = rescale_boxes(detection, self.img_size, img_shape).numpy()

        ### Post processing       
        for i, img_detections in enumerate(imgs_detections):
            if img_detections is not None:
                img_detections = [detection for detection in img_detections if int(detection[-1]) in self.idx2targetcls]
                img_detections = diff_cls_nms(img_detections, self.nms_thres, sort_by=sort_by)
                imgs_detections[i] = img_detections

        return imgs_detections
