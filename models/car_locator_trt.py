import numpy as np

from .modules.yolo_trt import YOLO_trt
from utils.utils import load_classes, get_correct_path
from utils.bbox import diff_cls_nms

class CarLocatorTRT():
    def __init__(self, cfg):
        self.input_size = cfg['input_size'] # (h,w)
        self.model_name = cfg['model_name']
        self.conf_thres = cfg['conf_thres']
        self.n_classes = cfg['n_classes']
        self.model = YOLO_trt(self.model_name, self.input_size, self.n_classes)
        self.classes = load_classes(get_correct_path(cfg['class_path']))

        self.target_classes = ['car', 'bus', 'truck']
        self.idx2targetcls = {idx:cls_name for idx, cls_name in enumerate(self.classes) if cls_name in self.target_classes}

    def predict(self, imgs_list, sort_by='conf'):
        batch_boxes, batch_confs, batch_clss = self.model.detect(imgs_list, self.conf_thres)
        imgs_detections = []
        for batch_idx in range(len(batch_boxes)):
            single_img_boxes = batch_boxes[batch_idx]
            single_img_confs = np.expand_dims(batch_confs[batch_idx], axis=-1)
            single_img_clss = np.expand_dims(batch_clss[batch_idx], axis=-1)
            imgs_detections.append(np.concatenate((single_img_boxes, single_img_confs, single_img_clss), axis=-1))
        for i, img_detections in enumerate(imgs_detections):
            if img_detections is not None:
                img_detections = [detection for detection in img_detections if int(detection[-1]) in self.idx2targetcls]
                img_detections = diff_cls_nms(img_detections, self.nms_thres, sort_by=sort_by)
                imgs_detections[i] = img_detections

        return imgs_detections