import numpy as np

from .modules.yolo_trt import TrtYOLO
from ..utils.utils import load_classes, get_correct_path
from ..utils.bbox import diff_cls_nms, rescale_boxes


class CarLocatorTRT:
    def __init__(self, cfg):
        self.input_size = cfg['input_size'] # (h,w)
        self.model_path = cfg['model_path']
        self.conf_thres = cfg['conf_thres']
        self.nms_thres = cfg['nms_thres']
        self.n_classes = cfg['n_classes']
        self.max_batch_size = cfg['max_batch_size']
        self.model = TrtYOLO(self.model_path, self.input_size, self.n_classes, self.conf_thres, self.nms_thres, self.max_batch_size)
        self.classes = load_classes(get_correct_path(cfg['class_path']))
        self.target_classes = ['car', 'bus', 'truck']
        self.idx2targetcls = {idx:cls_name for idx, cls_name in enumerate(self.classes) if cls_name in self.target_classes}

    def predict(self, imgs_list, sort_by='conf'):
        '''
        Inputs
            imgs_list: list of np.array(h,w,c)
                Can be empty
                Cannot have any None elements
            sort_by: Sort output results by this criteria. Values: 'conf' or 'area'
            
        Outputs
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
        imgs_detections = self.model.detect(imgs_list)
        output = [[] for _ in range(len(imgs_list))]

        for i, (detections, img) in enumerate(zip(imgs_detections, imgs_list)):
            if detections is not None:
                img_shape = img.shape[:2]
                detections = [detection for detection in detections if int(detection[-1]) in self.idx2targetcls]
                detections = diff_cls_nms(detections, self.nms_thres, sort_by=sort_by)
                if len(detections) == 0: # no vehicles exist
                    detections = np.array(detections)
                else:
                    detections = rescale_boxes(np.array(detections), self.input_size[0], img_shape)
                '''
                detections:
                [
                    np.array([x1,y1,x2,y2,conf,cls_conf,cls]),
                    ...
                ]

                now make dict for output
                '''
                output[i] = [{
                    'coords': tuple(det[:4]),
                    'confidence': det[4]    
                } for det in detections]

        return output
