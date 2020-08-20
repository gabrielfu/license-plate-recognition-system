import numpy as np
import time
from .modules.yolo_trt import TrtYOLO
from utils.utils import load_classes, get_correct_path
from utils.bbox import rescale_boxes

class PlateDetectorTRT():
    def __init__(self, cfg):
        self.input_size = cfg['input_size'] # (h,w)
        self.model_path = cfg['model_path']
        self.conf_thres = cfg['conf_thres']
        self.nms_thres = cfg['nms_thres']
        self.n_classes = cfg['n_classes']
        self.max_batch_size = cfg['max_batch_size']
        self.model = TrtYOLO(self.model_path, self.input_size, self.n_classes, self.conf_thres, self.nms_thres, self.max_batch_size)
        self.classes = load_classes(get_correct_path(cfg['class_path']))

    def predict(self, imgs_list, sort_by='conf'):
        '''
        Inputs
            imgs_list: list of np.array(h,w,c)
                Can be empty
                Cannot have any None elements
            sort_by: Sort output results by this criteria. Values: 'conf' or 'area'
            
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
#        start = time.time()
        if not imgs_list: # Empty imgs list
            return []

        imgs_detections = self.model.detect(imgs_list)

        for i, (detections, img) in enumerate(zip(imgs_detections, imgs_list)):
            img_shape = img.shape[:2]

            if detections is not None:

                # Rescale boxes to original image
                if len(detections) == 0: # no plates exist
                    imgs_detections[i] = np.array(detections)
                else:
                    if sort_by == 'conf':
                        detections = sorted(detections, key=lambda x: x[4], reverse=True)
                    imgs_detections[i] = rescale_boxes(np.array(detections), self.input_size[0], img_shape)

        imgs_detections = imgs_detections[:i+1]
#        print(f'plate_detector_trt predict time: {time.time() - start}')
        return imgs_detections
