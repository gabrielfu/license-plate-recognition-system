import torch
import cv2

from .modules.darknet import Darknet
from .utils.utils import to_tensor, load_classes, get_correct_path, non_max_suppression, rescale_boxes, diff_cls_nms
from .utils.preprocess import resize, pad_to_square, cv_resize, cv_preprocess

class Segmentator():
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

        # define line threshold
        self.char_line_threshold = cfg['char_line_threshold']

    def predict(self, plates_list):
        boxes_list, boxes_centres_list = self.get_rois(plates_list)
        for i, (boxes, boxes_centres) in enumerate(zip(boxes_list, boxes_centres_list)):
            if boxes is None:
                continue
            boxes_list[i] = self.sort_boxes_single(boxes, boxes_centres)
        return boxes_list

    def get_rois(self, plates_list):
        img_detections = super().predict(plates_list)
        boxes_list = [box.astype('int')[:, :4] if box is not None else box for box in img_detections]
        for i, boxes in enumerate(boxes_list): # 3D array of each char coords in each imgs
            if boxes is None:
                continue
            for j, box in enumerate(boxes):
                xmin, ymin, xmax, ymax = box
                # Include more area
                ymin = int(ymin)-2
                ymax = int(ymax)+2
                xmin = int(xmin)-2
                xmax = int(xmax)+2

                # Avoid out of boundary of image
                ymin = max(0,ymin)
                ymax = max(0,ymax)
                xmin = max(0,xmin)
                xmax = max(0,xmax)
                ymin = min(plates_list[i].shape[0],ymin)
                ymax = min(plates_list[i].shape[0],ymax)
                xmin = min(plates_list[i].shape[1],xmin)
                xmax = min(plates_list[i].shape[1],xmax)

                boxes[j] = (xmin, ymin, xmax, ymax)

        boxes_centres_list = [] # 2D array of each center point coords of each char
        for boxes in boxes_list:
            if boxes is None:
                boxes_centres_list.append(None)
                continue
            box_centres = []
            for box in boxes:
                box_centres.append(((box[0]+box[2])//2, (box[1]+box[3])//2)) #x, y
            boxes_centres_list.append(box_centres)
        return boxes_list, boxes_centres_list

    def sort_boxes_single(self, boxes, boxes_centres):
        if boxes == []:
            return []

        def avg_rois_height(boxes):
            avg_h = 0
            for box in boxes:
                avg_h += box[3]-box[1]
            avg_h /= len(boxes)
            return avg_h

        char_line_threshold = self.char_line_threshold*avg_rois_height(boxes)

        all_x = [row[0] for row in boxes_centres]
        all_y = [row[1] for row in boxes_centres]
        #zip boxes, x, y together correspondingly in to one list
        all_in_one = list(zip(boxes, all_x, all_y))
        all_in_one = sorted(all_in_one, key=lambda k: k[1])
        diff_y = []
        for i in range(len(all_in_one)-1):
            diff_y.append(abs(all_in_one[i][2]-all_in_one[i+1][2]))

        if diff_y == []:
            return None
        #if only one line, sort by x axis and done
        if max(diff_y) < char_line_threshold:
            all_in_one = sorted(all_in_one, key=lambda k: k[1])
            #print('oneline')
            return [row[0] for row in all_in_one]

        #print('twolines')
        #two lines case
        #if max_y - y coordinate of the char is larger than threshold, then its grouped into first line
        first_line = [one for one in all_in_one if one[2] - min(all_y) < char_line_threshold]
        #sort the first line by x axis and 
        first_line = sorted(first_line, key=lambda k: k[1])
        sorted_first_line = [row[0] for row in first_line]
        #all others will be grouped in to second line
        second_line = [one for one in all_in_one if one[2] - min(all_y) >= char_line_threshold]
        second_line = sorted(second_line, key=lambda k: k[1])
        sorted_second_line = [row[0] for row in second_line]

        return sorted_first_line+sorted_second_line