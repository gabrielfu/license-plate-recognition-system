import sys
import os
import yaml
from shapely.geometry import Polygon, box


def read_yaml(path):
    return yaml.load(open(path, 'r'), Loader=yaml.FullLoader)

def get_correct_path(relative_path):
    '''
    Used when packaged app with PyInstaller
    To find external paths outside of the packaged app
    '''
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def compute_area(box):
    '''
    Compute area of a bbox
    '''
    x1, y1, x2, y2 = box[:4]
    return (x2-x1)*(y2-y1)

def compute_iou(bbox1, bbox2):
    '''
    Compute IOU of 2 bboxes
    bbox1: (x1, y1, x2, y2)\n
    bbox2: (x1, y1, x2, y2)
    '''
    x11, y11, x12, y12 = bbox1[:4]
    x21, y21, x22, y22 = bbox2[:4]

    intersect = max(min(x12,x22)-max(x11,x21), 0) * max(min(y12,y22)-max(y11,y21), 0)
    if intersect == 0:
        return 0

    area1 = (x12-x11) * (y12-y11)
    area2 = (x22-x21) * (y22-y21)
    return intersect / (area1+area2-intersect+1e-16)

def bbox_polygon_intersection(trigger_zone, bbox):
    '''
    Check if bbox intersect with trigger_zone Polygon
    Inputs
        trigger_zone: shapely.Geometry.Polygon
        bbox: tuple (x1,y1,x2,y2)
    Outputs
        Boolean
    '''
    bbox_polygon = box(*bbox)
    return bbox_polygon.intersection(trigger_zone).area
