from shapely.geometry import Polygon, box
import torch
import numpy as np

def dot_inside_bbox(dot, bbox):
    """
    Check if dot is inside bbox
    dot: (x,y)
    bbox: (x1, y1, x2, y2)
    """
    x1,y1,x2,y2 = bbox
    dot_x, dot_y = dot
    if x1 < dot_x < x2 and y1 < dot_y < y2:
        return True
    else:
        return False

def bbox_polygon_intersect(trigger_zone, bbox):
    """
    Check if bbox intersect with trigger_zone Polygon
    Inputs
        trigger_zone: shapely.Geometry.Polygon
        bbox: tuple (x1,y1,x2,y2)
    Outputs
        Boolean
    """
    bbox_polygon = box(*bbox)
    return bbox_polygon.intersects(trigger_zone)

def bbox_polygon_intersection(trigger_zone, bbox):
    """
    Check if bbox intersect with trigger_zone Polygon
    Inputs
        trigger_zone: shapely.Geometry.Polygon
        bbox: tuple (x1,y1,x2,y2)
    Outputs
        Boolean
    """
    bbox_polygon = box(*bbox)
    return bbox_polygon.intersection(trigger_zone).area

def bbox_polygon_iou(trigger_zone, bbox):
    """
    Compute IoU between bbox & trigger_zone Polygon
    Inputs
        trigger_zone: shapely.Geometry.Polygon
        bbox: tuple (x1,y1,x2,y2)
    Outputs
        IoU (float)
    """
    bbox_polygon = box(*bbox)
    return bbox_polygon.intersection(trigger_zone).area / bbox_polygon.union(trigger_zone).area

def compute_iou(bbox1, bbox2):
    """
    Compute IOU of 2 bboxes
    bbox1: (x1, y1, x2, y2)\n
    bbox2: (x1, y1, x2, y2)
    """
    x11, y11, x12, y12 = bbox1[:4]
    x21, y21, x22, y22 = bbox2[:4]

    intersect = max(min(x12,x22)-max(x11,x21), 0) * max(min(y12,y22)-max(y11,y21), 0)
    if intersect == 0:
        return 0

    area1 = (x12-x11) * (y12-y11)
    area2 = (x22-x21) * (y22-y21)
    return intersect / (area1+area2-intersect+1e-16)

def compute_area(box):
    """
    Compute area of a bbox
    """
    x1, y1, x2, y2 = box[:4]
    return (x2-x1)*(y2-y1)

def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    Resizes directly and doesn't maintain aspect ratio
    Output:
        np.array(n,7)
    """
    orig_h, orig_w = original_shape
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = np.maximum(0, (boxes[:, 0] * orig_w)).astype('int')
    boxes[:, 1] = np.maximum(0, boxes[:, 1] * orig_h).astype('int')
    boxes[:, 2] = np.minimum(orig_w, boxes[:, 2] * orig_w).astype('int')
    boxes[:, 3] = np.minimum(orig_h, boxes[:, 3] * orig_h).astype('int')
    return boxes

def rescale_boxes_with_pad(boxes, current_dim, original_shape):
    """ 
    Rescales bounding boxes to the original shape 
    Maintains aspect ratio & pad to shape
    Output:
        list of (x1,y1,x2,y2,conf,cls_conf,cls)
    """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

def xywh2xyxy(inputs):
    outputs = inputs.new(inputs.shape)
    outputs[..., 0] = inputs[..., 0] - inputs[..., 2] / 2
    outputs[..., 1] = inputs[..., 1] - inputs[..., 3] / 2
    outputs[..., 2] = inputs[..., 0] + inputs[..., 2] / 2
    outputs[..., 3] = inputs[..., 1] + inputs[..., 3] / 2
    return outputs

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def compute_ious(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two sets of bounding boxes
    If box1 (1,7) & box2 (n,7), compare them one by one and return (n,7)
    If box1 (n,7) & box2 (n,7), compare them correspondingly and return (n,7)
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.

    Args:
        prediction: tensor (num_imgs, num_anchors, num_classes+5)
        conf_thres: discard all predictions with confidence below this value
        nms_thres: discard all predictions with IoU above this value

    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[...,:4] = xywh2xyxy(prediction[...,:4])

    output = [None for _ in range(len(prediction))] ## shd change to []?
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        
        # Label class by max probability
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        
        # Sort by score, which is object confidence times class confidence
        score = detections[:, 4] * detections[:, 5]
        detections = detections[(-score).argsort()]
        
        # Perform non-maximum suppression
        keep_boxes = []
        unique_classes = torch.unique(detections[:,-1])
        # Extract detections of the same class
        for c in unique_classes:
            cls_detections = detections[detections[:,-1] == c]
            while len(cls_detections) > 0:
                # Calculate IoU w.r.t. first item, and mark for removal if it's large
                large_iou_scores = compute_ious(cls_detections[0, :4].unsqueeze(0), cls_detections[:, :4]) >= nms_thres
                to_remove = cls_detections[large_iou_scores]
                # Merge the to-be-removed boxes weighted by their confidence
                weights = to_remove[:,4:5]
                merged_box = (weights * to_remove[:, :4]).sum(0) / weights.sum()
                cls_detections[0, :4] = merged_box
                # Remove candidate boxes
                keep_boxes.append(cls_detections[0])
                cls_detections = cls_detections[~large_iou_scores]

        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

# def diff_cls_nms_filter_conf(imgs_detections, conf_thres=0.5, nms_thres=0.4, sort_by='conf', xywh=True):
#     '''
#     Different class NMS for multiple image predictions
#     Args:
#         prediction: tensor (num_imgs, num_anchors, num_classes+5)
#         conf_thres: discard all predictions with confidence below this value
#         nms_thres: discard all predictions with IoU above this value
#         sort_by: sort filtered predictions by confidence or area
#     Returns detections with shape:
#         (x1, y1, x2, y2, object_conf, class_score, class_pred)
#     '''
#     # From (center x, center y, width, height) to (x1, y1, x2, y2)
#     if xywh:
#         prediction[...,:4] = xywh2xyxy(prediction[...,:4])
####
#### Not implementing because of unknown effects from merging all bboxes of different classes    

def diff_cls_nms(img_detections, nms_thres=0.4, sort_by='conf'):
    '''
    Different class NMS for one image prediction
    Inputs:
    - img_detections: list of np arrays [array(x1, y1, x2, y2, conf, cls_conf, cls)]
    '''

    if len(img_detections) == 0:
        return img_detections

    # Sort in descending order by confidence first
    if sort_by == 'conf':
        img_detections = sorted(img_detections, key=lambda x: x[4], reverse=True)
    elif sort_by == 'area':
        img_detections = sorted(img_detections, key=lambda x: compute_area(x), reverse=True)
    
    valid_img_detections = []    
    while len(img_detections) > 0:
        to_keep_idx = []
        box1 = img_detections[0] # highest conf box
        for i, box2 in enumerate(img_detections):
            iou = compute_iou(box1, box2)
            if iou < nms_thres:
                to_keep_idx.append(i)
        # remove boxes
        valid_img_detections.append(img_detections[0])
        img_detections = [img_detections[j] for j in to_keep_idx]

    return valid_img_detections

