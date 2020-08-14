from shapely.geometry import Polygon
import yaml

def read_yaml(path):
    return yaml.load(open(path, 'r'), Loader=yaml.FullLoader)

def dot_inside_bbox(dot, bbox):
    '''
    dot: (x,y)
    bbox: (x1, y1, x2, y2)
    '''
    x1,y1,x2,y2 = bbox
    dot_x, dot_y = dot
    if x1 < dot_x and x2 > dot_x and y1 < dot_y and y2 > dot_y:
        return True
    else:
        return False

def polygons_intersect(trigger_zone, bbox):
    '''
    trigger_zone: [bot_left, bot_right, top_right, top_left], all in tuple(x,y)
    bbox: (x1, y1, x2, y2)
    '''
    # bot_left, bot_right, top_right, top_left
    trigger_polygon = Polygon(trigger_zone)

    x1, y1, x2, y2 = bbox
    bbox_polygon = Polygon([(x1, y2), (x2,y2), (x2,y1), (x1, y1)])
    return bbox_polygon.intersects(trigger_polygon)  # True/False