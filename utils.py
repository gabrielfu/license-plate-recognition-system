import sys
import os
import yaml

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