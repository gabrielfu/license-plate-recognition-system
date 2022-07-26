import sys
import os
import yaml

def read_yaml(path):
    return yaml.load(open(path, 'r'), Loader=yaml.FullLoader)

def get_correct_path(relative_path):
    """
    Used when packaged app with PyInstaller
    To find external paths outside of the packaged app
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names
