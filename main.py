import sys

from .utils import read_yaml

def exit_app():
    ''' Shut down the whole application'''
    sys.exit()

if __name__ == '__main__':

    # Read configs
    app_cfg = read_yaml('config/app.yaml')
    camera_cfg = read_yaml('config/cameras.yaml')
    model_cfg = read_yaml('config/models.yaml')

    # Import Car Locator model
    if app_cfg['car_locator']['trt']:
        from models.car_locator_trt import CarLocatorTRT as CarLocator
    else:
        from models.car_locator import CarLocator
    
    # Import LPR model
    from models.lpr import LPR
