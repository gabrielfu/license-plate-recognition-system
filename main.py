import sys
import time

from .camera.manager import CameraManager
from .utils import read_yaml

def exit_app():
    ''' Shut down the whole application'''
    sys.exit()

if __name__ == '__main__':

    # Read configs
    app_cfg = read_yaml('config/app.yaml')
    cameras_cfg = read_yaml('config/cameras.yaml')
    models_cfg = read_yaml('config/models.yaml')

    # Import & initialize models
    if app_cfg['car_locator']['trt']:
        from models.car_locator_trt import CarLocatorTRT
        car_locator = CarLocatorTRT(models_cfg['car_locator_trt'])
    else:
        from models.car_locator import CarLocator
        car_locator = CarLocatorTRT(models_cfg['car_locator'])

    from models.lpr import LPR
    lpr = LPR(models_cfg)

    # Initialize cameras
    camera_manager = CameraManager(cameras_cfg)

    # Start app
    camera_manager.start_cameras_streaming()
    time.sleep(5)

    while True:
        all_frames = camera_manager.get_all_frames()
        pass

