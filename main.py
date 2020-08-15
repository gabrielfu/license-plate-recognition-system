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
        car_locator = CarLocator(models_cfg['car_locator'])

    from models.lpr import LPR
    lpr = LPR(models_cfg)

    # Initialize cameras and start streaming
    camera_manager = CameraManager(cameras_cfg)
    camera_manager.start_cameras_streaming()

    time.sleep(5)

    while True:
        # Get all the frames for prediction
        all_frames = camera_manager.get_all_frames()

        # @fu TODO:
        # Get all car locations in all_frames[cam_ip]['new_frame']
        # For example: (see expected type of all_car_locations in CameraManager.update_camera_trigger_status)
        ## all_car_locations = car_locator.get_locations(all_frames)

        # Update the trigger status of all cameras based on car locations
        camera_manager.update_camera_trigger_status(all_car_locations)

        # @fu TODO:
        # Get license number in all_frames[cam_ip]['accum_frames']
        # the largest plate/ inside camera_manager.cameras[cam_ip]['last_triggered_coords']. To be decided
        # For example: (license_numbers should be a dict(cam_ip: dict(plate_num: str, confidence: float, coords: tuple), ...))
        ## license_numbers = lpr.get_plate_num(all_frames)
        ## license_numbers = lpr.get_plate_num(all_frames, camera_manager.cameras)

        # @mike TODO:
        # Output the license number and correcponding camera ip, and maybe the coords(?) to kafka
        # For example:
        ## kafka_sender.send(license_numbers)



