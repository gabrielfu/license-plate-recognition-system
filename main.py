import sys
import time

from .camera.manager import CameraManager
from .utils import read_yaml
from .models.utils import compute_area, majority_vote

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

        # TODO: handle batch size
        # Extract new frame for each camera
        new_frames = []
        missing_cam_idx = []
        for i, (ip, frames) in enumerate(all_frames.items()):
            frame = frames['new_frame']
            if frame is not None:
                new_frames.append()
            else:
                missing_cam_idx.append(i)

        # Batch predict car detection
        car_locations = car_locator.predict(new_frames)
        # Maintain same order between car_locations & all_frames.keys()
        for i in sorted(missing_cam_idx, reverse=True):
            car_locations.insert(i, [])
        
        # Update the trigger status of all cameras based on car locations
        all_car_locations = {
            ip: car for ip, car in zip(all_frames.keys(), car_locations)
        }
        camera_manager.update_camera_trigger_status(all_car_locations)
        
        # Predict license numbers
        # For each camera (as a batch)
        license_numbers = {}
        for ip, frames in all_frames.items():
            # Continue if no accum_frames
            accum_frames = frames['accum_frames']
            if accum_frames is None:
                continue
            
            lpr_output = lpr.predict([f for f in accum_frames if f is not None]) # handle any None frame in accum_frames
            plate_nums = []
            # For each frame, find the plate with largest area
            for frame in lpr_output:
                max_area = 0
                best_plate = None
                for plate in frame:
                    area = compute_area(plate['plate']['coords'])
                    if area > max_area:
                        max_area = area
                        best_plate = plate
                # Found a plate
                if best_plate is not None:
                    result = best_plate['plate_num']
                    plate_nums.append((result['numbers'], result['confidence']))
            # Perform majority vote & put into dict
            license_numbers[ip] = majority_vote(plate_nums)
            
        '''
        license_numbers == {
            '123.0.0.1': ('AB1234', 0.99),
            '123.0.0.2': ('CD5678', 0.88)
        }
        '''

        # @mike TODO:
        # Output the license number and correcponding camera ip, and maybe the coords(?) to kafka
        # For example:
        ## kafka_sender.send(license_numbers)
