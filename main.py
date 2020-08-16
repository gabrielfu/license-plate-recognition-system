import sys
import os
import time
import logging

sys.path.insert(0, os.getcwd())

from camera.manager import CameraManager
from utils.utils import read_yaml
from utils.bbox import compute_area
from logger_handlers import setup_logging

def exit_app():
    ''' Shut down the whole application'''
    sys.exit()

def majority_vote(ocr_results):
    '''
    Input:
    - ocr_results: list of tuples e.g. [('PV1954',0.99),('PV1954',0.97),('PV1934',0.91),...]

    Output:
    - tuple(num, conf) e.g. ('PV1954', 0.99)
    '''
    if not ocr_results:  # Empty
        return ''

    counter = {}
    license_num_prob = {}
    for license_num, min_conf in ocr_results:
        # Count number of votes
        counter[license_num] = counter.get(license_num, 0) + 1

#             if license_num not in license_num_max_prob:
#                 license_num_max_prob[license_num] = avg_conf
#             elif avg_conf > license_num_max_prob[license_num]:
#                 license_num_max_prob[license_num] = avg_conf
        if license_num not in license_num_prob:
            license_num_prob[license_num] = [min_conf]
        else:
            license_num_prob[license_num].append(min_conf)
    
    license_num_prob = {num:(sum(scores)/len(scores)) for num, scores in license_num_prob.items()}
    # Unqiue majority --> output major result, Multi/No majority --> output highest avg_conf result
    major_candidates = [lic for lic, count in counter.items() if count == max(counter.values())]
    major_candidates_conf = {lic:license_num_prob[lic] for lic in major_candidates}
    lic_num, conf = max(major_candidates_conf.items(), key=lambda x: x[1])
    return lic_num, conf

if __name__ == '__main__':

    # Read configs
    app_cfg = read_yaml('config/app.yaml')
    cameras_cfg = read_yaml('config/cameras.yaml')
    models_cfg = read_yaml('config/models.yaml')
    logger_cfg = read_yaml('config/logger.yaml')

    # Setup logging handlers & initialize logger
    setup_logging(logger_cfg)
    logger = logging.getLogger(__name__)
    logger.info('-------------------------')
    logger.info('Starting Application...')

    # Import & initialize models
    use_trt = app_cfg['car_locator']['trt']
    logger.info(f'Initializing Car Locator... (TensorRT={use_trt})')
    try:
        if use_trt:
            from models.car_locator_trt import CarLocatorTRT
            car_locator = CarLocatorTRT(models_cfg['car_locator_trt'])
        else:
            from models.car_locator import CarLocator
            car_locator = CarLocator(models_cfg['car_locator'])
    except:
        logger.exception('Failed to initialize Car Locator!')

    logger.info('Initializing LPR...')
    try:
        from models.lpr import LPR
        lpr = LPR(models_cfg)
    except:
        logger.exception('Failed to initialize LPR!')


    # Initialize cameras and start streaming
    logger.info('Streaming cameras...')
    try:
        camera_manager = CameraManager(cameras_cfg)
        camera_manager.start_cameras_streaming()
    except:
        logger.exception('Failed to stream camera!')

    time.sleep(5)

    # Run application
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
                new_frames.append(frame)
            else:
                missing_cam_idx.append(i)

        # Batch predict car detection
        try:
            car_locations = car_locator.predict(new_frames)
            # Maintain same order between car_locations & all_frames.keys()
            for i in sorted(missing_cam_idx, reverse=True):
                car_locations.insert(i, [])
        except:
            logger.exception('Failed to predict car detection')
        
        # Update the trigger status of all cameras based on car locations
        try:
            all_car_locations = {
                ip: car for ip, car in zip(all_frames.keys(), car_locations)
            }
            camera_manager.update_camera_trigger_status(all_car_locations)
        except:
            logger.exception('Error when triggering cameras')
        
        # Predict license numbers
        # For each camera (as a batch)
        license_numbers = {}
        for ip, frames in all_frames.items():
            try:
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
            except:
                logger.exception(f'{ip}: Error in lpr prediction')
            
        '''
        license_numbers == {
            '123.0.0.1': ('AB1234', 0.99),
            '123.0.0.2': ('CD5678', 0.88)
        }
        '''
        if license_numbers:
            logger.info(f'LPR result: {license_numbers}')

        # @mike TODO:
        # Output the license number and correcponding camera ip, and maybe the coords(?) to kafka
        # For example:
        ## kafka_sender.send(license_numbers)
