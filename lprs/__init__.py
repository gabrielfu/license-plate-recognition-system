import os
import sys
sys.path.insert(0, os.getcwd())
import time
import logging
from collections import defaultdict, OrderedDict
from typing import Dict

from lprs.camera.manager import CameraManager
from lprs.sender import KafkaSender
from lprs.utils.utils import read_yaml
from lprs.utils.bbox import compute_area
from lprs.utils.majority_vote import majority_vote
from lprs.logger import setup_logging
from lprs.models import LPR


def exit_app():
    """ Shut down the whole application"""
    logging.info('Shutting down application')
    sys.exit()


def init_car_locator(models_cfg, use_trt):
    """ Import & initialize Car Locator """
    logging.info(f'Initializing Car Locator... (TensorRT={use_trt["car_locator"]})')
    if use_trt["car_locator"]:
        from lprs.models.trt import CarLocatorTRT
        car_locator = CarLocatorTRT(models_cfg['car_locator_trt'])
        car_batch_size = int(models_cfg['car_locator_trt']['max_batch_size'])
    else:
        from lprs.models import CarLocator
        car_locator = CarLocator(models_cfg['car_locator'])
        car_batch_size = int(models_cfg['car_locator']['batch_size'])
    return car_locator, car_batch_size


def fixed_batch_car_locator(all_frames, car_locator, car_batch_size, camera_manager):
    """ Wrapper function to do fixed batch car location detection """
    # Extract new frame for each camera
    new_frames = []
    cam_ips = []
    for ip, frames in all_frames.items():
        frame = frames['new_frame']
        if frame is not None:
            new_frames.append(frame)
            cam_ips.append(ip)
    # Fixed batch predict car detection
    for i in range(0, len(new_frames), car_batch_size):
        i_end = min(i+car_batch_size, len(new_frames))
        try:
            car_locations = car_locator.predict(new_frames[i:i_end], sort_by='conf')
        except Exception:
            logging.exception(f'{cam_ips[i:i_end]}: Failed to predict car detection')
            continue
        # Update the trigger status of each batch cameras based on car locations
        try:
            all_car_locations = {
                ip: car for ip, car in zip(cam_ips[i:i_end], car_locations)
            }
            camera_manager.update_camera_trigger_status(all_car_locations)
        except Exception:
            logging.exception(f'{cam_ips[i:i_end]}: Error when triggering cameras')
            continue


def single_img_car_locator(all_frames, car_locator, car_batch_size, camera_manager):
    """ Wrapper function to do single img car location detection """
    # all_car_locations = {}
    for ip, frames in all_frames.items():
        # Extract new frame for each camera
        frame = frames['new_frame']
        if frame is None:
            continue
        # Single img prediction
        try:
            car = car_locator.predict([frame], sort_by='conf')[0]
            car_locations = {ip: car}
        except Exception:
            logging.exception(f'{ip}: Failed to predict car detection')
            continue

        # Update the trigger status of all cameras based on car locations
        try:
            camera_manager.update_camera_trigger_status(car_locations)
        except Exception:
            logging.exception(f'{ip}: Error when triggering cameras')
            continue


def license_plate_recognition(lpr: LPR, all_frames: Dict):
    """
    Args:
        lpr (LPR): LPR instance
        all_frames (Dict): list of frames from camera manager

    Returns:
        license_numbers == {
            '123.0.0.1': {
                plate_num: AB1234,
                confidence: 0.99,
                image: <np.array>
            },
            '123.0.0.2': {
                plate_num: CD5678,
                confidence: 0.88,
                image: <np.array>
            },
        }
    """
    # Predict license numbers
    # For each camera (as a batch)
    license_numbers = {}
    num_lpr_predict = 0

    for ip, frames in all_frames.items():
        try:
            # Continue if no accum_frames
            accum_frames = frames['accum_frames']
            if accum_frames is None:
                continue
            num_lpr_predict += 1
            # handle any None frame in accum_frames
            accum_frames = [f for f in accum_frames if f is not None]
            if len(accum_frames) == 0:
                continue
            # Do prediction
            lpr_output = lpr.predict(accum_frames)
            # For each frame, find the plate with largest area
            plate_nums = []
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
            # Perform majority vote
            plate_num, conf = majority_vote(plate_nums)
            # Put into dict
            license_numbers[ip] = {
                'plate_num': plate_num,
                'confidence': conf,
                'image': accum_frames[0]
            }
        except Exception:
            logging.exception(f'{ip}: Error in lpr prediction')

    return license_numbers, num_lpr_predict


def _run():

    #####################################
    ###       App Initialization      ###
    #####################################
    # Read configs
    app_cfg = read_yaml('./configs/app.yaml')
    cameras_cfg = read_yaml('./configs/cameras.yaml')
    models_cfg = read_yaml('./configs/models.yaml')
    logger_cfg = read_yaml('./configs/logger.yaml')
    kafka_cfg = read_yaml('./configs/kafka.yaml')

    # Setup logging handlers & initialize logger
    os.makedirs(logger_cfg['log_dir'], exist_ok=True)
    setup_logging(logger_cfg)
    logging.info('-------------------------------------------------------')
    logging.info('Starting Application...')
    
    # Print loop time every x loops
    print_every = int(app_cfg['app'].get('print_time_every_loops', 0))
    
    # Check if use trt or not
    # Needs to initialize all torch models before initializing trt models
    # Otherwise, due to cuda context issues, the models will predict None all the time
    # So, sort use_trt by values so that False's are in the front
    use_trt = OrderedDict(sorted(app_cfg['use_trt'].items(), key=lambda x: x[1], reverse=False))

    # Validate majority vote setting for TRT model
    if use_trt['plate_detector']:
        if models_cfg['plate_detector_trt']['max_batch_size'] < cameras_cfg['properties']['num_votes']:
            raise Exception(f"Number of majority votes ({cameras_cfg['properties']['num_votes']}) "
                            f"is smaller than maximum batch size of "
                            f"PlateDetectorTRT ({models_cfg['plate_detector_trt']['max_batch_size']})")
            

    if use_trt['car_locator']: # need to initialize LPR first
        lpr = LPR(models_cfg, use_trt)
        car_locator, car_batch_size = init_car_locator(models_cfg, use_trt)
    else: # need to initialize Car Locator first
        car_locator, car_batch_size = init_car_locator(models_cfg, use_trt)
        lpr = LPR(models_cfg, use_trt)
        
    # Initialize cameras and start frame streaming
    logging.info('Starting cameras...')
    camera_manager = CameraManager(cameras_cfg)
    camera_manager.start_cameras_streaming()

    # Initialize kafka sender and start output streaming
    logging.info('Starting Kafka sender...')
    sender = KafkaSender(kafka_cfg)
    sender.start_kafka_streaming()

    # Sleep to let other services finish up their init
    time.sleep(app_cfg['app']['sleep_after_init'])

    #####################################
    ###          Start Loop           ###
    #####################################   
    loop_count = defaultdict(int)
    loop_time_ttl = defaultdict(float)

    while True:
        loop_start = time.time()
        # Get all the frames for prediction
        all_frames = camera_manager.get_all_frames()


        # Car Detection
        if car_batch_size > 1:
            fixed_batch_car_locator(all_frames, car_locator, car_batch_size, camera_manager)
        else:
            single_img_car_locator(all_frames, car_locator, car_batch_size, camera_manager)


        # License Plate Recognition
        license_numbers, num_lpr_predict = license_plate_recognition(lpr, all_frames)


        # Output with Kafka
        if license_numbers:
            logging.info(f'LPR RESULT: {license_numbers}')
            try:
                sender.send(license_numbers)
            except Exception:
                logging.critical("Sender failed to send LPR results!")


        # Timer
        loop_time = time.time() - loop_start
        if loop_time < 0.01: # doesn't count since it's likely empty loop
            continue
        loop_time_ttl[num_lpr_predict] += loop_time
        loop_count[num_lpr_predict] += 1
        if loop_count[0] >= print_every > 0:
            logging.info('***************Speed evaluation******************')
            for num_lpr_predict in loop_count.keys():
                avg_time = loop_time_ttl[num_lpr_predict] / (loop_count[num_lpr_predict]+1e-16)
                logging.info(f'[num_lpr_prediction in loop: {num_lpr_predict}] {loop_count[num_lpr_predict]}-loop average time: {avg_time:.2f} s')
            all_fps = camera_manager.get_all_fps()
            logging.info(f'Camera fps: {all_fps}')
            loop_count = defaultdict(float)
            loop_time_ttl = defaultdict(int)


def run():
    try:
        _run()
    except Exception as e:
        logging.exception(e)
        exit_app()


if __name__ == '__main__':
    run()
