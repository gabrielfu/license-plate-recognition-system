import sys
import os
import time
import logging

sys.path.insert(0, os.getcwd())

from camera.manager import CameraManager
from sender.sender import KafkaSender
from utils.utils import read_yaml
from utils.bbox import compute_area
from logger import setup_logging

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
        return 'Recognition fail', None

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

    #####################################
    ###       App Initialization      ###
    #####################################
    # Read configs
    app_cfg = read_yaml('config/app.yaml')
    cameras_cfg = read_yaml('config/cameras.yaml')
    models_cfg = read_yaml('config/models.yaml')
    logger_cfg = read_yaml('config/logger.yaml')
    kafka_cfg = read_yaml('config/kafka.yaml')
    
    car_batch_size = int(models_cfg['car_locator']['batch_size'])

    # Setup logging handlers & initialize logger
    log_dir = logger_cfg['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setup_logging(logger_cfg)
    logging.info('-------------------------------------------------------')
    logging.info('Starting Application...')

    # Import & initialize models
    use_trt = app_cfg['car_locator']['trt']
    logging.info(f'Initializing Car Locator... (TensorRT={use_trt})')
    try:
        if use_trt:
            from models.car_locator_trt import CarLocatorTRT
            car_locator = CarLocatorTRT(models_cfg['car_locator_trt'])
        else:
            from models.car_locator import CarLocator
            car_locator = CarLocator(models_cfg['car_locator'])
    except:
        logging.exception('Failed to initialize Car Locator!')
        exit_app()

    logging.info('Initializing LPR...')
    try:
        from models.lpr import LPR
        lpr = LPR(models_cfg)
    except:
        logging.exception('Failed to initialize LPR!')
        exit_app()


    # Initialize cameras and start frame streaming
    logging.info('Streaming cameras...')
    try:
        camera_manager = CameraManager(cameras_cfg)
        camera_manager.start_cameras_streaming()
    except:
        logging.exception('Failed to stream camera!')
        exit_app()

    # Initialize kafka sender and start output streaming
    logging.info('Streaming Kafka...')
    try:
        sender = KafkaSender(kafka_cfg)
        sender.start_kafka_streaming()
    except:
        logging.exception('Failed to stream Kafka!')
        exit_app()
    time.sleep(5)

    while True:
        # Get all the frames for prediction
        all_frames = camera_manager.get_all_frames()

    #####################################
    ###         Car detection         ###
    #####################################        
        #############
        # If Fixed batch prediction
        ############
        if car_batch_size > 1:
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
                try:
                    car_locations = car_locator.predict(
                        new_frames[i:min(i+car_batch_size, len(new_frames))],
                        sort_by='conf')
                
                except:
                    logging.exception('Failed to predict car detection')
            
                # Update the trigger status of each batch cameras based on car locations
                try:
                    all_car_locations = {
                        ip: car for ip, car in zip(cam_ips[i:min(i+car_batch_size, len(cam_ips))], car_locations)
                    }
                    camera_manager.update_camera_trigger_status(all_car_locations)
                except:
                    logging.exception('Error when triggering cameras')

        #############
        # Else, Single img prediction
        ############
        else:
            # all_car_locations = {}
            for i, (ip, frames) in enumerate(all_frames.items()):
                # Extract new frame for each camera
                frame = frames['new_frame']
                if frame is None:
                    continue
                # Single img prediction
                try:
                    car = car_locator.predict([frame], sort_by='conf')[0]
                    car_locations = {ip: car}
                except:
                    logging.exception(f'{ip}: Failed to predict car detection')
                    continue

                # Update the trigger status of all cameras based on car locations
                try:
                    camera_manager.update_camera_trigger_status(car_locations)
                except:
                    logging.exception('Error when triggering cameras')
                    continue


    #####################################
    ###      License Recognition      ###
    #####################################
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
                logging.exception(f'{ip}: Error in lpr prediction')
            
        '''
        license_numbers == {
            '123.0.0.1': ('AB1234', 0.99),
            '123.0.0.2': ('CD5678', 0.88)
        }
        '''
    
        #####################################
        ###       Output with Kafka       ###
        #####################################
        if license_numbers:
            logging.info('LPR result: {\n'+'\n'.join([repr(k)+':'+repr(v) for k,v in license_numbers.items()])+'\n}')     
            try:
                sender.send(license_numbers)
            except:
                logging.exception(f'Error in sender')

