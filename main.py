import sys
import os
import time
import logging
import collections

sys.path.insert(0, os.getcwd())

from camera.manager import CameraManager
from sender.sender import KafkaSender
from utils.utils import read_yaml
from utils.bbox import compute_area
from logger import setup_logging

import pycuda.autoinit

def exit_app():
    ''' Shut down the whole application'''
    logging.info('Shutting down application')
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

def init_LPR(use_trt):
    ''' Import & initialize LPR '''
    try:
        from models.lpr import LPR
        lpr = LPR(models_cfg, use_trt)
    except:
        logging.critical('Failed to initialize LPR!')
        exit_app()
    return lpr

def init_car_locator(use_trt):
    ''' Import & initialize Car Locator '''
    logging.info(f'Initializing Car Locator... (TensorRT={use_trt["car_locator"]})')
    try:
        if use_trt["car_locator"]:
            from models.car_locator_trt import CarLocatorTRT
            car_locator = CarLocatorTRT(models_cfg['car_locator_trt'])
            car_batch_size = int(models_cfg['car_locator_trt']['max_batch_size'])
        else:
            from models.car_locator import CarLocator
            car_locator = CarLocator(models_cfg['car_locator'])
            car_batch_size = int(models_cfg['car_locator']['batch_size'])
    except:
        logging.critical('Failed to initialize Car Locator!')
        exit_app()
    return car_locator, car_batch_size

def fixed_batch_car_locator(all_frames, car_locator, car_batch_size, camera_manager):
    ''' Wrapper function to do fixed batch car location detection '''
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
        except KeyboardInterrupt:
            logging.info('Keyboard Interrupt')
            exit_app()
        except:
            logging.exception(f'{cam_ips[i:i_end]}: Failed to predict car detection')
        # Update the trigger status of each batch cameras based on car locations
        try:
            all_car_locations = {
                ip: car for ip, car in zip(cam_ips[i:i_end], car_locations)
            }
            camera_manager.update_camera_trigger_status(all_car_locations)
        except KeyboardInterrupt:
            logging.info('Keyboard Interrupt')
            exit_app()
        except:
            logging.exception(f'{cam_ips[i:i_end]}: Error when triggering cameras')

def single_img_car_locator(all_frames, car_locator, car_batch_size, camera_manager):
    ''' Wrapper function to do single img car location detection '''
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
        except KeyboardInterrupt:
            logging.info('Keyboard Interrupt')
            exit_app()
        except:
            logging.exception(f'{ip}: Failed to predict car detection')
            continue

        # Update the trigger status of all cameras based on car locations
        try:
            camera_manager.update_camera_trigger_status(car_locations)
        except KeyboardInterrupt:
            logging.info('Keyboard Interrupt')
            exit_app()
        except:
            logging.exception(f'{ip}: Error when triggering cameras')
            continue

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
    
    # Setup logging handlers & initialize logger
    os.makedirs(logger_cfg['log_dir'], exist_ok=True)
    setup_logging(logger_cfg)
    logging.info('-------------------------------------------------------')
    logging.info('Starting Application...')
    
    # Print loop time every x loops
    print_every = int(app_cfg['app']['print_time_every_loops'])
    
    # Check if use trt or not
    # use_trt is sorted by values so that False's are in the front
    # use_trt = OrderedDict(sorted(app_cfg['use_trt'].items(), key=lambda x: x[1], reverse=False))
    use_trt = app_cfg['use_trt']
    if use_trt['plate_detector']:
        if models_cfg['plate_detector_trt']['max_batch_size'] < cameras_cfg['properties']['num_votes']:
            logging.critical(f"Number of majority votes ({cameras_cfg['properties']['num_votes']}) is smaller than maximum batch size of PlateDetectorTRT ({models_cfg['plate_detector_trt']['max_batch_size']})")
            exit_app()
            
    '''
    Needs to initialize all torch models before initializing trt models
    Otherwise, cuda context issues, or models will predict None all the time
    '''
    if use_trt['car_locator']: # need to initialize LPR first
        lpr = init_LPR(use_trt)
        car_locator, car_batch_size = init_car_locator(use_trt)
    else: # need to initialize Car Locator first
        car_locator, car_batch_size = init_car_locator(use_trt)
        lpr = init_LPR(use_trt)
    
    # Create wrapper function to handle config car locator batch size, so that it's not checked in every loop
    if car_batch_size > 1:
        car_locator_wrapper = fixed_batch_car_locator
    else:
        car_locator_wrapper = single_img_car_locator
        
    # Initialize cameras and start frame streaming
    logging.info('Starting cameras...')
    try:
        camera_manager = CameraManager(cameras_cfg)
        camera_manager.start_cameras_streaming()
    except KeyboardInterrupt:
        logging.info('Keyboard Interrupt')
        exit_app()
    except:
        logging.critical('Failed to start camera!')
        exit_app()

    # Initialize kafka sender and start output streaming
    logging.info('Starting Kafka sender...')
    try:
        sender = KafkaSender(kafka_cfg)
        sender.start_kafka_streaming()
    except KeyboardInterrupt:
        logging.info('Keyboard Interrupt')
        exit_app()
    except:
        logging.critical('Failed to start Kafka sender!')
        exit_app()
        
    try:
        time.sleep(app_cfg['app']['sleep_after_init'])
    except:
        pass

    #####################################
    ###          Start Loop           ###
    #####################################   
    loop_count = collections.defaultdict(float)
    loop_time_ttl = collections.defaultdict(float)
    while True:
        loop_start = time.time()
        # Get all the frames for prediction
        all_frames = camera_manager.get_all_frames()

        #####################################
        ###         Car detection         ###
        #####################################      
        car_locator_wrapper(all_frames, car_locator, car_batch_size, camera_manager)  

        #####################################
        ###      License Recognition      ###
        #####################################
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
            except KeyboardInterrupt:
                logging.info('Keyboard Interrupt')
                exit_app()
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
            logging.info('LPR RESULT: {\n'+'\n'.join([repr(k)+':'+repr(v) for k,v in license_numbers.items()])+'\n}')     
            try:
                sender.send(license_numbers)
            except KeyboardInterrupt:
                logging.info('Keyboard Interrupt')
                exit_app()
            except:
                logging.critical("Sender failed to send LPR results!")
        
        loop_time = time.time() - loop_start
        if loop_time < 0.01: # doesn't count since it's likely empty loop
            continue
        loop_time_ttl[num_lpr_predict] += loop_time
        loop_count[num_lpr_predict] += 1
        
        if loop_count[0] >= print_every:
            for num_lpr_predict in loop_count.keys():                
                avg_time = loop_time_ttl[num_lpr_predict] / (loop_count[num_lpr_predict]+1e-16)
                logging.info(f'{loop_count[num_lpr_predict]}-loop average time: {avg_time:.2f} s')
            all_fps = camera_manager.get_all_fps()
            logging.info(f'Camera fps: {all_fps}')
            loop_count = collections.defaultdict(float)
            loop_time_ttl = collections.defaultdict(float)







