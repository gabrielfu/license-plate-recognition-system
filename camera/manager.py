import time
import logging
from shapely.geometry import Polygon, box
from .camera import Camera, CameraType
from utils.bbox import compute_iou, bbox_polygon_iou

class CameraManager:
    def __init__(self, config):
        self.num_votes = config['properties']['num_votes']
        self.new_car_time_patient = config['properties']['new_car_time_patient']
        self.new_car_iou_threshold = config['properties']['new_car_iou_threshold']
        self.cameras = self.init_cameras(config['cameras'])

    def start_cameras_streaming(self):
        """To start all the cameras
        """
        for cam_ip in self.cameras.keys():
            self.cameras[cam_ip]['camera'].start()

    def init_cameras(self, cameras_dict):
        """Construct a dictionary that stores all camera info
        returns:
            cameras (dict('cam_ip': dict('camera': Camera, 'trigger_zone': Polygon, 'last_triggered_coords': tuple, 
            'last_triggered_time': datetime), ...)):
            example:
            {'128.1.1.0': {'camera': Camera(), 'trigger_zone': Polygon([(1,2), (3,4), (5,6), (7,8)]), 'last_triggered_coords': (1,1,2,2), 
                           'last_triggered_time': 20155654}
             ...
            }
        """
        cameras = {}
        for _, v in cameras_dict.items():
            cam_ip = v['ip']
            cam_type = v['type']
            if cam_type == 'entrance':
                cam_type = CameraType.entrance
            # elif cam_type == 'top':
            #     cam_type = CameraType.top
            # elif cam_type == 'bot':
            #     cam_type = CameraType.bot
            else:
                logging.error('Unimplemented cam_type: {}'.format(cam_type))
            trigger_zone = v['trigger_zone']
            cameras[cam_ip] = {'camera': Camera(cam_ip, cam_type, self.num_votes),
                               'trigger_zone': Polygon(trigger_zone),
                               'last_triggered_coords': None,
                               'last_triggered_time': 0}
        return cameras

    def get_all_frames(self):
        """Get camera.new_frame and camera.accum_frames for all self.cameras.
        return:
            all_frames (dict('cam_ip': dict(new_frame: None/np.array(h*w*c), accum_frames: None/np.array(num_votes*h*w*c)), ...)): 
            example:
            {'128.1.1.0': {'new_frame': np.array(h*w*c), 'accum_frames': np.array(num_votes*h*w*c)}
             '128.1.1.1': {'new_frame': np.array(h*w*c), 'accum_frames': None}
             '128.1.1.2': {'new_frame': None, 'accum_frames': None}
            ...
            }
        """
        all_frames = {}
        for cam_ip in self.cameras.keys():
            frames = {}
            camera = self.cameras[cam_ip]['camera']
            new_frame = camera.get_new_frame()
            accum_frames = camera.get_accumulated_frames()
            frames['new_frame'] = new_frame
            frames['accum_frames'] = accum_frames
            all_frames[cam_ip] = frames
        return all_frames

    def update_camera_trigger_status(self, all_car_locations):
        """Based on located car locations to justify if cameras need to start accumulating
        Args:
            all_car_locations (dict('cam_ip': [dict('coords': tuple(int x1, int y1, int x2, int y2), 'confidence': float), ...]))
        """
        for cam_ip, car_locations in all_car_locations.items():
            camera_dict = self.cameras[cam_ip]
            trigger_zone = camera_dict['trigger_zone']
            triggered_coords = self.find_triggered_car_coords(trigger_zone, car_locations)
            if triggered_coords is None:
                continue
            
            if camera_dict['camera'].cam_type == CameraType.entrance:
                last_triggered_time = camera_dict['last_triggered_time']
                cur_time = time.time()
                last_triggered_coords = camera_dict['last_triggered_coords']
                camera_dict['last_triggered_coords'] = triggered_coords
                camera_dict['last_triggered_time'] = cur_time
                # if this is the first time trigger
                if camera_dict['last_triggered_coords'] is None:
                    camera_dict['camera'].start_accumulate()
                    logging.debug(f'{cam_ip}: Trigger type: first time trigger')
                    continue
                # if the time difference between this car and last trigger car is large
                if cur_time - last_triggered_time > self.new_car_time_patient:
                    camera_dict['camera'].start_accumulate()
                    logging.debug(f'{cam_ip}: Trigger type: time window trigger: {(cur_time - last_triggered_time):.2f} seconds')
                    continue
                # if the iou between this car and last trigger car is large
                new_car_iou = compute_iou(triggered_coords, last_triggered_coords)
                if new_car_iou < self.new_car_iou_threshold:
                    camera_dict['camera'].start_accumulate()
                    logging.debug(f'{cam_ip}: Trigger type: iou trigger: {new_car_iou:.2f}')
                    continue
            else:
                logging.warning('UNEXPECTED: Not implemented non-entrance trigger logic!')

    @staticmethod
    def find_triggered_car_coords(trigger_zone, car_locations):
        """To find if there's any car's bbox touches trigger_zone. If multiple car do, return the max iou one.
        args:
            tigger_zone (Polygon): the camera's trigger zone
            car_locations (list(dict('coords': tuple(int x1, int y1, int x2, int y2), 'confidence': float), ...))
        returns:
            triggered_coords (tuple(int x1, y1, x2, y2) / None): the coordinates of the car that has max iou with trigger zone 
        """
        max_iou = 0
        triggered_coords = None
        for _, car in enumerate(car_locations):
            car_coords = car['coords']
            car_zone_iou = bbox_polygon_iou(trigger_zone, car_coords)
            if car_zone_iou > max_iou:
                max_iou = car_zone_iou
                triggered_coords = car_coords
        return triggered_coords
