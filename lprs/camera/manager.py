import time
import logging
from shapely.geometry import Polygon
from typing import List, Tuple, Dict, Optional
from .camera import Camera, CameraType
from ..utils.bbox import bbox_polygon_intersection, compute_area, compute_iou

class CameraManager:
    def __init__(self, config):
        self.num_votes = config['properties']['num_votes']
        self.new_car_time_patient = config['properties']['new_car_time_patient']
        self.new_car_iou_threshold = config['properties']['new_car_iou_threshold']
        self.cameras = self.init_cameras(config['cameras'])

    def get_camera(self, cam_ip) -> Camera:
        return self.cameras[cam_ip]["camera"]

    def start_cameras_streaming(self):
        """
        To start all the cameras
        """
        for cam_ip in self.cameras.keys():
            self.get_camera(cam_ip).start()

    def init_cameras(self, cameras_dict):
        """
        Construct a dictionary that stores all camera info
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
        for cam_name, v in cameras_dict.items():
            cam_ip = v['ip']
            cam_type = v['type']
            cam_fps_sim = v['fps_simulation']
            accum_time = v['accum_time']
            if cam_type == 'entrance':
                cam_type = CameraType.entrance
            # elif cam_type == 'top':
            #     cam_type = CameraType.top
            # elif cam_type == 'bot':
            #     cam_type = CameraType.bot
            else:
                logging.error(f'{cam_name}: Unimplemented cam_type ({cam_type})')
            trigger_zone = v.get('trigger_zone', None)
            cameras[cam_name] = {'camera': Camera(cam_name=cam_name,
                                                  cam_ip=cam_ip,
                                                  cam_type=cam_type,
                                                  num_votes=self.num_votes,
                                                  accum_time=accum_time,
                                                  cam_fps_sim=cam_fps_sim),
                               'trigger_zone': Polygon(trigger_zone),
                               'last_triggered_coords': None,
                               'last_triggered_time': 0}
        return cameras

    def get_all_fps(self):
        """
        Get all fps of all cameras
        return:
            all_fps (dict('cam_ip': float, ...))
        """
        all_fps = {}
        for cam_ip, camera_dict in self.cameras.items():
            all_fps[cam_ip] = camera_dict['camera'].get_fps()
        return all_fps

    def get_all_frames(self):
        """
        Get camera.new_frame and camera.accum_frames for all self.cameras.
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
            camera = self.get_camera(cam_ip)
            new_frame = camera.get_new_frame()
            accum_frames = camera.get_accumulated_frames()
            frames['new_frame'] = new_frame
            frames['accum_frames'] = accum_frames
            all_frames[cam_ip] = frames
        return all_frames

    def update_camera_trigger_status(self, all_car_locations):
        """
        Based on located car locations to justify if cameras need to start accumulating
        Args:
            all_car_locations (dict('cam_ip': [dict('coords': tuple(int x1, int y1, int x2, int y2), 'confidence': float), ...]))
        """
        for cam_ip, car_locations in all_car_locations.items():
            camera_dict = self.cameras[cam_ip]
            cam_name = camera_dict["camera"].cam_name
            trigger_zone = camera_dict['trigger_zone']
            triggered_coords = self.find_triggered_car_coords(trigger_zone, car_locations)
            if triggered_coords is None:
                continue
            
            if camera_dict['camera'].cam_type == CameraType.entrance:
                last_triggered_time = camera_dict['last_triggered_time']
                last_triggered_coords = camera_dict['last_triggered_coords']

                cur_time = time.time()
                camera_dict['last_triggered_coords'] = triggered_coords
                camera_dict['last_triggered_time'] = cur_time
                # if the time difference between this car and last trigger car is large
                if cur_time - last_triggered_time > self.new_car_time_patient:
                    camera_dict['camera'].start_accumulate()
                    logging.info(f'{cam_name}: TRIGGERED: time window == {(cur_time - last_triggered_time):.2f} seconds')
                    continue
                # if the iou between this car and last trigger car is large
                new_car_iou = compute_iou(triggered_coords, last_triggered_coords)
                if new_car_iou < self.new_car_iou_threshold:
                    camera_dict['camera'].start_accumulate()
                    logging.info(f'{cam_name}: TRIGGERED: iou == {new_car_iou:.2f}')
                    continue
            else:
                logging.warning('UNEXPECTED: Not implemented non-entrance trigger logic!')

    @staticmethod
    def find_triggered_car_coords(trigger_zone: Polygon, car_locations: List[Dict]) -> Optional[Tuple[int, int, int, int]]:
        """
        To find if there's any car's bbox touches trigger_zone. If multiple car do, return the max intersection one.
        args:
            tigger_zone (Polygon): the camera's trigger zone
            car_locations (list(dict('coords': tuple(int x1, int y1, int x2, int y2), 'confidence': float), ...))
        returns:
            triggered_coords (tuple(int x1, y1, x2, y2) / None): the coordinates of the car that has max iou with trigger zone 
        """
        max_intersection = 0.0
        triggered_coords = None
        for _, car in enumerate(car_locations):
            car_coords = car['coords']
            # if zone is unspecified
            if trigger_zone.is_empty:
                car_zone_intersect = compute_area(car_coords)
            else:
                car_zone_intersect = bbox_polygon_intersection(trigger_zone, car_coords)
            if car_zone_intersect > max_intersection:
                max_intersection = car_zone_intersect
                triggered_coords = car_coords
        return triggered_coords
