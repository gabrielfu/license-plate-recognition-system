import yaml
from .camera import Camera, CameraType
import time

class CameraManager:
    def __init__(self, config):
        self.num_votes = config['properties']['num_votes']
        self.cameras = self.init_cameras(config['cameras'])

    def start_cameras_streaming(self):
        """To start all the cameras
        """
        for cam_ip in self.cameras.keys():
            self.camerasp[cam_ip]['camera'].start()

    def init_cameras(cameras_dict):
        """Construct a dictionay that stores all camera info
        returns:
            cameras (dict('cam_ip': dict('camera': Camera, 'trigger_zone': list of tuples, 'last_triggered_coords': tuple, 
            'last_triggered_time': datetime), ...)):
            example:
            {'128.1.1.0': {'camera': Camera(), 'trigger_zone': [(1,2), (3,4), (5,6), (7,8)], 'last_triggered_coords': (1,1,2,2), 
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
                raise NotImplementedError('Unkown cam_type: {}'.format(cam_type))
            trigger_zone = v['trigger_zone']
            cameras[cam_ip] = {'camera': Camera(cam_ip, cam_type, self.num_votes),
                               'trigger_zone': trigger_zone,
                               'last_triggered_coords': None,
                               'last_triggered_time': time.time()}
        return cameras

    def get_all_frames(self):
        """Get Camera.new_frame and Camera.accum_frames for all self.cameras.
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

    @staticmethod
    def find_triggered_car_coords(trigger_zone, car_locations):
        """To find if there's any car's bbox touches trigger_zone. If multiple car do, return the max iou one.
        args:
            tigger_zone (list(tuples(x, y))): the camera's trigger zone
            car_locations (list(dict('coords': tuple(int x1, int y1, int x2, int y2), 'confidence': float), ...))
        returns:
            triggered_coords (tuple(int x1, y1, x2, y2) / None): the coordinates of the car that has max iou with trigger zone 
        """
        max_iou = 0
        triggered_coords = None
        for i, car in enumerate(car_locations):
            car_coords = car['coords']
            car_zone_iou = self.cal_car_zone_iou(trigger_zone, car_coords)
            if car_zone_iou > max_iou:
                max_iou = car_zone_iou
                triggered_coords = car_coords
        return triggered_coords

    @staticmethod
    def cal_car_zone_iou(trigger_zone, car_coords):
        pass

    @staticmethod
    def cal_car_car_iou(car_coords, car_coords):
        pass

    def trigger(self, cam_ip):
        self.cameras['cam_ip']['camera'].start_accumulate()
        self.cameras['cam_ip']['last_triggered_coords'] = triggered_coords
        self.cameras['cam_ip']['last_triggered_time'] = time.time()

    def update_camera_trigger_status(self, all_car_locations):
        """Based on located car locations to justify if cameras need to start accumulating
        Args:
            all_car_locations (dict('cam_ip': [dict('coords': tuple(int x1, int y1, int x2, int y2), 'confidence': float), ...]))
        """
        for cam_ip in self.cameras.keys():
            car_locations = all_car_locations[cam_ip]
            trigger_zone = self.cameras['cam_ip']['trigger_zone']
            triggered_coords = self.find_triggered_car_coords(car_locations, trigger_zone)
            
            if self.cameras['cam_ip']['camera'].cam_type == CameraType.entrance:
                if triggered_coords is None:
                    continue
                if self.cameras['cam_ip']['last_triggered_coords'] is None:
                    # if this is the first time trigger
                    self.trigger(cam_ip)
                    continue
                if self.cameras['cam_ip']['last_triggered_time'] - time.time() > self.new_car_time_patient:
                    # if the time difference between this car and last trigger car is large
                    self.trigger(cam_ip)
                    continue
                new_car_iou = self.cal_car_car_iou(triggered_coords, self.cameras['cam_ip']['last_triggered_coords'])
                if new_car_iou < self.new_car_iou_threshold:
                    # if the iou between this car and last trigger car is large
                    self.trigger(cam_ip)
                    continue
            else:
                print('Not implemented non-entrance trigger logic!')






