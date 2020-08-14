import yaml
from .camera import Camera, CameraType


class CameraManager:
	def __init__(self, config_file):
		with open(config_file) as file:
    		default_dict = yaml.load(file)
    	self.num_votes = default_dict['properties']['num_votes']
    	self.cameras = self.init_cameras(default_dict['cameras'])

    def start_cameras_streaming(self):
    	"""To start all the cameras
    	"""
    	for _, cam_dict in self.cameras.items():
    		cam_dict['camera'].start()

    def init_cameras(cameras_dict):
    	"""Construct a dictionay that stores all camera info
    	returns:
    		cameras (dict('cam_ip': dict('camera': Camera, 'trigger_zone': trigger_zone), ...)):
    		example:
    		{'128.1.1.0': {'camera': Camera(), 'trigger_zone': [(1,2), (3,4), (5,6), (7,8)]}
		     '128.1.1.1': {'camera': Camera(), 'trigger_zone': [(1,1), (2,2), (3,3), (4,4)]}
		     ...
    		}
    	"""
    	cameras = {}
    	for _, v in cameras_dict.items():
    		cam_ip = v['ip']
    		cam_type = v['type']
    		if cam_type == 'entrance':
    			cam_type = CameraType.entrance
    		elif cam_type == 'top':
    			cam_type = CameraType.top
    		elif cam_type == 'bot':
    			cam_type = CameraType.bot
    		else:
    			raise NotImplementedError('Unkown cam_type: {}'.format(cam_type))
    		trigger_zone = v['trigger_zone']
    		cameras[cam_ip] = {'camera_stream': Camera(cam_ip, cam_type, self.num_votes),
    						   'trigger_zone': trigger_zone}
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
		for cam_ip, cam_dict in self.cameras.item():
			frames = {}
			camera = cam_dict['camera_stream']
			new_frame = camera.get_new_frame()
			accum_frames = camera.get_accumulated_frames()
			frames['new_frame'] = new_frame
			frames['accum_frames'] = accum_frames
			all_frames[cam_ip] = cam_ip
		return all_frames

	def update_camera_accum_status(self, car_locations):
		"""based on located car locations to justify if cameras need to start accumulating
		"""
		pass