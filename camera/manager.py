from .camera import Camera, CameraType


class CameraManager:
	def __init__(self, config_file):
		pass

	def get_all_latest_frames(self):
		"""
		return:
			all_latest_frames (dict(str: np.array / None, ...)): keys: camera's ip,
															     values: a latest cv2 frame if exists, None if camera failed to read frame 
		"""
		pass

	def get_all_accum_frames(self):
		"""
		return:
			all_accum_frames (dict(str: np.array / None, ...)): keys: camera's ip,
															    values: np.array of shape (num_votes, h, w, c) if exists, None otherwise
		"""
		pass

	def update_camera_accum_status(self, car_locations):
		"""based on located car locations to justify if cameras need to start accumulating
		"""
		pass