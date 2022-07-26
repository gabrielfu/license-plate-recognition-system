import threading
import cv2
import queue
from enum import Enum
import time
import numpy as np
import logging

class CameraType(Enum):
    entrance = 0
    top = 1
    bot = 2

class Camera:
    def __init__(self,
                 cam_ip,
                 cam_type,
                 num_votes=5,
                 accum_time=.5,
                 cam_fps_sim=False
                 ):
        """
        Args:
            cam_ip (str/int 0): camera src for cv2.VideoCapture
            cam_type (CameraType): camera type
            num_votes (int): max length of frames_lst
        """
        if cam_type not in CameraType:
            logging.error("{}: Invalid camera type {}".format(cam_ip, cam_type))

        self.cam_ip = cam_ip
        self.cam_type = cam_type
        self.accum_time = accum_time
        self.start_accum_time = time.monotonic()
        self.num_votes = num_votes
        self.frame_min_interval = 1/(cam_fps_sim + 1e-16) if cam_fps_sim else 0
        self.fps = None
        self.fps_update_freq = 50000
        self.num_frames = 0
        self.start_time = time.monotonic()
        self.last_update_time = time.monotonic()
        self.cap = cv2.VideoCapture(self.cam_ip)
        self.new_frame = None
        self.lock = threading.Lock()
        self.accum_frames = []
        self._is_started = False
        self._is_accumulating = False
        
    def __str__(self):
        return f'Camera({self.cam_type}, {self.cam_ip})'

    @property
    def is_started(self):
        return self._is_started

    @property
    def is_accumulating(self):
        return self._is_accumulating

    def set(self, propId, value):
        """For setting a opencv VideoCapture property
        propId (float): for example: cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT
        value (value of the property)
        """
        self.cap.set(propId, value)

    def get_fps(self):
        with self.lock:
            fps = self.fps
        return fps

    def _update_fps(self, cur_time):
        with self.lock:
            self.num_frames += 1
            self.fps = self.num_frames / (cur_time - self.start_time + 1e-16)

        if self.num_frames % self.fps_update_freq == 0:
            with self.lock:
                self.num_frames = 0
                self.start_time = time.monotonic()

    def _updating(self):
        """Camera updating process
        1. keep updating self.new_frame if self._is_started
        2. keep accumulating if self._is_accumulating
        """
        while self._is_started:
            succ, frame = self.cap.read()
            if not succ:
                self.new_frame = None
                self.reconnecting()
                continue
            cur_time = time.monotonic()
            time_interval = cur_time - self.last_update_time
            self.last_update_time = cur_time
            if time_interval < self.frame_min_interval:
                time.sleep(self.frame_min_interval - time_interval)
            self._update_fps(cur_time)
            if self.cam_type == CameraType.bot or self.cam_type == CameraType.entrance:
                self._update_new_frame(frame)
            if self.cam_type == CameraType.top or self.cam_type == CameraType.entrance:
                self._accumulate(frame)

    def _update_new_frame(self, frame):
        """Update self.new_frame with lock"""
        with self.lock:
            self.new_frame = frame

    def _accumulate(self, frame):
        """Accumulate if self._is_accumulating is True. When exceed accum_time, set self._is_accumulating to be False
        """
        if self._is_accumulating:
            self.accum_frames.append(frame)
            if time.monotonic() - self.start_accum_time >= self.accum_time:
                self._is_accumulating = False

    def start(self):
        """To start reading frames """
        if self._is_started:
            logging.warning(f'{self.cam_ip}: attempted to start camera when it has already started')
            return None

        self._is_started = True
        self.thread = threading.Thread(target=self._updating, args=())
        self.thread.start()
        logging.info('Camera started: {}'.format(self.cam_ip))

    def start_accumulate(self):
        """To start putting frames into self.accum_frames. Please call this function only after calling start()"""
        if self._is_accumulating:
            logging.warning(f'{self.cam_ip}: started accumulating while previous accumulating process un-finished')
        if len(self.accum_frames) != 0:
            logging.warning(f'{self.cam_ip}: started accumulating while previous accumulated frames un-used')
            self._clear_accum_frames()
        self._is_accumulating = True
        self.start_accum_time = time.monotonic()


    def get_accumulated_frames(self):
        """To get all the accumulated frames only if its already full
        returns:
            accum_frames (None/np.array): None if self.accum_frames not full. np.array of shape self.num_votes*h*w*c otherwise
        """
        accum_frames = None
        accum_frames_length = len(self.accum_frames)
        if not self._is_accumulating and accum_frames_length > 0:
            num_votes = min(accum_frames_length, self.num_votes)
            if accum_frames_length < self.num_votes:
                logging.warning(f'{self.cam_ip}: less than {self.num_votes} accum_frames  (only {accum_frames_length})')
            idxes = [int((accum_frames_length-1)/(num_votes-1+1e-16) * i) for i in range(num_votes)]
            accum_frames = [self.accum_frames[i] for i in idxes]
            accum_frames = np.array(accum_frames)
            self._clear_accum_frames()
        return accum_frames

    def get_new_frame(self):
        """Get self.new_frame with lock
        returns:
            new_frame (None/np.array): None if camera failed to get frame, np.array of shape h*w*c otherwise
        """
        with self.lock:
            new_frame = self.new_frame
        return new_frame
                
    def reconnecting(self):
        """Try to reconnect untill succ"""
        logging.warning(f'{self.cam_ip}: camera cannot read frame, trying to reconnect...')
        while self._is_started:
            self.cap = cv2.VideoCapture(self.cam_ip)
            time.sleep(5)
            succ, _ = self.cap.read()
            if succ:
                logging.warning(f'{self.cam_ip}: camera reconnected')
                # reset fps
                with self.lock:
                    self.num_frames = 0
                    self.start_time = time.monotonic()
                return True

    def stop(self):
        self._is_started = False
        self.cap.release()
        self.thread.join()

    def _clear_accum_frames(self):
        del self.accum_frames
        self.accum_frames = []
