import threading
import cv2
import queue
from enum import Enum
import time


class CameraType(Enum):
    entrance = 0
    top = 1
    bot = 2


class Camera:
    def __init__(self,
                 cam_ip,
                 cam_type,
                 num_votes=8,
                 ):
        """
        Args:
            cam_ip (str/int 0): camera src for cv2.VideoCapture
            cam_type (CameraType): camera type
            num_votes (int): max length of frames_lst
        """
        if cam_type not in CameraType:
            raise ValueError("{}: Invalid camera type {}".format(cam_ip, cam_type))

        self.cam_ip = cam_ip
        self.cam_type = cam_type
        self.num_votes = num_votes
        self.fps = 0
        self.cap = cv2.VideoCapture(self.cam_ip)
        self.new_frame = None
        self.accum_frames = queue.Queue(maxsize=self.num_votes)
        self._is_started = False
        self._is_accumulating = False
        self.is_accum_starting = False
        self.is_accum_finished = False

    def set(self, propId, value):
        """ For setting a opencv VideoCapture property
        propId (float): for example: cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT
        value (value of the property)
        """
        self.cap.set(propId, value)

    def start(self):
        if self._is_started:
            print('{} had already been started'.format(self.cam_ip))
            return None

        self._is_started = True
        self.thread = threading.Thread(target=self.updating, args=())
        self.thread.start()

    def updating(self):
        while self._is_started:
            succ, frame = self.cap.read()
            if not succ:
                self.new_frame = None
                self.reconnecting()
                continue
            self.new_frame = frame
            if self.is_accum_starting or self._is_accumulating:
                if self.is_accum_starting:
                    self.clear_accum_frames()
                    self.is_accum_starting = False
                    self._is_accumulating = True
                self.accumulate(frame)
                if self.accum_frames.full():
                    self.is_accum_finished = True
                    self._is_accumulating = False

    def accumulate(self, frame):
        try:
            self.accum_frames.put_nowait(frame)
        except Full:
            print('UNEXPECTED: try to put frame when accum_frames is full: {}'.format(self.cam_ip))
                
    def reconnecting(self):
        # try to reconnect untill succ
        print('{} cannot read frame, trying to reconnect...'.format(self.cam_ip))
        while self._is_started:
            self.cap = cv2.VideoCapture(self.cam_ip)
            time.sleep(5)
            succ, _ = self.cap.read()
            if succ:
                print('{} is now reconnected!'.format(self.cam_ip))
                return True
            else:
                continue

    def stop(self):
        self._is_started = False
        self.cap.release()
        self.thread.join()

    def clear_accum_frames(self):
        self.accum_frames = queue.Queue(maxsize=self.num_votes)
