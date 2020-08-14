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
        self.lock = threading.Lock()
        self.accum_frames = queue.Queue(maxsize=self.num_votes)
        self._is_started = False
        self._accum_flag = False

    def set(self, propId, value):
        """ For setting a opencv VideoCapture property
        propId (float): for example: cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT
        value (value of the property)
        """
        self.cap.set(propId, value)

    def updating(self):
        while self._is_started:
            succ, frame = self.cap.read()
            if not succ:
                self.new_frame = None
                self.reconnecting()
                continue
            self.update_new_frame(frame)
            self.accumulate(frame)

    def update_new_frame(self, frame):
        with self.lock:
            self.new_frame = frame

    def accumulate(self, frame):
        if self._accum_flag:
            try:
                self.accum_frames.put_nowait(frame)
            except Full:
                print('UNEXPECTED: try to put frame when accum_frames is full: {}'.format(self.cam_ip))

            if self.accum_frames.full():
                self._accum_flag = False

    def start(self):
        if self._is_started:
            print('UNEXPECTED: camera had already started: {}'.format(self.cam_ip))
            return None

        self._is_started = True
        self.thread = threading.Thread(target=self.updating, args=())
        self.thread.start()

    def start_accumulate(self):
        """to start uptting frames into self.accum_frames"""
        if self._accum_flag:
            print('UNEXPECTED: started accumulating while previous accumulating process un-finished: {}'.format(self.cam_ip))
        if not self.accum_frames.empty():
            print('UNEXPECTED: started accumulating while previous accumulated frames un-used: {}'.format(self.cam_ip))
            self.clear_accum_frames()
        self._accum_flag = True

    def get_accumulated_frames(self):
        """to get all the accumulated frames only if its already full"""
        accum_frames = None
        if self.accum_frames.full():
            accum_frames = []
            while not self.accum_frames.empty():
                try:
                    accum_frames.append(self.accum_frames.get_nowait())
                except Empty:
                    print('UNEXPECTED: try to get frame when accum_frames is empty: {}'.format(self.cam_ip))
            accum_frames = np.array(accum_frames)
        return accum_frames

    def get_new_frame(self):
        with self.lock:
            new_frame = self.new_frame
        return new_frame
                
    def reconnecting(self):
        # try to reconnect untill succ
        print('UNEXPECTED: camera cannot read frame, trying to reconnect...: {}'.format(self.cam_ip))
        while self._is_started:
            self.cap = cv2.VideoCapture(self.cam_ip)
            time.sleep(5)
            succ, _ = self.cap.read()
            if succ:
                print('Camera is now reconnected: {}'.format(self.cam_ip))
                return True
            else:
                continue

    def stop(self):
        self._is_started = False
        self.cap.release()
        self.thread.join()

    def clear_accum_frames(self):
        self.accum_frames = queue.Queue(maxsize=self.num_votes)
