from logger import setup_logging
from sender.sender import SocketSender
import time
import cv2
import sys
import logging


def exit_app():
    ''' Shut down the whole application'''
    logging.info('Shutting down application')
    sys.exit()

## init sender
socket_cfg = read_yaml('config/socket.yaml')
try:
    # sender = KafkaSender(kafka_cfg)
    # sender.start_kafka_streaming()
    sender = SocketSender(socket_cfg)
    sender.start_socket_streaming()
except Exception:
    logging.exception('Failed to start sender!')
    exit_app()

img = cv2.imread('./asset/sample_img.jpg')
## construct demo data
lpr_results == {
    '123.0.0.1': {
        'plate_num': 'AB1234',
        'confidence': 0.99,
        'image': img
    },
    '123.0.0.2': {
        'plate_num': 'CD5678',
        'confidence': 0.88,
        'image': img
    },
}

while True:
    sender.send(lpr_results)
    time.sleep(60)