import threading
import queue
from kafka import KafkaProducer
import json
import time
import logging
from datetime import datetime


def cam_ip_to_cam_location(cam_ip):
    return cam_ip

def encode_json(m):
    return json.dumps(m).encode('ascii')

class KafkaSender:
    def __init__(self, config):
    self.topic = config['topic']
    self.bootstrap_servers = config['bootstrap_servers']
    self.max_stored_msg = config['max_stored_msg']

    self.messages = queue.Queue(maxsize=self.max_stored_msg)
    self._is_started = False

    while True:
        try:
            self.producer =  KafkaProducer(bootstrap_servers=self.bootstrap_servers, 
                                           value_serializer=encode_json)
            break
        except:
            logging.error('Failed to initialize Kafka producer, re-initializing...: bootstrap_servers {}'.format(self.bootstrap_servers))
            continue

    def start_kafka_streaming(self):
        if self._is_started:
            logging.warning('Attempted to start sender when it has already started')
            return None

        self._is_started = True
        self.thread = threading.Thread(target=self._sending, args=())
        self.thread.start()
        logging.info('Sender started!')

    def _sending(self):
        while self._is_started:
            if self.messages.empty():
                continue
            try:
                msg = self.messages.get_nowait()
            except queue.Empty:
                self.logger.exception('Sender try to get message when messages queue is empty (unexpected)')
                continue

            date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            for cam_ip, (license_num, conf) in msg.items():
                if conf is None: # Recognition fail
                    logging.warning('Recognition fail at {}'.format(date_time))
                    continue
                output = {}
                output['Camera ID'] = cam_ip
                output['License number'] = license_num
                output['Time'] = date_time
                try:
                    self.producer.send(self.topic, output)
                    logging.info('Sender sent result {}'.format(output))
                except:
                    logging.exception('Failed to send message {}'.format(output))

    def send(self, license_numbers):
        try:
            self.max_stored_msg.put_nowait(license_numbers)
        except queue.Full:
            logging.exception('Sender messages queue full!')
