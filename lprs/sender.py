import threading
import queue
from kafka import KafkaProducer
import json
import time
import logging
from datetime import datetime
import base64

def encode_json(m):
    return json.dumps(m).encode('ascii')

class KafkaSender:
    def __init__(self, config):
        self.topic = config['topic']
        self.bootstrap_servers = config['bootstrap_servers']
        self.max_stored_msg = config['max_stored_msg']

        self.messages = queue.Queue(maxsize=self.max_stored_msg)
        self._is_started = False
        self.producer = self.init_producer()

    def init_producer(self):
        first_time = True
        while True:
            try:
                self.producer =  KafkaProducer(bootstrap_servers=self.bootstrap_servers, 
                                               value_serializer=encode_json)
                return self.producer
            except Exception:
                if first_time:
                    logging.error(f'Failed to initialize Kafka producer, re-initializing...: bootstrap_servers {self.bootstrap_servers}')
                    first_time = False
                time.sleep(5)
                continue

    def start_kafka_streaming(self):
        if self._is_started:
            logging.warning('Attempted to start sender when it has already started')
            return None

        self._is_started = True
        self.thread = threading.Thread(target=self._sending, args=())
        self.thread.daemon = True
        self.thread.start()
        logging.info('Sender started!')

    def _sending(self):
        while self._is_started:
            if self.messages.empty():
                continue
            try:
                msg = self.messages.get_nowait()
            except queue.Empty:
                logging.exception('Sender try to get message when messages queue is empty (unexpected)')
                continue

            date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for cam_ip, item in msg.items():
                license_number = item["plate_num"]
                confidence = item["confidence"]
                image = item["image"]
                if confidence is None: # Recognition fail
                    logging.info(f'{cam_ip}: recognition failed')
                    continue
                output = {
                    'camera': cam_ip,
                    'license_number': license_number,
                    'confidence': confidence,
                    # 'image': base64.b64encode(image).decode('ascii'), # image too large, don't send
                    'time': date_time,
                }
                try:
                    self.producer.send(self.topic, output)
                    logging.info(f'Sender sent result {output}')
                except Exception:
                    logging.exception(f'Failed to send message {output}')

    def send(self, license_numbers):
        try:
            self.messages.put_nowait(license_numbers)
        except queue.Full:
            logging.error('Sender messages queue full!')
            raise

    def stop(self):
        self._is_started = False
        self.thread.join()
