import threading
import queue
from kafka import KafkaProducer
import json
import time
import logging
from datetime import datetime
import socket
import base64
import cv2
import struct

class SocketSender:
    def __init__(self, config):
        self.uri = (config['ip'], config['port'])
        self.img_buffer_size = config['img_buffer_size']  # (w,h)
        self.messages = queue.Queue(maxsize=config['max_stored_msg'])  # one msg might contain > 1 lpr results
        self.connection_threads = []

    def send(self, lpr_results):
        try:
            self.messages.put_nowait(lpr_results)
        except queue.Full:
            logging.error('Sender messages queue full!')
            raise

    def start_socket_streaming(self):
        if self._is_started:
            logging.warning('Attempted to start sender when it has already started')
            return None

        self._is_started = True
        logging.info('Sender started!')

    def _sending():
        while self._is_started:
            if self.messages.empty():
                continue

            try:
                lpr_results = self.messages.get_nowait()
            except queue.Empty:
                logging.exception('Sender try to get message when messages queue is empty (unexpected)')
                continue
            for cam_ip, lpr_result in lpr_results.items():
                if conf is None: # Recognition fail
                    logging.info(f'{cam_ip}: recognition failed')
                    continue
                plate_num = lpr_result['plate_num']
                conf = lpr_result['confidence']
                img = lpr_result['image']
                timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

            t = threading.Thread(target=self._send_single_msg,
                                 args=(timestamp, cam_ip, plate_num, conf, img))
            self.connection_threads.append(t)
            for t in self.connection_threads:
                t.join()
            self.connection_threads = []

    def _send_single_msg(self, timestamp, cam_ip, plate_num, conf, img):
        meta_data = self.encode_msg(timestamp, cam_ip, plate_num, conf)
        jpg_b64_text = self.encode_img(img)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            try:
                server.connect(self.uri)
                server.sendall(struct.pack(">I", len(meta_data)))  # pack as BE 32-bit unsigned int
                server.sendall(meta_data)
                server.sendall(jpg_b64_text)
                logging.info(f'Socket sent [{timestamp}, {cam_ip}, {plate_num}, {conf}, and image]')
            except socket.error as e:
                logging.error('Failed to send message via socket: {}'.format(e))

    def encode_msg(self, timestamp, cam_ip, plate_num, conf):
        meta_data = {'timestamp':timestamp, 
                     'camera_ip': cam_ip,
                     'license_num': license_num,
                     'confidence': conf
                     }
        meta_data = json.dumps(meta_data).encode('utf8')
        return meta_data

    def encode_img(self, img):
        img = cv2.resize(img, self.img_buffer_size, interpolation=cv2.INTER_LINEAR)
        retval, buffer = cv2.imencode('.jpg', img)
        jpg_b64_text = base64.b64encode(buffer)
        return jpg_b64_text






# def encode_json(m):
#     return json.dumps(m).encode('ascii')

# class KafkaSender:
#     def __init__(self, config):
#         self.topic = config['topic']
#         self.bootstrap_servers = config['bootstrap_servers']
#         self.max_stored_msg = config['max_stored_msg']

#         self.messages = queue.Queue(maxsize=self.max_stored_msg)
#         self._is_started = False
#         self.producer = self.init_producer()

#     def init_producer(self):
#         first_time = True
#         while True:
#             try:
#                 self.producer =  KafkaProducer(bootstrap_servers=self.bootstrap_servers, 
#                                                value_serializer=encode_json)
#                 return self.producer
#             except KeyboardInterrupt:
#                 raise
#             except:
#                 if first_time:
#                     logging.error('Failed to initialize Kafka producer, re-initializing...: bootstrap_servers {}'.format(self.bootstrap_servers))
#                     first_time = False
#                 time.sleep(5)
#                 continue

#     def start_kafka_streaming(self):
#         if self._is_started:
#             logging.warning('Attempted to start sender when it has already started')
#             return None

#         self._is_started = True
#         self.thread = threading.Thread(target=self._sending, args=())
#         self.thread.start()
#         logging.info('Sender started!')

#     def _sending(self):
#         while self._is_started:
#             if self.messages.empty():
#                 continue
#             try:
#                 msg = self.messages.get_nowait()
#             except queue.Empty:
#                 logging.exception('Sender try to get message when messages queue is empty (unexpected)')
#                 continue

#             date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
#             for cam_ip, (license_num, conf) in msg.items():
#                 if conf is None: # Recognition fail
#                     logging.info(f'{cam_ip}: recognition failed')
#                     continue
#                 output = {}
#                 output['Camera ID'] = cam_ip
#                 output['License number'] = license_num
#                 output['Time'] = date_time
#                 try:
#                     self.producer.send(self.topic, output)
#                     logging.info('Sender sent result {}'.format(output))
#                 except:
#                     logging.exception('Failed to send message {}'.format(output))

#     def send(self, license_numbers):
#         try:
#             self.messages.put_nowait(license_numbers)
#         except queue.Full:
#             logging.error('Sender messages queue full!')
#             raise

#     def stop(self):
#         self._is_started = False
#         self.thread.join()
