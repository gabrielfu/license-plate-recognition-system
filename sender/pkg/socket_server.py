import socket
import threading
import json
import struct
import cv2
import time

import base64
import numpy as np

class ThreadedServer(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

    def listen(self):
        self.sock.listen(5)
        while True:
            client, address = self.sock.accept()
            print('=========================== New client comes in ===========================')
            client.settimeout(60)
            # threading.Thread(target = self.listenToClient,args = (client,address)).start()
            self.listenToClient(client, address)

    def listenToClient(self, client, address):
        while True:
            # try:
            #     payload_size = struct.unpack(">I", client.recv(4))[0]
            # except:
            #     payload_size = 0

            try:
                # =========================== Meta data ===========================
                meta_size = struct.unpack(">I", client.recv(4))[0]  # first 4 bytes indicate the meta data size & rest bytes are image
                meta_data = client.recv(meta_size)
                meta_data = meta_data.decode('utf8')

                # =========================== Base64 image data ===========================
                b64_img_data = b''
                chunk = client.recv(16384)
                while chunk:
                    b64_img_data += chunk
                    chunk = client.recv(16384)

                if meta_data:
                    # Set the response to echo back the recieved data 
                    # response = data
                    # client.send(response)
                    print(type(meta_data))
                    print(meta_size, meta_data, len(b64_img_data))
                    # decoded_string = base64.b64decode(b64_img_data)
                    # decoded_img = np.fromstring(decoded_string, dtype=np.uint8)
                    # decoded_img = cv2.imdecode(decoded_img,cv2.IMREAD_COLOR)
                    # print(decoded_img.shape)
                    # cv2.imshow("decoded", decoded_img)
                    # cv2.waitKey(0)
                else:
                    raise error('Client disconnected')
            except Exception as e:
                # print('Error message: ', e)
                client.close()
                return False

if __name__ == "__main__":
    while True:
        port_num = input("Port? ")
        try:
            port_num = int(port_num)
            break
        except ValueError:
            pass

    ThreadedServer('127.0.0.1',port_num).listen()