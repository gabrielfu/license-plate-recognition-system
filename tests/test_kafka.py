import os
import sys
sys.path.insert(0, os.getcwd())
import time
import logging
import json
import threading
import numpy as np
from kafka import KafkaConsumer

from lprs.sender.sender import KafkaSender
from lprs.utils.utils import read_yaml
from lprs.logger import setup_logging


# Read configs
logger_cfg = read_yaml('./config/logger.yaml')
kafka_cfg = read_yaml('./config/kafka.yaml')

# Setup logging handlers & initialize logger
os.makedirs(logger_cfg['log_dir'], exist_ok=True)
setup_logging(logger_cfg)

def produce():

    # Initialize kafka sender and start output streaming
    logging.info('Starting Kafka sender...')
    sender = KafkaSender(kafka_cfg)
    sender.start_kafka_streaming()

    for _ in range(3):
        license_numbers = {
            '123.0.0.1': {
                "plate_num": "AB1234",
                "confidence": 0.99,
                "image": np.random.rand(5, 3),
            },
            '123.0.0.2': {
                "plate_num": "CD5678",
                "confidence": 0.88,
                "image": np.random.rand(5, 3),
            },
        }

        # Output with Kafka
        sender.send(license_numbers)

        time.sleep(2)

    return


def consume():
    consumer = KafkaConsumer(kafka_cfg["topic"],
                             bootstrap_servers=[kafka_cfg["bootstrap_servers"]],
                             value_deserializer=lambda m: json.loads(m.decode('ascii')))

    for message in consumer:
        logging.info(f"=> Received {message.value}")


def run():
    consumer = threading.Thread(target=consume, args=())
    consumer.daemon = True
    consumer.start()

    producer = threading.Thread(target=produce, args=())
    producer.daemon = True
    producer.start()
    producer.join()

if __name__ == '__main__':
    run()
