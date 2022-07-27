import os
import sys
sys.path.insert(0, os.getcwd())
import time
import logging
import numpy as np

from lprs.sender.sender import KafkaSender
from lprs.utils.utils import read_yaml
from lprs.logger import setup_logging


def run():

    # Read configs
    logger_cfg = read_yaml('./config/logger.yaml')
    kafka_cfg = read_yaml('./config/kafka.yaml')

    # Setup logging handlers & initialize logger
    os.makedirs(logger_cfg['log_dir'], exist_ok=True)
    setup_logging(logger_cfg)
    logging.info('-------------------------------------------------------')
    logging.info('Starting Application...')

    # Initialize kafka sender and start output streaming
    logging.info('Starting Kafka sender...')
    sender = KafkaSender(kafka_cfg)
    sender.start_kafka_streaming()

    while True:
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
        logging.info(f'LPR RESULT: {license_numbers}')
        try:
            sender.send(license_numbers)
        except Exception:
            logging.critical("Sender failed to send LPR results!")

        time.sleep(4)


if __name__ == '__main__':
    run()
