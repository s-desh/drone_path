import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(__name__):
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)

    log_file = "logs/sim.log"
    
    file_handler = RotatingFileHandler(log_file, maxBytes=1e6, backupCount=3)

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def clear_logs():
    files = os.listdir('logs')

    # Loop through the files and remove them
    for file_name in files:
        file_path = os.path.join('logs', file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            else:
                print(f"Skipped: {file_path} (not a file)")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")