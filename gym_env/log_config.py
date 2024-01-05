import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(__name__):
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)

    log_file = "sim.log"
    
    file_handler = RotatingFileHandler(log_file, maxBytes=1e6, backupCount=3)

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger