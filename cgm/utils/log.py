import logging
import os
from datetime import datetime


def init_logger(log_path, model, tag):
    fmt = '%(asctime)s %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=date_fmt, filemode='w',
                        filename=os.path.join(log_path, model, tag, tag + '.log'))
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger()
    return logger