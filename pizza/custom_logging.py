import logging

import os
import sys
from logging.handlers import RotatingFileHandler

"""
Import logger and use it to log the files
"""
LOG_FP_ENV = os.getenv("LOG_FILE_PATH", ".")
LOG_FP = os.path.join(LOG_FP_ENV, 'pizza.log') if os.path.isdir(LOG_FP_ENV) else 'pizza.log'
print(f"Log file path: {os.path.abspath(LOG_FP)}")
MAX_MB = 10
BACKUP_COUNT = 5


def _get_logger():
    loglevel = logging.INFO
    l = logging.getLogger(__name__)
    if not l.handlers:
        l.setLevel(loglevel)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s',
                                      datefmt='%Y/%m/%d %H:%M:%S')
        # Stream Handler
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(formatter)
        l.addHandler(h)

        # File Handler
        fh = RotatingFileHandler(LOG_FP, maxBytes=MAX_MB * 1024 * 1024, backupCount=BACKUP_COUNT)
        fh.setFormatter(formatter)
        l.addHandler(fh)

        l.setLevel(loglevel)
        l.handler_set = True
        l.propagate = False

    return l


logger = _get_logger()
