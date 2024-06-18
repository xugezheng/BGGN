import logging
import numpy as np
import datetime
import importlib
import time
import os.path as osp
import copy

__all__ = ["TxtLogger"]


def TxtLogger(filename, verbosity="info", logname="intersectionalFair"):
    level_dict = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    formatter = logging.Formatter(
        fmt=
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(logname)
    logger.setLevel(level_dict[verbosity])
    # file handler
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # stream handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger, fh, sh


def set_logger(key_info, seed, output_dir):
    now = int(round(time.time() * 1000))
    log_file = f"{key_info}_{str(seed)}_{str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(now / 1000)))}.log"
    log_dir = osp.abspath(
            osp.join(
                output_dir,
                "logs"
            ))
    return log_dir, log_file



    
