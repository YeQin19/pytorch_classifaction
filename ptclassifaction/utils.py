"""
Misc Utility functions
"""
import os
import logging
import datetime
import numpy as np
import torch.nn as nn

from collections import OrderedDict

def make_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.root.setLevel(logging.INFO)
    return logger