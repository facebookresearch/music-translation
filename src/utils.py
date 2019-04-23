# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os
import sys
import time
from datetime import timedelta

import numpy
import numpy as np
import torch
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')
from scipy.io import wavfile


class timeit:
    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger is None:
            print(f'{self.name} took {(time.time() - self.start) * 1000} ms')
        else:
            self.logger.debug('%s took %s ms', self.name, (time.time() - self.start) * 1000)


def mu_law(x, mu=255):
    x = numpy.clip(x, -1, 1)
    x_mu = numpy.sign(x) * numpy.log(1 + mu*numpy.abs(x))/numpy.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')


def inv_mu_law(x, mu=255.0):
    x = numpy.array(x).astype(numpy.float32)
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return numpy.sign(y) * (1./mu) * ((1. + mu)**numpy.abs(y) - 1.)


class LossMeter(object):
    def __init__(self, name):
        self.name = name
        self.losses = []

    def reset(self):
        self.losses = []

    def add(self, val):
        self.losses.append(val)

    def summarize_epoch(self):
        if self.losses:
            return np.mean(self.losses)
        else:
            return 0

    def sum(self):
        return sum(self.losses)


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_output_dir(opt, path: Path):
    if hasattr(opt, 'rank'):
        filepath = path / f'main_{opt.rank}.log'
    else:
        filepath = path / 'main.log'

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    if hasattr(opt, 'rank') and opt.rank != 0:
        sys.stdout = open(path / f'stdout_{opt.rank}.log', 'w')
        sys.stderr = open(path / f'stderr_{opt.rank}.log', 'w')

    # Safety check
    if filepath.exists() and not opt.checkpoint:
        logging.warning("Experiment already exists!")

    # Create log formatter
    log_formatter = LogFormatter()

    # Create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # create console handler and set level to info
    if hasattr(opt, 'rank') and opt.rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info(opt)
    return logger


def setup_logger(logger_name, filename):
    logger = logging.getLogger(logger_name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        stderr_handler.setLevel(logging.WARNING)
    else:
        stderr_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    return logger


def wrap(data, **kwargs):
    if torch.is_tensor(data):
        var = data.cuda(non_blocking=True)
        return var
    else:
        return tuple([wrap(x, **kwargs) for x in data])


def save_audio(x, path, rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, rate, x)


def save_wav_image(wav, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(15, 5))
    plt.plot(wav)
    plt.savefig(path)