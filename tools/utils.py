# -*- coding: utf-8 -*-
# @Time : 2022/5/13 20:28
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

import os
import pickle
import time

import numpy as np


def get_args(*p_args, **p_kwargs):
    """Get args using argparse."""
    import argparse
    parser = argparse.ArgumentParser(description='MATH8013 Final.')
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--seed", type=int, default=1234321, help="Seed.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU No.")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="Total epochs.")
    parser.add_argument("--network", type=str, default='vgg11',
                        help="Select network.\n"
                             "Optional='vgg11','vgg11_bn',"
                             "'vgg13','vgg13_bn', "
                             "'vgg16', 'vgg16_bn',"
                             "'vgg19', 'vgg19_bn'"
                        )
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path of pretrained model")

    _args = parser.parse_args(*p_args, **p_kwargs)
    return _args


def seed_everything(seed: int):
    """Seed everything.

    Copied from https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    """

    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def makedirs(path):
    """Make directories if not exist."""

    if not os.path.exists(path):
        os.makedirs(path)


def get_timestamp(file_name=None):
    """Get the current timestamp."""

    localtime = time.localtime(time.time())
    date_time = "{}_{}_{}_{}_{}_{}".format(localtime.tm_year, localtime.tm_mon, localtime.tm_mday, localtime.tm_hour,
                                           localtime.tm_min, localtime.tm_sec)
    if file_name:
        return file_name + '_' + date_time
    return date_time


def save_parameters(root_dir: str, para: dict, save_pickle: bool = False):
    """Save the parameter dictionary as file.

    :param save_pickle: Bool. Whether to save a pickle file.
    :param root_dir: Root directory for output.
    :param para: Parameter dictionary to be saved.
    """

    makedirs(root_dir)
    if save_pickle:
        with open(os.path.join(root_dir, "data.pickle"), 'wb') as f:
            pickle.dump(para, f, protocol=4)

    text_file = open(os.path.join(root_dir, "parameter-list.txt"), "w")
    for p in para:
        if np.size(para[p]) > 20:
            continue
        text_file.write(f'{p}:\t{para[p]}\n')
    text_file.close()
