import argparse
import os
import random
from enum import Enum

import numpy as np
import torch
from torch.backends import cudnn


def create_folder(output_dir, with_checking=False):
    """
    Creates a folder.
    :param output_dir: Path to the location
    :return: full path to the folder
    """
    if with_checking and os.path.exists(output_dir):
        raise ValueError("The folder exists")
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir


def set_logging(log_dir: str, file_name: str):
    """
    Creates a logging file.
    :param log_dir: a logging directory
    """
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/{file_name}.log"),
            logging.StreamHandler()
        ]
    )


def set_seed(seed):
    """
    Sets seed for reproducible results.
    :param seed: int value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


def create_args() -> argparse.ArgumentParser:
    """
    Creates the argument parser.

    Returns:
        An argument parser.
    """
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--project_path',
                        type=str,
                        help='A path to the folder containing the EEG data '
                             'called "raw"')
    parser.add_argument('--seeds',
                        type=int,
                        help='Number of seeds to use.',
                        default=10)
    parser.add_argument('--benchmark',
                        type=str,
                        default='w',
                        help='"w" or "s"')

    return parser


class Relevance(Enum):
    RELEVANT = 1

    IRRELEVANT = 0
