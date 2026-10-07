import argparse
import functools
import os
import time

import torch
import torchvision

import src.utils.Tree_Dataset
import src.utils.distributed_utils

""" This file will (likely) end up being the main script that will be run on JSC.
    Most of this is taken from [https://gitlab.jsc.fz-juelich.de/sdlaml/pytorch-at-jsc/-/blob/main/pytorch-ddp-example/main.py?ref_type=heads]
"""


def parse_args():
    # TODO:
    """ Utility function for parsing settings from command line arguments when running on JSC.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path',
        type=Path,
        default='./',
        help='Path to root folder containing Trees.json and ./images/.',
    )

    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Ratio of examples in test set.',
    )

    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.2,
        help='Ratio of examples in test set.',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed for randomly splitting and shuffling datasets.',
    )

    parser.add_argument(
        '--batch_size',
        time=int,
        default=32,
        help=(
            'How many samples to use per batch. '
            'Note that this is the local batch size; '
            'the effective, or global, batch size will be obtained by '
            'multiplying this number with the number of processes.'
        ),
    )

    parser.add_argument(
        '--train-num-workers',
        type=int,
        default=0,
        help='How many workers to use for processing the training dataset.',
    )

    parser.add_argument(
        '--valid-num-workers',
        type=int,
        default=0,
        help='How many workers to use for processing the validation dataset.',
    )

    # more args can be added as needed

    args = parser.parse_args()
    return args



