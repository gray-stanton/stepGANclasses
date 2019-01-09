import argparse
import logging
import sys
import os
from shutil import rmtree

def get_logger(model_name, model_dir):
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler("{0}/log/{1}.log".format(model_dir, model_name)),
        logging.StreamHandler(sys.stdout)
    ])
    logger = logging.getLogger(model_name)
    return logger
    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                       type=str)
    parser.add_argument('--model-name',
                        default=None,
                        type=str)
    parser.add_argument('--hparams',
                        type=str)
    parser.add_argument('--data-dir',
                        type=str)
    parser.add_argument('--base-dir',
                       type=str)
    parser.add_argument('--no-restore',
                        default=False,
                        action='store_true')
    parser.add_argument('--load-checkpoint',
                        default = None,
                        type=str)
    parser.add_argument('--dataset-name',
                        type=str)
    parser.add_argument('--dataset-type',
                        type=str)
    parser.add_argument('--seq-length',
                        default=16,
                       type=int)
    parser.add_argument('--use-frac',
                        default=1.0,
                        type=float)
    parser.add_argument('--train-frac',
                        default=0.7,
                        type=float)
    parser.add_argument('--batch-size',
                        default=32,
                        type=int)
    parser.add_argument('--epochs',
                        default=1,
                        type=int)
    parser.add_argument('--early-stopping',
                        default=False,
                        action='store_true')
    parser.add_argument('--es-delta',
                        default=0.005,
                        type=float)
    parser.add_argument('--patience',
                       default=3,
                       type=int)
    parser.add_argument('--num-samples',
                        default=1,
                        type=int)
    parser.add_argument('--seed-text',
                       type=str,
                       default='A')
    parser.add_argument('--sample-length',
                        type=int,
                        default=50)
    parser.add_argument('--no-gpu',
                        default=False,
                        action='store_true')
    parser.add_argument('--log-interval',
                        default=20,
                        type=int)
    parser.add_argument('--run-name',
                        default='',
                        type=str)
    parser.add_argument('--temperature',
                        default=1.0,
                        type=float)
    parser.add_argument('--n-classes',
                        default=2,
                        type=int)
    parser.add_argument('--pretrained-lm-dir',
                        default=None,
                        type=str)
    return parser


def remove_history(model_dir):
    try:
        rmtree(model_dir + 'log/')
        rmtree(model_dir + 'ckpts/')
        rmtree(model_dir + 'output/')
    except FileNotFoundError:
        pass


def setup_model_dir(model_dir, create_base=True):
    if create_base:
        os.mkdir(model_dir)
    os.mkdir(model_dir + '/log/')
    os.mkdir(model_dir + '/ckpts/')
    os.mkdir(model_dir + '/output/')

def split(dataset, size, use_frac, train_frac):
    dataset = dataset.shuffle(10000, reshuffle_each_iteration=False)
    use_dataset = dataset.take(int(use_frac * size))
    train_ds = use_dataset.take(int(train_frac * use_frac * size))
    valid_ds = use_dataset.skip(int(train_frac * use_frac * size))
    return train_ds, valid_ds

