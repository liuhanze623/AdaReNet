from datasets import load_dataset 
from datasets_test import load_dataset_my                      
from noise2noise_Liu import Noise2Noise          ### original
# from noise2noise_Liu_Fconv import Noise2Noise    ### Fconv
# from noise2noise_addloss_Liu import Noise2Noise    ### AdaReNet
from argparse import ArgumentParser
import torch
import numpy as np
import random
import os


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/valid')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=500, type=int)
    parser.add_argument('-ts', '--train-size', help='size of train dataset', type=int)
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', type=int)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)                            
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'hdr'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'poisson', 'text', 'mc'], default='gaussian', type=str)
    parser.add_argument('-p', '--noise-param', help='noise parameter (e.g. std for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=128, type=int)
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')
    parser.add_argument('-cm', '--crop-size-my', help='random crop size my', default=480, type=int)                                         

    return parser.parse_args()


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Parse training parameters
    params = parse_args()

    # Set random seed
    if params.seed is not None:
        seed_all(params.seed)
        
    # Train/valid datasets
    train_loader = load_dataset(params.train_dir, params.train_size, params, shuffled=True)       
    valid_loader = load_dataset_my(params.valid_dir, params.valid_size, params, shuffled=False)
    

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)
    n2n.train(train_loader, valid_loader)
