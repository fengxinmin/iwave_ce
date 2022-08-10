import argparse
import torch
from torch.autograd import Variable
import os
import glob as gb
from PIL import Image
import numpy as np
from torch.nn import functional as F
import Quant
import copy
import time
import json

from enc_lossy import enc_lossy
from enc_lossless import enc_lossless

parser = argparse.ArgumentParser(description='IEEE 1857.11 FVC CFP', conflict_handler='resolve')
# parameters
parser.add_argument('--cfg_file', type=str, default='/ghome/liqiang/code/iwave/cfg/encode.cfg')

args, unknown = parser.parse_known_args()

cfg_file = args.cfg_file
with open(cfg_file, 'r') as f:
    cfg_dict = json.load(f)
    
    for key, value in cfg_dict.items():
        if isinstance(value, int):
            parser.add_argument('--{}'.format(key), type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument('--{}'.format(key), type=float, default=value)
        else:
            parser.add_argument('--{}'.format(key), type=str, default=value)

cfg_args, unknown = parser.parse_known_args()

parser.add_argument('--input_dir', type=str, default=cfg_args.input_dir)
parser.add_argument('--img_name', type=str, default=cfg_args.img_name)
parser.add_argument('--bin_dir', type=str, default=cfg_args.bin_dir)
parser.add_argument('--log_dir', type=str, default=cfg_args.log_dir)
parser.add_argument('--recon_dir', type=str, default=cfg_args.recon_dir)
parser.add_argument('--isLossless', type=int, default=cfg_args.isLossless)
parser.add_argument('--model_qp', type=int, default=cfg_args.model_qp)
parser.add_argument('--qp_shift', type=int, default=cfg_args.qp_shift)
parser.add_argument('--isPostGAN', type=int, default=cfg_args.isPostGAN)

parser.add_argument('--model_path', type=str, default=cfg_args.model_path) # store all models

parser.add_argument('--code_block_size', type=int, default=cfg_args.code_block_size)


def main():
    args = parser.parse_args()

    if args.isLossless == 0:
        enc_lossy(args)
    else:
        enc_lossless(args)

if __name__ == "__main__":
    main()
