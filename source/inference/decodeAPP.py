import argparse
import torch
from torch.autograd import Variable
import os
import glob as gb
from PIL import Image
import numpy as np
from torch.nn import functional as F
import time
import json
import arithmetic_coding as ac

from dec_lossy import dec_lossy
from dec_lossless import dec_lossless

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

parser = argparse.ArgumentParser(description='IEEE 1857.11 FVC CFP', conflict_handler='resolve')

parser.add_argument('--cfg_file', type=str, default='/ghome/liqiang/code/iwave/cfg/decode.cfg')

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

# parameters
parser.add_argument('--bin_file', type=str, default=cfg_args.bin_file)
parser.add_argument('--recon_dir', type=str, default=cfg_args.recon_dir)
parser.add_argument('--log_dir', type=str, default=cfg_args.log_dir)

parser.add_argument('--model_path', type=str, default=cfg_args.model_path) # store all models

parser.add_argument('--code_block_size', type=int, default=cfg_args.code_block_size)


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def dec_binary(dec, bin_num):
    value = 0
    freqs = ac.SimpleFrequencyTable([1, 1])
    for i in range(bin_num):
        dec_c = dec.read(freqs)
        value = value + (2**(bin_num-1-i))*dec_c
    return value


def main():
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.recon_dir):
        os.makedirs(args.recon_dir)

    bin_path = args.bin_file
    bin_name = os.path.basename(bin_path)[0:-4]

    logfile = open(args.log_dir + '/dec_log_{}.txt'.format(bin_name), 'w')

    start = time.time()

    print(bin_path)
    logfile.write(bin_path + '\n')
    logfile.flush()

    bit_in = ac.BitInputStream(open(bin_path, "rb"))
    dec = ac.ArithmeticDecoder(bit_in)
    freqs_resolution = 1e6

    freqs = ac.SimpleFrequencyTable(np.ones([2], dtype=np.int))
    isLossless = dec.read(freqs)

    if isLossless == 0:
        dec_lossy(args, bin_name, dec, freqs_resolution, logfile)
    else:
        dec_lossless(args, bin_name, dec, freqs_resolution, logfile)

    end = time.time()
    print('decoding time: ', end - start)
    logfile.write('decoding time: ' + str(end - start) + '\n')

    logfile.flush()

    logfile.close()

if __name__ == "__main__":
    main()
