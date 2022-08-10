import Model
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


model_qps = range(28) # 13 MSE models + 14 perceptual models + 1 lossless model
model_lambdas = [0.4, 0.25, 0.16, 0.10, 0.0625, 0.039, 0.024, 0.015, 0.0095, 0.006, 0.0037, 0.002, 0.0012,
                 0.4, 0.25, 0.16, 0.10, 0.018, 0.012, 0.0075, 0.0048, 0.0032, 0.002, 0.00145, 0.0008, 0.00055, 0.00035,
                 9999
                 ]
qp_shifts=[
[8, 7.5, 8.5, 7, 9, 6.5, 9.5],
[8, 7.5, 8.5, 7, 9, 6.5, 9.5],
[8, 7.5, 8.5, 7, 9, 6.5, 9.5],
[16, 15, 17, 14, 18, 13, 19],
[16, 15, 17, 14, 18, 13, 19],
[16, 15, 18, 14, 20, 13, 22],
[32, 30, 33, 34, 35, 36, 37],
[32, 31, 34, 30, 36, 29, 38],
[32, 30, 36, 28, 40, 26, 44],
[64, 60, 66, 56, 68, 70, 72],
[64, 62, 70, 58, 76, 54, 82],
[64, 58, 72, 52, 80, 46, 88],
[64, 56, 72, 48, 80, 40, 88],

[8, 7.5, 8.5, 7, 9, 6.5, 9.5],
[8, 7.5, 8.5, 7, 9, 6.5, 9.5],
[8, 7.5, 8.5, 7, 9, 6.5, 9.5],
[16, 15, 17, 14, 18, 13, 19],
[16, 15, 17, 14, 18, 13, 19],
[16, 15, 18, 14, 20, 13, 22],
[32, 30, 33, 34, 35, 36, 37],
[32, 31, 34, 30, 36, 29, 38],
[32, 30, 36, 28, 40, 26, 44],
[64, 60, 66, 56, 68, 52, 70],
[64, 62, 70, 60, 76, 58, 82],
[64, 58, 72, 52, 80, 46, 88],
[64, 56, 72, 48, 80, 40, 88],
[64, 56, 72, 48, 80, 40, 88],

[1]
]


def img2patch(x, h, w, stride):
    size = x.size()
    x_tmp = x[:, :, 0:h, 0:w]
    for i in range(0, size[2], stride):
        for j in range(0, size[3], stride):
            x_tmp = torch.cat((x_tmp, x[:, :, i:i+h, j:j+w]), dim=0)
    return x_tmp[size[0]::, :, :, :]


def patch2img(x, img_h, img_w):
    size = x.size()
    img = torch.zeros(3, 1, img_h, img_w).cuda()
    k = 0
    for i in range(img_h // size[2]):
        for j in range(img_w // size[3]):
            img[:, :, i*size[2]:(i+1)*size[2], j*size[3]:(j+1)*size[3]] = x[k*3:k*3+3, :, :, :]
            k = k + 1
    return img


def img2patch_padding(x, h, w, stride, padding):
    size = x.size()
    x_tmp = x[:, :, 0:h, 0:w]
    for i in range(0, size[2]-2*padding, stride):
        for j in range(0, size[3]-2*padding, stride):
            x_tmp = torch.cat((x_tmp, x[:, :, i:i+h, j:j+w]), dim=0)
    return x_tmp[size[0]::, :, :, :]


def rgb2yuv_lossless(x):
    x = np.array(x, dtype=np.int32)

    r = x[:, :, 0:1]
    g = x[:, :, 1:2]
    b = x[:, :, 2:3]

    yuv = np.zeros_like(x, dtype=np.int32)

    Co = r - b
    tmp = b + np.right_shift(Co, 1)
    Cg = g - tmp
    Y = tmp + np.right_shift(Cg, 1)

    yuv[:, :, 0:1] = Y
    yuv[:, :, 1:2] = Co
    yuv[:, :, 2:3] = Cg

    return yuv


def yuv2rgb_lossless(x):
    x = np.array(x, dtype=np.int32)

    Y = x[:, :, 0:1]
    Co = x[:, :, 1:2]
    Cg = x[:, :, 2:3]

    rgb = np.zeros_like(x, dtype=np.int32)

    tmp = Y - np.right_shift(Cg, 1)
    g = Cg + tmp
    b = tmp - np.right_shift(Co, 1)
    r = b + Co

    rgb[:, :, 0:1] = r
    rgb[:, :, 1:2] = g
    rgb[:, :, 2:3] = b

    return rgb


def rgb2yuv(x):
    convert_mat = np.array([[0.299, 0.587, 0.114],
                            [-0.169, -0.331, 0.499],
                            [0.499, -0.418, -0.0813]], dtype=np.float32)

    y = x[:, 0:1, :, :] * convert_mat[0, 0] +\
        x[:, 1:2, :, :] * convert_mat[0, 1] +\
        x[:, 2:3, :, :] * convert_mat[0, 2]

    u = x[:, 0:1, :, :] * convert_mat[1, 0] +\
        x[:, 1:2, :, :] * convert_mat[1, 1] +\
        x[:, 2:3, :, :] * convert_mat[1, 2] + 128.

    v = x[:, 0:1, :, :] * convert_mat[2, 0] +\
        x[:, 1:2, :, :] * convert_mat[2, 1] +\
        x[:, 2:3, :, :] * convert_mat[2, 2] + 128.
    return torch.cat((y, u, v), dim=1)


def yuv2rgb(x):
    inverse_convert_mat = np.array([[1.0, 0.0, 1.402],
                                    [1.0, -0.344, -0.714],
                                    [1.0, 1.772, 0.0]], dtype=np.float32)
    r = x[:, 0:1, :, :] * inverse_convert_mat[0, 0] +\
        (x[:, 1:2, :, :] - 128.) * inverse_convert_mat[0, 1] +\
        (x[:, 2:3, :, :] - 128.) * inverse_convert_mat[0, 2]
    g = x[:, 0:1, :, :] * inverse_convert_mat[1, 0] +\
        (x[:, 1:2, :, :] - 128.) * inverse_convert_mat[1, 1] +\
        (x[:, 2:3, :, :] - 128.) * inverse_convert_mat[1, 2]
    b = x[:, 0:1, :, :] * inverse_convert_mat[2, 0] +\
        (x[:, 1:2, :, :] - 128.) * inverse_convert_mat[2, 1] +\
        (x[:, 2:3, :, :] - 128.) * inverse_convert_mat[2, 2]
    return torch.cat((r, g, b), dim=1)


def find_min_and_max(LL, HL_list, LH_list, HH_list):

    min_v = [[1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000.],
             [1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000.],
             [1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000.]]
    max_v = [[-1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000.],
             [-1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000.],
             [-1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000.]]

    for channel_idx in range(3):
        tmp = LL[channel_idx, 0, :, :]
        min_tmp = torch.min(tmp).item()
        max_tmp = torch.max(tmp).item()
        if min_tmp < min_v[channel_idx][0]:
            min_v[channel_idx][0] = min_tmp
        if max_tmp > max_v[channel_idx][0]:
            max_v[channel_idx][0] = max_tmp

        for s_j in range(4):
            s_i = 4 - 1 - s_j
            tmp = HL_list[s_i][channel_idx, 0, :, :]
            min_tmp = torch.min(tmp).item()
            max_tmp = torch.max(tmp).item()
            if min_tmp < min_v[channel_idx][3 * s_i + 1]:
                min_v[channel_idx][3 * s_i + 1] = min_tmp
            if max_tmp > max_v[channel_idx][3 * s_i + 1]:
                max_v[channel_idx][3 * s_i + 1] = max_tmp

            tmp = LH_list[s_i][channel_idx, 0, :, :]
            min_tmp = torch.min(tmp).item()
            max_tmp = torch.max(tmp).item()
            if min_tmp < min_v[channel_idx][3 * s_i + 2]:
                min_v[channel_idx][3 * s_i + 2] = min_tmp
            if max_tmp > max_v[channel_idx][3 * s_i + 2]:
                max_v[channel_idx][3 * s_i + 2] = max_tmp

            tmp = HH_list[s_i][channel_idx, 0, :, :]
            min_tmp = torch.min(tmp).item()
            max_tmp = torch.max(tmp).item()
            if min_tmp < min_v[channel_idx][3 * s_i + 3]:
                min_v[channel_idx][3 * s_i + 3] = min_tmp
            if max_tmp > max_v[channel_idx][3 * s_i + 3]:
                max_v[channel_idx][3 * s_i + 3] = max_tmp
    min_v = (np.array(min_v)).astype(np.int)
    max_v = (np.array(max_v)).astype(np.int)
    return min_v, max_v
