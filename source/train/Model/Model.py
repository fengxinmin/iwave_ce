import torch
import torch.optim as optim
from torch.autograd import Variable
import math
import sys
from torch.nn import functional as F
import numpy as np

import sys
sys.path.append("..")
import Util.Quant as Quant
from Model.rcan import RCAN as PostProcessing
import Model.PixelCNN as PixelCNN
import Model.learn_wavelet_trans_additive as learn_wavelet_trans_additive
import Model.learn_wavelet_trans_affine as learn_wavelet_trans_affine
import Model.wavelet_trans as wavelet_trans

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
    return torch.cat((y, u, v), dim=0)


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


class Model(torch.nn.Module):
    def __init__(self, wavelet_affine):
        super(Model, self).__init__()

        self.trans_steps = 4
        self.scale_net = Quant.Scale_net()
        self.wavelet_affine = wavelet_affine
        self.wavelet_transform_97 = wavelet_trans.Wavelet()
        if self.wavelet_affine:
            self.wavelet_transform_trainable = torch.nn.ModuleList(
                learn_wavelet_trans_affine.Wavelet(True) for _i in range(self.trans_steps))
        else:
            self.wavelet_transform_trainable = learn_wavelet_trans_additive.Wavelet(True)

        self.soft_round_quant = Quant.soft_round_Quant()

        self.coding_LL = PixelCNN.PixelCNN()
        self.coding_HL_list = torch.nn.ModuleList([PixelCNN.PixelCNN_Context(1) for _i in range(self.trans_steps)])
        self.coding_LH_list = torch.nn.ModuleList([PixelCNN.PixelCNN_Context(2) for _i in range(self.trans_steps)])
        self.coding_HH_list = torch.nn.ModuleList([PixelCNN.PixelCNN_Context(3) for _i in range(self.trans_steps)])
        
        self.mse_loss = torch.nn.MSELoss()
        self.post = PostProcessing(n_resgroups=10, n_resblocks=10, n_feats=32)

    def forward(self, x, alpha, train, scale_init, wavelet_trainable, coding=1):
        
        self.scale = bool(wavelet_trainable) * self.scale_net() + scale_init

        size = x.size()
        yuv_x = rgb2yuv(x)
        # forward transform
        LL = yuv_x
        HL_list = []
        LH_list = []
        HH_list = []
        for i in range(self.trans_steps):
            if wavelet_trainable:
                if self.wavelet_affine:
                    LL, HL, LH, HH = self.wavelet_transform_trainable[i].forward_trans(LL)
                else:
                    LL, HL, LH, HH = self.wavelet_transform_trainable.forward_trans(LL)
            else:
                LL, HL, LH, HH = self.wavelet_transform_97.forward_trans(LL)

            if train and wavelet_trainable:
                HL_list.append(self.soft_round_quant.forward(HL, self.scale, alpha))
                LH_list.append(self.soft_round_quant.forward(LH, self.scale, alpha))
                HH_list.append(self.soft_round_quant.forward(HH, self.scale, alpha))
            else:
                HL_list.append(torch.round(HL / self.scale))
                LH_list.append(torch.round(LH / self.scale))
                HH_list.append(torch.round(HH / self.scale))
        if train and wavelet_trainable:
            LL = self.soft_round_quant.forward(LL, self.scale, alpha)
        else:
            LL = torch.round(LL / self.scale)

        if not coding:
            bits = torch.cuda.FloatTensor((20,1))*0.0
        else:
            bits = self.coding_LL(LL)
        LL = LL * self.scale

        for i in range(self.trans_steps):
            j = self.trans_steps - 1 - i

            if coding:
                bits = bits + self.coding_HL_list[j](HL_list[j],LL)
            HL_list[j] = HL_list[j] * self.scale

            if coding:
                bits = bits + self.coding_LH_list[j](LH_list[j], torch.cat((LL, HL_list[j]),1))
            LH_list[j] = LH_list[j] * self.scale

            if coding:
                bits = bits + self.coding_HH_list[j](HH_list[j], torch.cat((LL, HL_list[j], LH_list[j]),1))
            HH_list[j] = HH_list[j] * self.scale

            if wavelet_trainable:
                if self.wavelet_affine:
                    LL = self.wavelet_transform_trainable[j].inverse_trans(LL, HL_list[j], LH_list[j], HH_list[j])
                else:
                    LL = self.wavelet_transform_trainable.inverse_trans(LL, HL_list[j], LH_list[j], HH_list[j])
            else:
                LL = self.wavelet_transform_97.inverse_trans(LL, HL_list[j], LH_list[j], HH_list[j])

        batch_size = size[0]
        rgb_org = yuv2rgb(torch.cat((LL[:batch_size], LL[batch_size:2*batch_size], LL[2*batch_size:]), 1))
        rgb_post = self.post(rgb_org)

        return self.mse_loss(x, rgb_org), self.mse_loss(x, rgb_post), bits, self.scale, rgb_post
