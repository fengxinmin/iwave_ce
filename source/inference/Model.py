import torch
from torch.autograd import Variable
import math
import sys
from torch.nn import functional as F

import learn_wavelet_trans_affine
import learn_wavelet_trans_additive
import learn_wavelet_trans_lossless
import PixelCNN
import PixelCNN_lossless
from rcan import RCAN as PostProcessing
from gan_post import GANPostProcessing
import Quant


class Transform_lossless(torch.nn.Module):
    def __init__(self, trainable_set=False):
        super(Transform_lossless, self).__init__()

        self.trans_steps = 4

        self.trainable_set = trainable_set

        self.wavelet_transform = torch.nn.ModuleList(
            learn_wavelet_trans_lossless.Wavelet(self.trainable_set) for _i in range(self.trans_steps))

    def forward_trans(self, x):
        LL = x
        HL_list = []
        LH_list = []
        HH_list = []
        for i in range(self.trans_steps):
            LL, HL, LH, HH = self.wavelet_transform[i].forward_trans(LL)
            HL_list.append(HL)
            LH_list.append(LH)
            HH_list.append(HH)

        return LL, HL_list, LH_list, HH_list

    def inverse_trans(self, LL, HL, LH, HH, layer):

        LL = self.wavelet_transform[layer].inverse_trans(LL, HL, LH, HH)

        return LL

class CodingLL_lossless(torch.nn.Module):
    def __init__(self):
        super(CodingLL_lossless, self).__init__()

        self.coding_LL = PixelCNN_lossless.PixelCNN()

    def forward(self, LL, lower_bound, upper_bound):
        prob = self.coding_LL(LL, lower_bound, upper_bound)
        return prob


class CodingHL_lossless(torch.nn.Module):
    def __init__(self):
        super(CodingHL_lossless, self).__init__()

        self.trans_steps = 4

        self.coding_HL_list = torch.nn.ModuleList([PixelCNN_lossless.PixelCNN_Context(1) for _i in range(self.trans_steps)])

    def forward(self, HL, context, lower_bound, upper_bound, layer):
        prob = self.coding_HL_list[layer](HL, context, lower_bound, upper_bound)
        return prob


class CodingLH_lossless(torch.nn.Module):
    def __init__(self):
        super(CodingLH_lossless, self).__init__()

        self.trans_steps = 4

        self.coding_LH_list = torch.nn.ModuleList([PixelCNN_lossless.PixelCNN_Context(2) for _i in range(self.trans_steps)])

    def forward(self, LH, context, lower_bound, upper_bound, layer):
        prob = self.coding_LH_list[layer](LH, context, lower_bound, upper_bound)
        return prob



class CodingHH_lossless(torch.nn.Module):
    def __init__(self):
        super(CodingHH_lossless, self).__init__()

        self.trans_steps = 4

        self.coding_HH_list = torch.nn.ModuleList([PixelCNN_lossless.PixelCNN_Context(3) for _i in range(self.trans_steps)])

    def forward(self, HH, context, lower_bound, upper_bound, layer):
        prob = self.coding_HH_list[layer](HH, context, lower_bound, upper_bound)
        return prob


class Transform_aiWave(torch.nn.Module):
    def __init__(self, scale, isAffine, trainable_set=True):
        super(Transform_aiWave, self).__init__()

        self.trans_steps = 4

        self.trainable_set = trainable_set
        self.isAffine = isAffine

        if isAffine:
            self.wavelet_transform = torch.nn.ModuleList(
                learn_wavelet_trans_affine.Wavelet(trainable_set) for _i in range(self.trans_steps))
        else:
            self.wavelet_transform = learn_wavelet_trans_additive.Wavelet(trainable_set)

        self.scale_net = Quant.Scale_net()
        self.scale_init = scale

    def forward_trans(self, x):

        scale = float(self.trainable_set) * self.scale_net() + self.scale_init

        LL = x
        HL_list = []
        LH_list = []
        HH_list = []
        for i in range(self.trans_steps):
            if self.isAffine:
                LL, HL, LH, HH = self.wavelet_transform[i].forward_trans(LL)
            else:
                LL, HL, LH, HH = self.wavelet_transform.forward_trans(LL)
            HL_list.append(torch.round(HL/scale))
            LH_list.append(torch.round(LH/scale))
            HH_list.append(torch.round(HH/scale))

        LL = torch.round(LL/scale)

        return LL, HL_list, LH_list, HH_list, scale

    def inverse_trans(self, LL, HL, LH, HH, layer):

        if self.isAffine:
            LL = self.wavelet_transform[layer].inverse_trans(LL, HL, LH, HH)
        else:
            LL = self.wavelet_transform.inverse_trans(LL, HL, LH, HH)

        return LL

    def used_scale(self):
        scale = float(self.trainable_set) * self.scale_net() + self.scale_init
        return scale


class CodingLL(torch.nn.Module):
    def __init__(self):
        super(CodingLL, self).__init__()

        self.coding_LL = PixelCNN.PixelCNN()

    def forward(self, LL, lower_bound, upper_bound):
        prob = self.coding_LL(LL, lower_bound, upper_bound)
        return prob


class CodingHL(torch.nn.Module):
    def __init__(self):
        super(CodingHL, self).__init__()

        self.trans_steps = 4

        self.coding_HL_list = torch.nn.ModuleList([PixelCNN.PixelCNN_Context(1) for _i in range(self.trans_steps)])

    def forward(self, HL, context, lower_bound, upper_bound, layer):
        prob = self.coding_HL_list[layer](HL, context, lower_bound, upper_bound)
        return prob


class CodingLH(torch.nn.Module):
    def __init__(self):
        super(CodingLH, self).__init__()

        self.trans_steps = 4

        self.coding_LH_list = torch.nn.ModuleList([PixelCNN.PixelCNN_Context(2) for _i in range(self.trans_steps)])

    def forward(self, LH, context, lower_bound, upper_bound, layer):
        prob = self.coding_LH_list[layer](LH, context, lower_bound, upper_bound)
        return prob



class CodingHH(torch.nn.Module):
    def __init__(self):
        super(CodingHH, self).__init__()

        self.trans_steps = 4

        self.coding_HH_list = torch.nn.ModuleList([PixelCNN.PixelCNN_Context(3) for _i in range(self.trans_steps)])

    def forward(self, HH, context, lower_bound, upper_bound, layer):
        prob = self.coding_HH_list[layer](HH, context, lower_bound, upper_bound)
        return prob


class Post(torch.nn.Module):
    def __init__(self):
        super(Post, self).__init__()

        self.post = PostProcessing(n_resgroups=10, n_resblocks=10, n_feats=32)

    def forward(self, x):

        post_recon = self.post(x)

        return post_recon


class PostGAN(torch.nn.Module):
    def __init__(self):
        super(PostGAN, self).__init__()

        self.gan_post = GANPostProcessing()

    def forward(self, x):

        post_recon = self.gan_post(x)

        return post_recon