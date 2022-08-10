import torch
import torch.optim as optim
from torch.autograd import Variable
import math
import sys
from torch.nn import functional as F
import numpy as np

import Model.PixelCNN_lossless as PixelCNN
import Model.learn_wavelet_trans_lossless as learn_wavelet_trans


class Model(torch.nn.Module):
    def __init__(self, trainable_set):
        super(Model, self).__init__()

        self.trans_steps = 4
        self.trainable_set = trainable_set
        self.wavelet_transform = torch.nn.ModuleList(learn_wavelet_trans.Wavelet(self.trainable_set)  for _i in range(self.trans_steps))

        self.coding_LL = PixelCNN.PixelCNN()
        self.coding_HL_list = torch.nn.ModuleList([PixelCNN.PixelCNN_Context(1) for _i in range(self.trans_steps)])
        self.coding_LH_list = torch.nn.ModuleList([PixelCNN.PixelCNN_Context(2) for _i in range(self.trans_steps)])
        self.coding_HH_list = torch.nn.ModuleList([PixelCNN.PixelCNN_Context(3) for _i in range(self.trans_steps)])

    def forward(self, x):
        # forward transform
        LL = x
        HL_list = []
        LH_list = []
        HH_list = []
        for i in range(self.trans_steps):

            LL, HL, LH, HH = self.wavelet_transform[i].forward_trans(LL)
            HL_list.append(HL)
            LH_list.append(LH)
            HH_list.append(HH)

        bits = self.coding_LL(LL)

        for i in range(self.trans_steps):

            j = self.trans_steps - 1 - i

            bits = bits + self.coding_HL_list[j](HL_list[j],LL)

            bits = bits + self.coding_LH_list[j](LH_list[j], torch.cat((LL, HL_list[j]),1))

            bits = bits + self.coding_HH_list[j](HH_list[j], torch.cat((LL, HL_list[j], LH_list[j]),1))

            LL = self.wavelet_transform[j].inverse_trans(LL, HL_list[j], LH_list[j], HH_list[j])
        
        
        return bits, LL

