import torch
import torch.optim as optim
from torch.autograd import Variable
import math
import sys
from torch.nn import functional as F
import numpy as np

class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = torch.logical_or(x >= 1e-6, g < 0.0)
        t = pass_through_if+0.0

        return grad1 * t

class Distribution_for_entropy2(torch.nn.Module):
    def __init__(self):
        super(Distribution_for_entropy2, self).__init__()

    def forward(self, x, p_dec): 
        channel = p_dec.size()[1]
        if channel % 3 != 0:
            raise ValueError(
                "channel number must be multiple of 3")
        gauss_num = channel // 3
        temp = torch.chunk(p_dec, channel, dim=1)
        
        # keep the weight  summation of prob == 1
        probs = torch.cat(temp[gauss_num*2: ], dim=1)
        probs = F.softmax(probs, dim=1)
        
        
        # process the scale value to non-zero  如果为0，就设为最小值1*e-6
        for i in range(gauss_num, gauss_num*2):
            # temp[i] = torch.abs(temp[i])
            temp[i][temp[i] == 0] = 1e-6

        gauss_list = []
        for i in range(gauss_num):
            gauss_list.append(torch.distributions.normal.Normal(temp[i], temp[i+gauss_num]))

        likelihood_list = []
        for i in range(gauss_num):
            likelihood_list.append(torch.abs(gauss_list[i].cdf(x + 0.5/255.)-gauss_list[i].cdf(x-0.5/255.)))

        likelihoods = 0
        for i in range(gauss_num):
            likelihoods += probs[:,i:i+1,:,:] * likelihood_list[i]
            
        return likelihoods

class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class MaskResBlock(torch.nn.Module):
    def __init__(self, internal_channel):
        super(MaskResBlock, self).__init__()

        self.conv1 = MaskedConv2d('B', in_channels=internal_channel, out_channels=internal_channel, kernel_size=3, stride=1, padding=0)
        self.conv2 = MaskedConv2d('B', in_channels=internal_channel, out_channels=internal_channel, kernel_size=3, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x[:,:,2:-2,2:-2]


class ResBlock(torch.nn.Module):
    def __init__(self, internal_channel):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=internal_channel, out_channels=internal_channel, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=internal_channel, out_channels=internal_channel, kernel_size=3, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1((x))
        out = self.relu(out)
        out = self.conv2((out))
        return out + x[:,:,2:-2,2:-2]


class PixelCNN(torch.nn.Module):
    def __init__(self):
        super(PixelCNN, self).__init__()

        self.internal_channel = 128
        self.num_params = 9

        self.relu = torch.nn.ReLU(inplace=False)

        self.padding_constant = torch.nn.ConstantPad2d(6, 0)
        self.conv_pre = MaskedConv2d('A', in_channels=1, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)
        self.res1 = MaskResBlock(self.internal_channel)
        self.res2 = MaskResBlock(self.internal_channel)
        self.conv_post = MaskedConv2d('B', in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1,padding=0)

        def infering():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.num_params, kernel_size=1, stride=1, padding=0)
            )
        self.infer = infering()
        self.gaussin_entropy_func = Distribution_for_entropy2()

    def forward(self, x):
        x = x / 255.

        lable = x

        x = self.padding_constant(x)
        x = self.conv_pre(x)
        conv1 = x
        x = self.res1(x)
        x = self.res2(x)
        x = conv1[:,:,4:-4,4:-4] + x
        x = self.conv_post(x)
        x = self.relu(x)

        params = self.infer(x)
        
        prob = self.gaussin_entropy_func(lable, params)
        prob = Low_bound.apply(prob)

        bits = -torch.sum(torch.log2(prob))

        return bits


class PixelCNN_Context(torch.nn.Module):
    def __init__(self, context_num):
        super(PixelCNN_Context, self).__init__()

        self.internal_channel = 128
        self.num_params = 9

        self.relu = torch.nn.ReLU(inplace=False)

        self.conv_pre = MaskedConv2d('A', in_channels=1, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)
        self.res1 = MaskResBlock(self.internal_channel)
        self.res2 = MaskResBlock(self.internal_channel)
        self.conv_post = MaskedConv2d('B', in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1,padding=0)

        self.padding_reflect = torch.nn.ReflectionPad2d(6)
        self.padding_constant = torch.nn.ConstantPad2d(6, 0)
        self.conv_pre_c = torch.nn.Conv2d(in_channels=context_num, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)
        self.res1_c = ResBlock(self.internal_channel)
        self.res2_c = ResBlock(self.internal_channel)

        def infering():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.num_params, kernel_size=1, stride=1, padding=0)
            )
        self.infer = infering()
        self.gaussin_entropy_func = Distribution_for_entropy2()

    def forward(self, x, context):
        x = x / 255.
        context = context / 255.

        lable = x
        x = self.padding_constant(x)
        context = self.padding_reflect(context)
        x = self.conv_pre(x)
        conv1 = x
        context = self.conv_pre_c((context))
        x = x + context

        x = self.res1(x)
        context = self.res1_c(context)
        x = x + context
        x = self.res2(x)
        context = self.res2_c(context)
        x = x + context

        x = conv1[:,:,4:-4,4:-4] + x
        x = self.conv_post(x)
        x = self.relu(x)

        params = self.infer(x)
        
        prob = self.gaussin_entropy_func(lable, params)
        prob = Low_bound.apply(prob)
        bits = -torch.sum(torch.log2(prob))

        return bits


