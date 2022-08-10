import torch
import torch.optim as optim
from torch.autograd import Variable
import math
import sys
from torch.nn import functional as F

class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        return x.round()
    @staticmethod
    def backward(ctx, g):

        return g
# Usage: xq1 = RoundNoGradient.apply(x1)

class Quant(torch.nn.Module):
    def __init__(self):
        super(Quant, self).__init__()
    def forward(self, x, scale):
        return RoundNoGradient.apply(x/scale)

class DeQuant(torch.nn.Module):
    def __init__(self):
        super(DeQuant, self).__init__()
    def forward(self, x, scale):
        return x*scale


class soft_round_Quant(torch.nn.Module):
    def __init__(self):
        super(soft_round_Quant, self).__init__()

    def add_noise(self, x):
        shape = x.size()
        noise = torch.cuda.FloatTensor(shape)
        torch.rand(shape, out=noise)
        return x + noise - 0.5

    def s_a(self, x, alpha):
        if alpha < 1e-3:
            print("alpha is too small!")
            exit()
        alpha_bounded = alpha
        m = torch.floor(x) + 0.5
        r = x - m
        z = torch.tanh(alpha_bounded / 2.) * 2.
        y = m + torch.tanh(alpha_bounded * r) / z
        return y

    def forward(self, x, scale, alpha):
        x = x / scale
        return self.s_a(self.add_noise(self.s_a(x, alpha)), alpha)
        # return self.Sa(self.add_noise(self.Sa(x, alpha)), alpha)


class soft_and_hard_Quant(torch.nn.Module):
    def __init__(self):
        super(soft_and_hard_Quant, self).__init__()

    def s_a(self, x, alpha):
        # if alpha < 1e-3:
        #     print("alpha is too small!")
        #     exit()
        alpha_bounded = alpha
        m = torch.floor(x) + 0.5
        r = x - m
        z = torch.tanh(alpha_bounded / 2.) * 2.
        y = m + torch.tanh(alpha_bounded * r) / z
        return y

    def forward(self, x, scale, alpha):
        x = x / scale

        softQ = self.s_a(x, alpha)
        hardQ = torch.round(x)
        return softQ + (hardQ - softQ).detach()


class Scale_net(torch.nn.Module):
    def __init__(self):
        super(Scale_net, self).__init__()
        self.internal_channel = 16
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self):
        zero = torch.cuda.FloatTensor(1,1,1,1).fill_(0)
        out = self.conv1(zero)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        return out