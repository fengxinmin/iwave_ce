import torch
import torch.optim as optim
from torch.autograd import Variable
import math
import sys
from torch.nn import functional as F

class Lifting_Forward(torch.nn.Module):
    def __init__(self):
        super(Lifting_Forward, self).__init__()
        # Filter coefficients of bior4.4 wavelet (Orthogonal form of CDF 9/7 wavelet）
        self.lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 0.869864451624781, 1.149604398860241]

    def forward(self, L, H):
        # L[1:end+1] + L[0:end]
        paddings = (0,0,0,1)
        tmp = F.pad(L, paddings, "reflect")
        tmp = tmp[:,:,1::,:]
        H = H + self.lifting_coeff[0]*(L+tmp)

        paddings = (0, 0, 1, 0)
        tmp = F.pad(H, paddings, 'reflect')
        tmp = tmp[:,:,0:-1,:]
        L = L + self.lifting_coeff[1]*(H+tmp)

        paddings = (0, 0, 0, 1)
        tmp = F.pad(L, paddings, "reflect")
        tmp = tmp[:, :, 1::, :]
        H = H + self.lifting_coeff[2] * (L + tmp)

        paddings = (0, 0, 1, 0)
        tmp = F.pad(H, paddings, 'reflect')
        tmp = tmp[:, :, 0:-1, :]
        L = L + self.lifting_coeff[3] * (H + tmp)

        L = self.lifting_coeff[5] * L
        H = self.lifting_coeff[4] * H

        return L, H
class Lifting_Inverse(torch.nn.Module):
    def __init__(self):
        super(Lifting_Inverse, self).__init__()
        # Filter coefficients of bior4.4 wavelet (Orthogonal form of CDF9/7）
        self.lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 0.869864451624781, 1.149604398860241]

    def forward(self, L, H):
        L =  L / self.lifting_coeff[5]
        H =  H / self.lifting_coeff[4]

        paddings = (0, 0, 1, 0)
        tmp = F.pad(H, paddings, 'reflect')
        tmp = tmp[:, :, 0:-1, :]
        L = L - self.lifting_coeff[3] * (H + tmp)

        paddings = (0, 0, 0, 1)
        tmp = F.pad(L, paddings, "reflect")
        tmp = tmp[:, :, 1::, :]
        H = H - self.lifting_coeff[2] * (L + tmp)

        paddings = (0, 0, 1, 0)
        tmp = F.pad(H, paddings, 'reflect')
        tmp = tmp[:, :, 0:-1, :]
        L = L - self.lifting_coeff[1] * (H + tmp)

        paddings = (0, 0, 0, 1)
        tmp = F.pad(L, paddings, "reflect")
        tmp = tmp[:, :, 1::, :]
        H = H - self.lifting_coeff[0] * (L + tmp)

        return L, H

class Wavelet_Forward(torch.nn.Module):
    def __init__(self):
        super(Wavelet_Forward, self).__init__()

        self.lifting_f = Lifting_Forward()
    def forward(self, x):
        # transform for rows
        L = x[:,:,0::2,:]
        H = x[:,:,1::2,:]
        L, H = self.lifting_f(L, H)

        L = L.permute(0,1,3,2)
        LL = L[:,:,0::2,:]
        HL = L[:,:,1::2,:]
        LL, HL = self.lifting_f(LL, HL)
        LL = LL.permute(0,1,3,2)
        HL = HL.permute(0,1,3,2)

        H = H.permute(0,1,3,2)
        LH = H[:,:,0::2,:]
        HH = H[:,:,1::2,:]
        LH, HH = self.lifting_f(LH, HH)
        LH = LH.permute(0,1,3,2)
        HH = HH.permute(0,1,3,2)

        return LL, HL, LH, HH

class Wavelet_Inverse(torch.nn.Module):
    def __init__(self):
        super(Wavelet_Inverse, self).__init__()

        self.lifting_i = Lifting_Inverse()
    def forward(self, LL, HL, LH, HH):

        LH = LH.permute(0, 1, 3, 2)
        HH = HH.permute(0, 1, 3, 2)
        H = torch.zeros(LH.size()[0], LH.size()[1], LH.size()[2] + HH.size()[2], LH.size()[3]).cuda()
        LH, HH = self.lifting_i(LH, HH)
        H[:, :, 0::2, :] = LH
        H[:, :, 1::2, :] = HH
        H = H.permute(0, 1, 3, 2)

        LL = LL.permute(0, 1, 3, 2)
        HL = HL.permute(0, 1, 3, 2)
        L = torch.zeros(LL.size()[0], LL.size()[1], LL.size()[2] + HL.size()[2], LL.size()[3]).cuda()
        LL, HL = self.lifting_i(LL, HL)
        L[:, :, 0::2, :] = LL
        L[:, :, 1::2, :] = HL
        L = L.permute(0, 1, 3, 2)

        L, H = self.lifting_i(L, H)
        x = torch.zeros(L.size()[0], L.size()[1], L.size()[2] + H.size()[2], L.size()[3]).cuda()
        x[:, :, 0::2, :] = L
        x[:, :, 1::2, :] = H

        return x


class Wavelet(torch.nn.Module):
    def __init__(self):
        super(Wavelet, self).__init__()

        self.forward_t = Wavelet_Forward()
        self.inverse_t = Wavelet_Inverse()

    def forward_trans(self, x):

        return self.forward_t(x)

    def inverse_trans(self, LL, HL, LH, HH):

        return self.inverse_t(LL, HL, LH, HH)
    