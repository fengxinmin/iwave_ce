import torch
import torch.nn as nn


class APNet(torch.nn.Module):
    def __init__(self):
        super(APNet, self).__init__()

        self.ap1 = APNet_1()
        self.ap2 = APNet_2()
        self.ap4 = APNet_4()
        self.ap8 = APNet_8()

    def forward(self, x, y, if_D):
        if if_D:
            # out_x = self.ap1(x)
            # out_y = self.ap1(y)
            # loss = torch.mean(out_y - out_x)

            out_x = self.ap2(x)
            out_y = self.ap2(y)
            loss = torch.mean(out_y - out_x)

            # out_x = self.ap4(x)
            # out_y = self.ap4(y)
            # loss += torch.mean(out_y - out_x)

            # out_x = self.ap8(x)
            # out_y = self.ap8(y)
            # loss += torch.mean(out_y - out_x)
        else:
            # out_x = self.ap1(x)
            # out_y = self.ap1(y)
            # loss = torch.mean(torch.abs(out_x - out_y)) + torch.mean(out_y - out_x)

            out_x = self.ap2(x)
            out_y = self.ap2(y)
            loss = torch.mean(torch.abs(out_x - out_y)) + torch.mean(out_y - out_x)

            # out_x = self.ap4(x)
            # out_y = self.ap4(y)
            # loss += torch.mean(torch.abs(out_x - out_y)) + torch.mean(out_y - out_x)

            # out_x = self.ap8(x)
            # out_y = self.ap8(y)
            # loss += torch.mean(torch.abs(out_x - out_y)) + torch.mean(out_y - out_x)

        return loss


class APNet_8(torch.nn.Module):
    def __init__(self, internal_channel=64):
        super(APNet_8, self).__init__()

        self.internal_channel = internal_channel
        def main_branch():
            return torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=3, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0),


                torch.nn.ReLU(inplace=False),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0),

                torch.nn.ReLU(inplace=False),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3,
                                stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3,
                                stride=1, padding=0),

                torch.nn.ReLU(inplace=False),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3,
                                stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3,
                                stride=1, padding=0),
            )

        self.main_ = main_branch()

    def forward(self, x):
        out = self.main_(x)
        return out


class APNet_4(torch.nn.Module):
    def __init__(self, internal_channel=64):
        super(APNet_4, self).__init__()

        self.internal_channel = internal_channel
        def main_branch():
            return torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=3, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0),

                torch.nn.ReLU(inplace=False),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3,
                                stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3,
                                stride=1, padding=0),

                torch.nn.ReLU(inplace=False),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3,
                                stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3,
                                stride=1, padding=0),
            )

        self.main_ = main_branch()

    def forward(self, x):
        out = self.main_(x)
        return out


class APNet_2(torch.nn.Module):
    def __init__(self, internal_channel=64):
        super(APNet_2, self).__init__()

        self.internal_channel = internal_channel
        def main_branch():
            return torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=3, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0),

                torch.nn.ReLU(inplace=False),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3,
                                stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3,
                                stride=1, padding=0),

            )

        self.main_ = main_branch()

    def forward(self, x):
        out = self.main_(x)
        return out


class APNet_1(torch.nn.Module):
    def __init__(self, internal_channel=64):
        super(APNet_1, self).__init__()

        self.internal_channel = internal_channel
        def main_branch():
            return torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=3, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0),
            )

        self.main_ = main_branch()

    def forward(self, x):
        out = self.main_(x)
        return out