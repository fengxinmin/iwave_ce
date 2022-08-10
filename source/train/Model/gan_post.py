import torch

class ResBlock(torch.nn.Module):
    def __init__(self, internal_channel=64):
        super(ResBlock, self).__init__()

        self.internal_channel = internal_channel
        self.padding = torch.nn.ReflectionPad2d(1) 
        self.conv1 = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)


    def forward(self, x):
        out = self.conv1(self.padding(x))
        out = self.relu(out)
        out = self.conv2(self.padding(out))

        return x + out

class GANPostProcessing(torch.nn.Module):
    def __init__(self, internal_channel=128):
        super(GANPostProcessing, self).__init__()

        self.internal_channel = internal_channel
        self.padding = torch.nn.ReflectionPad2d(1)
        
        body = [ResBlock(self.internal_channel) for _i in range(9)]
        self.body = torch.nn.Sequential(*body)

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=3, kernel_size=3, stride=1, padding=0)
        
        self.relu = torch.nn.ReLU()
        
    def forward(self, resi):
        conv1 = self.conv1(self.padding(resi))
        body = self.body(conv1)
        conv2 = self.conv2(self.padding((body)))

        return resi + conv2
