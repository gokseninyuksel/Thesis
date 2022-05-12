import torch
from torch import nn

class DoubleConv(nn.Module):
  def __init__(self,in_channels,out_channels, use_batchNorm = True):
    super(DoubleConv,self).__init__()
    self.conv1 = nn.Conv2d(in_channels =in_channels, 
                           out_channels = out_channels, 
                           kernel_size = (3,3),
                           bias = False if use_batchNorm else True,
                           padding = 'same')
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace = False)
    self.net = nn.Sequential(self.conv1,
                            self.bn1,
                            self.relu
                            )
  def forward(self,X):
    return self.net(X)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down,self).__init__()
        self.double_conv = DoubleConv(in_channels,out_channels)
        # GAN, use strided convolution to learn the down sampling.
        self.down_sample = nn.MaxPool2d(kernel_size = (2,2))
    def forward(self, X):
        encoded = self.double_conv(X)
        return self.down_sample(encoded),encoded
  
class Up(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(Up,self).__init__()
    self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
    self.conv1 = DoubleConv(in_channels + in_channels // 2,out_channels)
    self.relu = nn.ReLU(inplace = False)
  def forward(self,X, X_connection):
    X = self.upSample(X)
    X = self.conv1(torch.cat((X,X_connection), dim = 1))
    return self.relu(X)