import torch
from torch import nn

class Discriminator(nn.Module):
  '''
  Patch-GAN discriminator to classify the image segments as real or fake
  '''
  def __init__(self,nr_sources = 1, in_channels = 1, baseline = 3):
    super(Discriminator,self).__init__()
    self.conv1 = nn.Conv2d(in_channels = nr_sources + in_channels, 
                           out_channels = baseline,
                           kernel_size = (3,3),
                           stride=(2,2),
                           padding = (1,1),
                           bias = False)
    self.conv2 = nn.Conv2d(in_channels = baseline, 
                           out_channels = baseline * 2,
                           kernel_size = (3,3),
                           stride=(2,2),
                           padding = (1,1), 
                           bias = False)
    self.conv3 = nn.Conv2d(in_channels = baseline * 2, 
                           out_channels = baseline * 4,
                           kernel_size = (3,3),
                           stride = (2,2),
                           padding = (1,1),
                           bias = False)
    self.conv4 = nn.Conv2d(in_channels = baseline * 4, 
                           out_channels = baseline* 2,
                           kernel_size = (3,3),
                           stride = (2,2),
                           padding = (1,1),
                           bias = False)
    self.conv5 = nn.Conv2d(in_channels = baseline * 2, 
                           out_channels = nr_sources,
                           kernel_size = (5,5)
                           )
    self.batchNorm1 = nn.BatchNorm2d(baseline)
    self.batchNorm2 = nn.BatchNorm2d(baseline * 2)
    self.batchNorm3 = nn.BatchNorm2d(baseline* 4)
    self.batchNorm4 = nn.BatchNorm2d(baseline * 2)

    self.lrelu = nn.LeakyReLU()
    self.sigmoid = nn.Sigmoid()
    self.net = nn.Sequential(self.conv1, 
                             self.batchNorm1,
                             self.lrelu,
                             self.conv2, 
                             self.batchNorm2,
                             self.lrelu,
                             self.conv3,
                             self.batchNorm3,
                             self.lrelu,
                             self.conv4,
                             self.batchNorm4,
                             self.lrelu,
                             self.conv5
                             )
  def forward(self,inp,target):
      inp = torch.concat((inp,target), dim = 1)
      return self.net(inp)