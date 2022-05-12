from numpy import False_
from model.Unet_components import * 
import torch
from torch import nn

class UNet(nn.Module):
  def __init__(self,baseline, in_channels, nr_sources, use_dummy_conv = True, dummy_conv_size = 64, mode = 'implicit'):
    super(UNet,self).__init__()
    self.mode = mode
    if use_dummy_conv:
      print("Using Dummy Convolutions to prepare the data for U-NET")
      self.dummy1 = torch.nn.Conv2d(in_channels = in_channels, 
                              out_channels = dummy_conv_size,
                              kernel_size = (4,6), padding = (1,1), bias = False)
      self.dummy2 = torch.nn.Conv2d(in_channels = nr_sources,
                                    out_channels =  dummy_conv_size // 4,
                                    kernel_size = (2,2), padding = (1,1), bias = False)
      self.dummy2_1 = torch.nn.Conv2d(in_channels = dummy_conv_size // 4,
                                    out_channels =  dummy_conv_size,
                                    kernel_size = (3,2), padding = (1,1), bias = False)
      self.dummy2_2 = torch.nn.Conv2d(in_channels =  dummy_conv_size,
                                    out_channels = nr_sources,
                                    kernel_size = (3,2), padding = (1,1), bias = False)
    self.encoder1 = Down(dummy_conv_size, baseline)
    self.encoder2 = Down(baseline, baseline * 2)
    self.encoder3 = Down(baseline * 2, baseline * 4)
    self.encoder4 = Down(baseline * 4 , baseline * 8)
    self.bottleNeck = DoubleConv(baseline * 8, baseline * 16)
    self.decoder1 = Up(baseline * 16, baseline * 8)
    self.decoder2 = Up(baseline * 8, baseline * 4)
    self.decoder3 = Up(baseline * 4, baseline * 2)
    self.decoder4 = Up(baseline * 2, baseline)
    self.spectrogram = DoubleConv(baseline,nr_sources)
    self.relu = nn.ReLU(inplace = False)
    self.batchnorm1 = nn.BatchNorm2d(dummy_conv_size)
    self.batchnorm3 = nn.BatchNorm2d(dummy_conv_size // 4)
    self.batchnorm4 = nn.BatchNorm2d(dummy_conv_size)
  def forward(self,X):
    # First dummy convolution to prepare data for U_Net
    X = self.dummy1(X)
    X = self.batchnorm1(X)
    X = self.relu(X)
    X,X_connection0 = self.encoder1(X)
    X,X_connection1 = self.encoder2(X)
    X,X_connection2 = self.encoder3(X)
    X,X_connection3 = self.encoder4(X)
    X = self.bottleNeck(X)
    X = self.decoder1(X,X_connection3)
    X = self.decoder2(X,X_connection2)
    X = self.decoder3(X,X_connection1)
    X = self.decoder4(X,X_connection0)
    X = self.spectrogram(X)

    # Last Dummy Convolution to output spectrogram
    X = self.dummy2(X)
    X = self.batchnorm3(X)
    X = self.relu(X)
    X = self.dummy2_1(X)
    X = self.batchnorm4(X)
    X = self.relu(X)
    X = self.dummy2_2(X)
    X = self.relu(X) if self.mode == 'implicit' else torch.sigmoid(X)
    
    return X