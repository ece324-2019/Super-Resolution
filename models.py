import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# Activation function used for both networks (from paper: https://arxiv.org/pdf/1710.05941.pdf)
def swish_actf(x):
    return x * F.sigmoid(x)



# Model (Generator)
# Can experiment with number of residual blocks, channel numbers, kernel size, and stride number. 

class Generator1(nn.Module):
  def __init__(self, num_blocks=16, upsample_factor=2, out_channel_num=64, kernel_size=3, stride_num=1):
    super(Generator1, self).__init__()
    self.name = "Generator"
    self.num_blocks = num_blocks
    self.upsample_factor = upsample_factor
    self.out_channel_num = out_channel_num
    self.kernel_size = kernel_size
    self.stride_num = stride_num

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.out_channel_num, kernel_size=self.kernel_size, stride=self.stride_num, padding=self.kernel_size/2) # for upscaling the image before feeding it into residual blocks
    
    for i in range(self.num_blocks):
      self.add_module('residual_block' + str(i+1), residual_block())
    
    self.conv2 = nn.Conv2d(in_channels=out_channel_num, out_channels=self.out_channel_num, kernel_size=self.kernel_size, stride=self.stride_num, padding=self.kernel_size/2)
    self.bn2 = nn.BatchNorm2d(out_channel_num)

    for j in range(self.upsample_factor/2):
      self.add_module('upsample_block' + str(j+1), upsample_block(in_channel=self.out_channel_num, kernels=3, strides=1, up_scale_factor=upsample_factor))

    self.conv3 = nn.Conv2d(in_channels=self.out_channel_num, out_channels=3, kernel_size=9, stride=self.stride_num, padding=4) ######################################### kernel size and padding???

  def forward(self, x):
    x = swish_actf(self.conv1(x))
    x1 = x.copy()
    for i in range(self.n_residual_blocks):
      x = self.__getattr__('residual_block' + str(i+1))(x)

    x = x1 + self.bn2(self.conv2(x))

    for i in range(self.upsample_factor/2):
      x = self.__getattr__('upsample_block' + str(i+1))(x)

    return self.conv3(x)
    
# Notes:
# Batch normalization is used to decrease variance when training deeper network
# Residual blocks are the recursive CNN blocks that helpful when training deep network
# Upscaling before feeding into the residual CNN blocks is helpful in making the model learn the filters first (from paper: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network: https://arxiv.org/pdf/1609.04802.pdf)


# The following will be called multiple times in the Generator
class residual_block(nn.Module):
  def __init__(self, num_channel=64, kernels=3, strides=1):
    super(residual_block, self).__init__()
    self.name = "residual_block"
    self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=kernels, stride=strides, padding=kernels/2)
    self.batch_norm1 = nn.BatchNorm2d(num_channel)
    self.conv2 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=kernels, stride=strides, padding=kernels/2)
    self.batch_norm2 = nn.BatchNorm2d(num_channel)

  def forward(self, x):
    y = swish_actf(self.batch_norm1(self.conv1(x)))
    return x + self.batch_norm2(self.conv2(y))


# Upsampling the image at the end of the Generator
class upsample_block(nn.Module):
  def __init__(self, in_channel=64, kernels=3, strides=1, up_scale_factor=2):
    super(upsample_block, self).__init__()
    self.name = "upsample_block"
    self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel*up_scale_factor**2, kernel_size=kernels, stride=strides, padding=kernels/2)
    self.shuffler = nn.PixelShuffle(up_scale_factor)

  def forward(self, x):
    y = self.shuffler(self.conv1(x))
    y = swish_actf(y)
    return y



# Model (Discriminator) --> binary classification (real vs. created)
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.name = "Discriminator"
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
    self.batch_norm1 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
    self.batch_norm2 = nn.BatchNorm2d(128)
    self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
    self.batch_norm3 = nn.BatchNorm2d(128)
    self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
    self.batch_norm4 = nn.BatchNorm2d(256)
    self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
    self.batch_norm5 = nn.BatchNorm2d(256)
    self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
    self.batch_norm6 = nn.BatchNorm2d(512)
    self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
    self.batch_norm7 = nn.BatchNorm2d(512)
    self.pool1 = nn.AdaptiveAvgPool2d(1).cuda()  # added cuda
    self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1) ### maybe not need this many layers --> if so, change the in_channels of the next line and remove this line
    self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1)
  
  def forward(self, x):
    batch_size = x.size(0)
    x = swish_actf(self.conv1(x))
    x = swish_actf(self.batch_norm1(self.conv2(x)))
    x = swish_actf(self.batch_norm2(self.conv3(x)))
    x = swish_actf(self.batch_norm3(self.conv4(x)))
    x = swish_actf(self.batch_norm4(self.conv5(x)))
    x = swish_actf(self.batch_norm5(self.conv6(x)))
    x = swish_actf(self.batch_norm6(self.conv7(x)))
    x = swish_actf(self.batch_norm7(self.conv8(x)))
    x = self.pool1(x)
    x = self.conv10(swish_actf(self.conv9(x)))
    x = x.view(batch_size)
    return F.sigmoid(x)