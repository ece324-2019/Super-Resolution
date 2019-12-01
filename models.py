import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from skimage.transform import resize
from skimage.measure import compare_psnr
from tqdm import tqdm
from torch.autograd import Variable

import torchvision.utils as utils # Used in Training loop

torch.manual_seed(1000)


class ImageDataset(data.Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        label = self.labels[index]
        return features, label


# Activation function used for both networks (from paper: https://arxiv.org/pdf/1710.05941.pdf)
def swish_actf(x):
    # m = nn.LeakyReLU(0.1)             # leaky relu
    # return m(x)
    return x * torch.sigmoid(x)         # swish


# Model (Generator)
# Can experiment with number of residual blocks, channel numbers, kernel size, and stride number.
# 46 conv layers 
# 37 BN layers
# No linear layers

class Generator1(nn.Module):
    def __init__(self, num_blocks=16, upsample_factor=2, out_channel_num=64, kernel_size=3, stride_num=1):
        super(Generator1, self).__init__()
        self.name = "Generator"
        self.num_blocks = num_blocks
        self.upsample_factor = upsample_factor
        self.out_channel_num = out_channel_num
        self.kernel_size = kernel_size
        self.stride_num = stride_num

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.out_channel_num, kernel_size=self.kernel_size,
                               stride=self.stride_num,
                               padding=int(self.kernel_size / 2))  # for upscaling the image before feeding it into residual blocks

        for i in range(self.num_blocks):
            self.add_module('residual_block' + str(i + 1), residual_block(self.out_channel_num, self.kernel_size, self.stride_num))

        self.conv2 = nn.Conv2d(in_channels=out_channel_num, out_channels=self.out_channel_num,
                               kernel_size=self.kernel_size, stride=self.stride_num, padding=int(self.kernel_size / 2))
        self.bn2 = nn.BatchNorm2d(out_channel_num)

        out_chan = self.out_channel_num*(self.upsample_factor**2)
        self.upsample_block1 = upsample_block(in_channel=self.out_channel_num, out_channel=out_chan, kernels=3, strides=1,
                                up_scale_factor=upsample_factor)

        self.conv3 = nn.Conv2d(in_channels=self.out_channel_num, out_channels=3, kernel_size=9, stride=self.stride_num,
                               padding=4)

    def forward(self, x):
        x = swish_actf(self.conv1(x))
        x1 = x.clone()
        for i in range(self.num_blocks):
            x = self.__getattr__('residual_block' + str(i + 1))(x)
        x = x1 + self.bn2(self.conv2(x))
        x = self.upsample_block1(x)
        return self.conv3(x)


# Notes
# Batch normalization is used to decrease variance when training deeper network
# Residual blocks are the recursive CNN blocks that helpful when training deep network
# Upscaling before feeding into the residual CNN blocks is helpful in making the model learn the filters first (from paper: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network: https://arxiv.org/pdf/1609.04802.pdf)


# The following will be called multiple times in the Generator
class residual_block(nn.Module):
    def __init__(self, num_channel, kernels, strides):
        super(residual_block, self).__init__()
        self.name = "residual_block"
        self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=kernels, stride=strides,
                               padding=int(kernels / 2))
        self.batch_norm1 = nn.BatchNorm2d(num_channel)
        self.conv2 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=kernels, stride=strides,
                               padding=int(kernels / 2))
        self.batch_norm2 = nn.BatchNorm2d(num_channel)

    def forward(self, x):
        y = swish_actf(self.batch_norm1(self.conv1(x)))
        return x + self.batch_norm2(self.conv2(y))


# Upsampling the image at the end of the Generator
class upsample_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernels, strides, up_scale_factor):
        super(upsample_block, self).__init__()
        self.name = "upsample_block"
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=kernels, stride=strides, padding=int(kernels / 2))
        self.shuffler = nn.PixelShuffle(up_scale_factor)

    def forward(self, x):
        y = self.shuffler(self.conv1(x))
        y = swish_actf(y)
        return y


# 8 BN Layers
# 9 conv layers
class Discriminator1(nn.Module):
    def __init__(self, batch_size):
        super(Discriminator1, self).__init__()
        self.name = "Discriminator"
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.batch_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.batch_size, out_channels=self.batch_size, kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(self.batch_size)
        self.conv3 = nn.Conv2d(in_channels=self.batch_size, out_channels=self.batch_size*2, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(self.batch_size*2)
        self.conv4 = nn.Conv2d(in_channels=self.batch_size*2, out_channels=self.batch_size*2, kernel_size=3, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(self.batch_size*2)
        self.conv5 = nn.Conv2d(in_channels=self.batch_size*2, out_channels=self.batch_size*4, kernel_size=3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(self.batch_size*4)
        self.conv6 = nn.Conv2d(in_channels=self.batch_size*4, out_channels=self.batch_size*4, kernel_size=3, stride=2, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(self.batch_size*4)
        self.conv7 = nn.Conv2d(in_channels=self.batch_size*4, out_channels=self.batch_size*8, kernel_size=3, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(self.batch_size*8)
        self.conv8 = nn.Conv2d(in_channels=self.batch_size*8, out_channels=self.batch_size*8, kernel_size=3, stride=2, padding=1)
        self.batch_norm7 = nn.BatchNorm2d(self.batch_size*8)
        self.conv9 = nn.Conv2d(in_channels=self.batch_size*8, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = swish_actf(x)
        x = self.conv2(x)
        x = swish_actf(self.batch_norm1(x))
        x = swish_actf(self.batch_norm2(self.conv3(x)))
        x = swish_actf(self.batch_norm3(self.conv4(x)))
        x = swish_actf(self.batch_norm4(self.conv5(x)))
        x = swish_actf(self.batch_norm5(self.conv6(x)))
        x = swish_actf(self.batch_norm6(self.conv7(x)))
        x = swish_actf(self.batch_norm7(self.conv8(x)))
        x = self.conv9(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.sigmoid(x)
        x = x.view(x.size()[0], -1)
        return x


def load_model(gen_lr, dis_lr, resid_block_num, num_channel, kernel_size, sample_fac, batch_s):
    gen1 = Generator1(num_blocks=resid_block_num, upsample_factor=sample_fac, out_channel_num=num_channel,
                      kernel_size=kernel_size, stride_num=1)
    dis1 = Discriminator1(batch_s)
    dis_loss1 = nn.BCELoss()
    gen_optim1 = optim.Adam(gen1.parameters(), lr=gen_lr, betas=(0.9, 0.999))  # betas can be changed
    dis_optim1 = optim.Adam(dis1.parameters(), lr=dis_lr, betas=(0.9, 0.999))  # betas can be changed
    return gen1, dis1, dis_loss1, gen_optim1, dis_optim1


def load_data(HR_train, HR_val, HR_test, LR_train, LR_val, LR_test, batch_s):
    train_file = ImageDataset(LR_train, HR_train)
    val_file = ImageDataset(LR_val, HR_val)
    test_file = ImageDataset(LR_test, HR_test)
    train_loader = torch.utils.data.DataLoader(train_file, batch_size=batch_s, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_file, batch_size=batch_s, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_file, batch_size=batch_s, shuffle=True)
    return train_loader, val_loader, test_loader

