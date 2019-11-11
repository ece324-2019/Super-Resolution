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

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.out_channel_num, kernel_size=self.kernel_size,
                               stride=self.stride_num,
                               padding=self.kernel_size / 2)  # for upscaling the image before feeding it into residual blocks

        for i in range(self.num_blocks):
            self.add_module('residual_block' + str(i + 1), residual_block())

        self.conv2 = nn.Conv2d(in_channels=out_channel_num, out_channels=self.out_channel_num,
                               kernel_size=self.kernel_size, stride=self.stride_num, padding=self.kernel_size / 2)
        self.bn2 = nn.BatchNorm2d(out_channel_num)

        for j in range(int(self.upsample_factor / 2)):
            self.add_module('upsample_block' + str(j + 1),
                            upsample_block(in_channel=self.out_channel_num, kernels=3, strides=1,
                                           up_scale_factor=upsample_factor))

        self.conv3 = nn.Conv2d(in_channels=self.out_channel_num, out_channels=3, kernel_size=9, stride=self.stride_num,
                               padding=4)  ######################################### kernel size and padding???

    def forward(self, x):
        x = swish_actf(self.conv1(x))
        x1 = x.copy()
        for i in range(self.n_residual_blocks):
            x = self.__getattr__('residual_block' + str(i + 1))(x)

        x = x1 + self.bn2(self.conv2(x))

        for i in range(int(self.upsample_factor / 2)):
            x = self.__getattr__('upsample_block' + str(i + 1))(x)

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
        self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=kernels, stride=strides,
                               padding=kernels / 2)
        self.batch_norm1 = nn.BatchNorm2d(num_channel)
        self.conv2 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=kernels, stride=strides,
                               padding=kernels / 2)
        self.batch_norm2 = nn.BatchNorm2d(num_channel)

    def forward(self, x):
        y = swish_actf(self.batch_norm1(self.conv1(x)))
        return x + self.batch_norm2(self.conv2(y))


# Upsampling the image at the end of the Generator
class upsample_block(nn.Module):
    def __init__(self, in_channel=64, kernels=3, strides=1, up_scale_factor=2):
        super(upsample_block, self).__init__()
        self.name = "upsample_block"
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * up_scale_factor ** 2,
                               kernel_size=kernels, stride=strides, padding=kernels / 2)
        self.shuffler = nn.PixelShuffle(up_scale_factor)

    def forward(self, x):
        y = self.shuffler(self.conv1(x))
        y = swish_actf(y)
        return y


class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
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
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024,
                               kernel_size=1)  ######## maybe not need this many layers --> if so, change the in_channels of the next line and remove this line
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


# GAN Training code

def load_model(gen_lr, dis_lr, resid_block_num, num_channel, kernel_size, sample_fac):
    gen1 = Generator1(num_blocks=resid_block_num, upsample_factor=sample_fac, out_channel_num=num_channel,
                      kernel_size=kernel_size, stride_num=1)
    dis1 = Discriminator1()
    gen_loss1 = nn.BCELoss()
    dis_loss1 = nn.BCELoss()
    gen_optim1 = optim.Adam(gen1.parameters(), lr=gen_lr, betas=(0.5, 0.999))  # betas can be changed
    dis_optim1 = optim.Adam(dis1.parameters(), lr=dis_lr, betas=(0.5, 0.999))  # betas can be changed
    return gen1, dis1, gen_loss1, dis_loss1, gen_optim1, dis_optim1


def load_data(HR_train, HR_val, HR_test, LR_train, LR_val, LR_test, batch_s):
    train_file = ImageDataset(LR_train, HR_train)
    val_file = ImageDataset(LR_val, HR_val)
    test_file = ImageDataset(LR_test, HR_test)
    train_loader = torch.utils.data.DataLoader(train_file, batch_size=batch_s, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_file, batch_size=batch_s, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_file, batch_size=batch_s, shuffle=True)
    return train_loader, val_loader, test_loader


def training_GAN(batch_size, gen_lr, dis_lr, epochs, resid_block_num, num_channel, kernel_size, gen_weights,
                 dis_weights, cuda1, train_loader, val_loader, test_loader):
    torch.manual_seed(1000)

    sample_fac = 2

    G, D, G_loss_func, D_loss_func, G_optim, D_optim = load_model(gen_lr, dis_lr, resid_block_num, num_channel,
                                                                  kernel_size, sample_fac)

    # If we want to continue training from where we left off:
    if gen_weights != '':
        G.load_state_dict(torch.load(gen_weights))
    if dis_weights != '':
        D.load_state_dict(torch.load(dis_weights))

    content_loss_func = nn.MSELoss()
    adv_loss_func = nn.BCELoss()

    if cuda1:
        G.cuda()
        D.cuda()
        content_loss_func.cuda()
        adv_loss_func.cuda()

    corr_fake_D = 0
    correct_D = 0
    psnr_gen = []  # Peak signal to noise ratio
    psnrint = []
    ssim = []  # Structual similarity index
    train_loss_D = []
    train_loss_G = []
    num_samples_trained = 0

    # num_save = 0

    for epoch in range(epochs):
        # train_bar = tqdm(train_loader)
        G.train()
        D.train()
        for i, batch in enumerate(train_loader):
            data1, label = batch
            num_samples_trained += batch_size

            # Training the Discriminator
            D.zero_grad()
            real_img = Variable(label).unsqueeze(1)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            output_fake_img = D(real_img.float())
            output_fake_img = output_fake_img.view(-1)
            correct_D += int(sum(output_fake_img > 0.5))

            ones1 = Variable(torch.ones(real_img.size()[0]))
            if torch.cuda.is_available():
                ones1 = ones1.cuda()
            D_loss_real_img = D_loss_func(output_fake_img, ones1)

            noise1 = Variable(data1)
            noise1 = noise1.unsqueeze(1)
            if torch.cuda.is_available():
                noise1 = noise1.cuda()
            fake_input = G(noise1.float())

            zeros1 = Variable(torch.zeros(real_img.size()[0]))
            if torch.cuda.is_available():
                zeros1 = zeros1.cuda()

            output_D = D(fake_input.detach()).view(-1)
            corr_fake_D += int(sum(output_D < 0.5))
            D_loss_fake_img = D_loss_func(output_D, zero_const)
            error_D = D_loss_real_img + D_loss_fake_img
            error_D.backward()
            D_optim.step()

            # Training the Generator
            G.zero_grad()
            G_content_loss = content_loss_func(fake_input, real_img.float())

            output_D = D(fake_input.detach()).view(-1)

            ones1 = Variable(torch.ones(real_img.size()[0]))
            if torch.cuda.is_available():
                ones1 = ones1.cuda()

            G_adv_loss = adv_loss_func(output_D, ones1.float())
            actual_G_loss = G_content_loss + 1e-3 * G_adv_loss  # We should probably change this equation

            actual_G_loss.backward()
            G_optim.step()

            #valid_loss_G, training_loss_G = evaluate_valid(val_loader, actual_G_loss.item(), fake_input, content_loss_func, adv_loss_func, training_loss_G, G, D)
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' %(epoch, 25, i, len(train_loader), error_D.data[0], actual_G_loss.data[0])) ############### --> idk

        print("Epoch Ended")
        torch.save(D.state_dict(), "Model Checkpoints/{}.pt".format(D.name))
        torch.save(G.state_dict(), "Model Checkpoints/{}.pt".format(G.name))
        print("Fake image accuracy(Discriminator)", corr_fake_D / len(train_loader.dataset))
        print("Real Image Accuracy (Discriminator)", correct_D / len(train_loader.dataset))
        print("Combined Accuracy (Discriminator)", ((corr_fake_D + correct_D) / (2 * len(train_loader.dataset))))
        corr_fake_D, correct_D = 0, 0
        utils.save_image(real_img[0], '/content/gdrive/My Drive/ECE324 Project/Model Checkpoints/real_images_Epoch_%03d.png' % (epoch), normalize=True)
        fake11 = G(noise1.float())
        utils.save_image(fake11[0], '/content/gdrive/My Drive/ECE324 Project/Model Checkpoints/fake_images_Epoch_%03d.png' % (epoch), normalize=True)

    training_loss_G = np.array(training_loss_G)
    valid_loss_G = np.array(valid_loss_G)
    psnrint = np.array(psnrint)
    psnr_gen = np.array(psnr_gen)

    np.save("Training_loss_G", training_loss_G)
    np.save("Validation_loss_g", valid_loss_G)
    np.save("psnr_I", psnr_I)
    np.save("psnr_G", psnr_G)

    torch.save(D.state_dict(), "{}.pt".format(D.name))
    torch.save(G.state_dict(), "{}.pt".format(G.name))

    return True


# load iterator
HR_train = np.load('HR_train.npy')
HR_valid = np.load('HR_valid.npy')
HR_test = np.load('HR_test.npy')
LR_train = np.load('LR_train.npy', allow_pickle=True)
LR_valid = np.load('LR_valid.npy', allow_pickle=True)
LR_test = np.load('LR_test.npy', allow_pickle=True)
batch_size1 = 16
train_loader, val_loader, test_loader = load_data(HR_train, HR_valid, HR_test, LR_train, LR_valid, LR_test, batch_size1)


# Evaluation on validation dataset
def evaluate_valid(valid_loader, actual_G_loss1, fake_input1, content_loss_func, adv_loss_func, training_loss_G, G, D):
  psnr_G = []
  psnr_I = []
  valid_loss_G = []
  for i, batch in enumerate(valid_loader):
    if i > 0 :
      break
    low_img, real_img = batch

    noise1 = Variable(low_img).unsqueeze(1)
    if torch.cuda.is_available():
      noise1 = noise1.cuda()
    noise_output = G(noise1.float())

    noise_output1 = noise_output.squeeze(1)
    noise_output1 = np.array(noise_output1.detach())
    low_img1 = np.array(low_img.detach())
    real_img1 = np.array(real_img.detach())

    total_psnr_G, total_psnr_I = evaluate(low_img1, noise_output1, real_img1)
    psnr_G.append(total_psnr_G)
    psnr_I.append(total_psnr_I)

    training_loss_G.append(actual_G_loss1)

    #Validation Loss Stuff
    real_img1 = real_img
    fake_inputA = fake_input1
    fake_input11 = noise_output.squeeze(1).float()
    if torch.cuda.is_available():
        real_img1 = real_img1.cuda()
        fake_input11 = fake_input11.cuda()

    G_content_l = content_loss_func(fake_input11, real_img1.float())

    output_val = D(fake_inputA.detach()).view(-1)

    ones11 = Variable(torch.ones(real_img1.size()[0]))
    if torch.cuda.is_available():
        ones11 = ones11.cuda()

    G_adv_l = adv_loss_func(output_val, ones11.float())
    total_loss_G = G_content_l + 1e-3 * G_adv_l  # prob wanna change this code
    valid_loss_G.append(total_loss_G.item())
    return valid_loss_G, training_loss_G


def evaluate(low_img, noise_output, real_img):
    batch_size = real_img.shape[0]
    interpolated_img = np.zeros(real_img.shape)
    psnr_I_np = np.zeros(batch_size)
    psnr_G_np = np.zeros(batch_size)

    for i in range(batch_size):
        interpolated_img[i] = resize(low_img[i], (112, 92))
        psnr_I_np[i] = compare_psnr(real_img[i], interpolated_img[i])
        psnr_G_np[i] = compare_psnr(real_img[i], noise_output[i])
    total_psnr_I = np.sum(psnr_I_np)
    total_psnr_G = np.sum(psnr_G_np)

    return total_psnr_G, total_psnr_I


training_GAN(batch_size=16, gen_lr=0.1, dis_lr=0.1, epochs=1, resid_block_num=16, num_channel=64, kernel_size=3,
             gen_weights="", dis_weights="", cuda1=False, train_loader=train_loader, val_loader=val_loader,
             test_loader=test_loader)
