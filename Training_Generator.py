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
    return x * torch.sigmoid(x)


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
                               padding=int(self.kernel_size / 2))  # for upscaling the image before feeding it into residual blocks

        for i in range(self.num_blocks):
            self.add_module('residual_block' + str(i + 1), residual_block(self.out_channel_num, self.kernel_size, self.stride_num))

        self.conv2 = nn.Conv2d(in_channels=out_channel_num, out_channels=self.out_channel_num,
                               kernel_size=self.kernel_size, stride=self.stride_num, padding=int(self.kernel_size / 2))
        self.bn2 = nn.BatchNorm2d(out_channel_num)

        for j in range(int(self.upsample_factor / 4)):
            out_chan = self.out_channel_num*(self.upsample_factor**2)
            self.add_module('upsample_block' + str(j + 1),
                            upsample_block(in_channel=self.out_channel_num, out_channel=out_chan, kernels=3, strides=1,
                                           up_scale_factor=upsample_factor))

        self.conv3 = nn.Conv2d(in_channels=self.out_channel_num, out_channels=3, kernel_size=9, stride=self.stride_num,
                               padding=4)

    def forward(self, x):
        x = swish_actf(self.conv1(x))
        x1 = x.clone()
        for i in range(self.num_blocks):
            x = self.__getattr__('residual_block' + str(i + 1))(x)
        #print('x', x.shape)
        x = x1 + self.bn2(self.conv2(x))
        #print('x', x.shape)
        for i in range(int(self.upsample_factor / 4)):
            x = self.__getattr__('upsample_block' + str(i + 1))(x)
        #print('x', x.shape)
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
        #self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * up_scale_factor ** 2,
        #                       kernel_size=kernels, stride=strides, padding=int(kernels / 2))
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=kernels, stride=strides, padding=int(kernels / 2))
        self.shuffler = nn.PixelShuffle(up_scale_factor)

    def forward(self, x):
        y = self.shuffler(self.conv1(x))
        y = swish_actf(y)
        return y


def load_model(gen_lr, resid_block_num, num_channel, kernel_size, sample_fac, batch_s):
    gen1 = Generator1(num_blocks=resid_block_num, upsample_factor=sample_fac, out_channel_num=num_channel,
                      kernel_size=kernel_size, stride_num=1)
    #dis1 = Discriminator1(batch_s)
    gen_loss1 = nn.BCELoss()
    #dis_loss1 = nn.BCELoss()
    gen_optim1 = optim.Adam(gen1.parameters(), lr=gen_lr, betas=(0.9, 0.999))  # betas can be changed
    #dis_optim1 = optim.Adam(dis1.parameters(), lr=dis_lr, betas=(0.9, 0.999))  # betas can be changed
    #return gen1, dis1, gen_loss1, dis_loss1, gen_optim1, dis_optim1
    return gen1, gen_loss1, gen_optim1


def load_data(HR_train, HR_val, HR_test, LR_train, LR_val, LR_test, batch_s):
    train_file = ImageDataset(LR_train, HR_train)
    val_file = ImageDataset(LR_val, HR_val)
    test_file = ImageDataset(LR_test, HR_test)
    train_loader = torch.utils.data.DataLoader(train_file, batch_size=batch_s, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_file, batch_size=batch_s, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_file, batch_size=batch_s, shuffle=True)
    return train_loader, val_loader, test_loader


def train_gen(batch_size, gen_lr, epochs, resid_block_num, num_channel, kernel_size, gen_weights,
              cuda1, train_loader, val_loader, test_loader):
    torch.manual_seed(1000)

    sample_fac = 4

    G, G_loss_func, G_optim = load_model(gen_lr, resid_block_num, num_channel, kernel_size, sample_fac, batch_size)

    # If we want to continue training from where we left off:
    if gen_weights != '':
        G.load_state_dict(torch.load(gen_weights))

    content_loss_func = nn.MSELoss()

    if cuda1:
        G.cuda()
        content_loss_func.cuda()

    #corr_fake_D = 0
    #correct_D = 0
    psnr_G = []  # Peak signal to noise ratio
    #psnr_I = []
    # ssim = []  # Structual similarity index
    # train_loss_D = []
    train_loss_G = []
    valid_loss_G = []
    num_samples_trained = 0

    for epoch in range(epochs):
        G.train()
        for i, batch in enumerate(train_loader):
            data1, label = batch
            num_samples_trained += batch_size

            G.zero_grad()

            real_img = Variable(label)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            real_img = torch.transpose(real_img, 1, 3)

            noise1 = Variable(data1)
            noise1 = noise1.unsqueeze(1)
            if torch.cuda.is_available():
                noise1 = noise1.cuda()
            noise1 = torch.transpose(noise1, 1, 4).squeeze()
            ''''''''''''''''''''''''
            fake_input = G(noise1.float())
            fake_input = torch.transpose(fake_input, 2, 3)

            '''Analyze the output'''
            G_content_loss = content_loss_func(fake_input, real_img.float())
            G_content_loss.backward()
            G_optim.step()

            valid_loss_G, train_loss_G, psnr_G, psnr_I = evaluate_valid(val_loader, G_content_loss.item(), fake_input,
                                                                        content_loss_func, train_loss_G, G)
            # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' %(epoch.item(), 25, i, len(train_loader), error_D.data[0].item(), actual_G_loss.data[0].item())) ############### --> idk

        print("Epoch" + str(epoch) + "Ended")
        torch.save(D.state_dict(), "Model Checkpoints/{}.pt".format(D.name))
        torch.save(G.state_dict(), "Model Checkpoints/{}.pt".format(G.name))
        print("Fake image accuracy(Discriminator)", corr_fake_D / len(train_loader.dataset))
        print("Real Image Accuracy (Discriminator)", correct_D / len(train_loader.dataset))
        print("Combined Accuracy (Discriminator)", ((corr_fake_D + correct_D) / (2 * len(train_loader.dataset))))
        corr_fake_D, correct_D = 0, 0

        '''save real image here'''
        temp = torch.transpose(real_img[0], 0, 2).numpy()
        temp = Image.fromarray(temp)
        temp.save('Model Checkpoints/real_images_Epoch_%03d.png' % epoch)

        '''save fake image here'''
        fake11 = G(noise1.float())
        temp = torch.transpose(fake11[0].detach(), 0, 2)
        temp = torch.transpose(temp.detach(), 1, 0)
        max_num = max(temp.flatten())
        min_num = min(temp.flatten())
        print('before sig', max_num, min_num)
        temp = (temp - min_num) / (max_num - min_num)
        # temp = torch.sigmoid(temp).numpy()
        # print('after sig', max(temp.flatten()), min(temp.flatten()))
        # print(type(temp), temp.shape, type(temp[0][0][0]), temp[0][0][0])
        # print(temp.shape)
        # temp = Image.fromarray(temp)
        # temp.save('Model Checkpoints/fake_images_Epoch_%03d.png' % epoch)
        plt.imshow(temp)
        # plt.show()
        plt.savefig('Model Checkpoints/fake_images_Epoch_%03d.png' % epoch)

        '''
        fake11 = G(noise1.float())
        img = fake11[0]
        #print(fake11[0].shape, type(fake11[0]))
        #temp = torch.transpose(fake11[0],1,2)
        #temp = torch.transpose(temp,0,2)
        utils.save_image(img, 'Model Checkpoints/fake_images_Epoch_%03d.png' % epoch, normalize=True)


        #temp = torch.transpose(real_img[0], 0, 2).numpy()
        #temp = Image.fromarray(temp.numpy())
        #temp.save('Model Checkpoints/real_images_Epoch_%03d.png' % epoch)
        #plt.imshow(temp)
        #plt.savefig('Model Checkpoints/real_images_Epoch_%03d.png' % epoch)

        fake11 = G(noise1.float())
        temp = torch.transpose(fake11[0].detach(), 0, 2)
        #print(type(temp), temp.shape)
        # utils.save_image(temp, 'Model Checkpoints/fake_images_Epoch_%03d.png' % epoch, normalize=True)

        temp = temp.numpy()
        #print(type(temp), temp.shape)
        # temp = Image.fromarray(temp)
        temp = np.clip(temp, 0.0, 1.0)
        plt.imshow(temp)
        plt.show()
        plt.savefig('Model Checkpoints/fake_images_Epoch_%03d.png' % epoch)
        # temp.save('Model Checkpoints/fake_images_Epoch_%03d.png' % epoch)
        '''

    train_loss_G = np.array(train_loss_G)
    valid_loss_G = np.array(valid_loss_G)
    psnr_I.append(np.array(psnr_I))
    psnr_G.append(np.array(psnr_G))

    np.save("Training_loss_G", train_loss_G)
    np.save("Validation_loss_g", valid_loss_G)
    np.save("psnr_I", psnr_I)
    np.save("psnr_G", psnr_G)

    torch.save(G.state_dict(), "{}.pt".format(G.name))

    return True


def evaluate_valid(valid_loader, actual_G_loss1, fake_input1, content_loss_func, training_loss_G, G):
  psnr_G = []
  psnr_I = []
  valid_loss_G = []
  for i, batch in enumerate(valid_loader):
    if i > 0 :
      break

    low_img, real_img = batch

    low_img = torch.transpose(low_img, 1, 3)
    real_img = torch.transpose(real_img, 1, 3)

    noise1 = Variable(low_img)
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

    print('psnr: ' ,psnr_G)

    training_loss_G.append(actual_G_loss1)

    #Validation Loss Stuff
    real_img1 = real_img
    fake_inputA = fake_input1
    fake_input11 = noise_output.squeeze(1).float()
    if torch.cuda.is_available():
        real_img1 = real_img1.cuda()
        fake_input11 = fake_input11.cuda()

    G_content_l = content_loss_func(fake_input11, real_img1.float())

    total_loss_G = G_content_l  # prob wanna change this code
    valid_loss_G.append(total_loss_G.item())
    return valid_loss_G, training_loss_G, psnr_G, psnr_I


def evaluate(low_img, noise_output, real_img):
    batch_size = real_img.shape[0]
    interpolated_img = np.zeros(real_img.shape)
    psnr_I_np = np.zeros(batch_size)
    psnr_G_np = np.zeros(batch_size)

    for i in range(batch_size):
        interpolated_img[i] = np.repeat(np.repeat(low_img[i], 4, axis=1), 4, axis=2)
        psnr_I_np[i] = compare_psnr(real_img[i], interpolated_img[i])
        psnr_G_np[i] = compare_psnr(real_img[i], noise_output[i])
    total_psnr_I = np.sum(psnr_I_np)
    total_psnr_G = np.sum(psnr_G_np)
    return total_psnr_G, total_psnr_I

