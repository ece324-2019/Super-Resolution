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
from models import *

torch.manual_seed(1000)

# GAN Training code

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
    psnr_G = []  # Peak signal to noise ratio
    psnr_I = []
    #ssim = []  # Structual similarity index
    #train_loss_D = []
    train_loss_G = []
    valid_loss_G = []
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
            D_loss_fake_img = D_loss_func(output_D, zeros1)
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

            valid_loss_G, train_loss_G, psnr_G, psnr_I = evaluate_valid(val_loader, actual_G_loss.item(), fake_input, content_loss_func, adv_loss_func, training_loss_G, G, D)
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

    train_loss_G = np.array(train_loss_G)
    valid_loss_G = np.array(valid_loss_G)
    psnr_I = np.array(psnr_I)
    psnr_G = np.array(psnr_G)

    np.save("Training_loss_G", train_loss_G)
    np.save("Validation_loss_g", valid_loss_G)
    np.save("psnr_I", psnr_I)
    np.save("psnr_G", psnr_G)

    torch.save(D.state_dict(), "{}.pt".format(D.name))
    torch.save(G.state_dict(), "{}.pt".format(G.name))

    return True



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
    return valid_loss_G, training_loss_G, psnr_G, psnr_I


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



if __name__ == "__main__":
    # load iterator
    HR_train = np.load('HR_train.npy')
    HR_valid = np.load('HR_valid.npy')
    HR_test = np.load('HR_test.npy')
    LR_train = np.load('LR_train.npy', allow_pickle=True)
    LR_valid = np.load('LR_valid.npy', allow_pickle=True)
    LR_test = np.load('LR_test.npy', allow_pickle=True)
    batch_size1 = 16
    train_loader, val_loader, test_loader = load_data(HR_train, HR_valid, HR_test, LR_train, LR_valid, LR_test, batch_size1)

    training_GAN(batch_size=16, gen_lr=0.1, dis_lr=0.1, epochs=1, resid_block_num=16, num_channel=64, kernel_size=3,
                 gen_weights="", dis_weights="", cuda1=False, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
