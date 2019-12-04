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
from PIL import Image
import matplotlib
import torchvision

import torchvision.utils as utils # Used in Training loop
from models import *

torch.manual_seed(1001)

# GAN Training code

def training_GAN(batch_size, gen_lr, dis_lr, epochs, resid_block_num, num_channel, kernel_size, gen_weights,
                 dis_weights, cuda1, train_loader, val_loader, test_loader):
    torch.manual_seed(1001)

    sample_fac = 4

    G, D, D_loss_func, G_optim, D_optim = load_model(gen_lr, dis_lr, resid_block_num, num_channel,
                                                                  kernel_size, sample_fac, batch_size)

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
    #train_loss_D = []
    train_loss_G = []
    valid_loss_G = []
    num_samples_trained = 0

    for epoch in range(epochs):
        G.train()
        D.train()
        for i, batch in enumerate(train_loader):
            data1, label = batch
            num_samples_trained += batch_size

            # Training the Discriminator
            D.zero_grad()
            real_img = Variable(label)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            real_img = torch.transpose(real_img, 1, 3)

            output_fake_img = D(real_img.float())
            output_fake_img = output_fake_img.view(-1)
            for element in output_fake_img:
                if element > 0.5:
                    correct_D += 1
            print('number of correct D ', correct_D, output_fake_img)

            ones1 = Variable(torch.ones(output_fake_img.size()[0]))
            if torch.cuda.is_available():
                ones1 = ones1.cuda()
            D_loss_real_img = D_loss_func(output_fake_img, ones1)
            print(i)
            noise1 = Variable(data1)
            noise1 = noise1.unsqueeze(1)
            if torch.cuda.is_available():
                noise1 = noise1.cuda()
            noise1 = torch.transpose(noise1, 1, 4).squeeze()

            fake_input = G(noise1.float())
            fake_input = torch.transpose(fake_input,2,3)

            output_D = D(fake_input)

            output_D = output_D.view(-1)
            for element in output_D:
                if element < 0.5:
                    corr_fake_D += 1
            print('number of corr fake D ', corr_fake_D, output_D)

            zeros1 = Variable(torch.zeros(real_img.size()[0]))
            if torch.cuda.is_available():
                zeros1 = zeros1.cuda()

            D_loss_fake_img = D_loss_func(output_D, zeros1)
            error_D = D_loss_real_img + D_loss_fake_img
            error_D.backward(retain_graph=True)
            D_optim.step()
            
            # Training the Generator
            G.zero_grad()

            output_D = D(fake_input.detach()).view(-1)

            ones1 = Variable(torch.ones(real_img.size()[0]))
            if torch.cuda.is_available():
                ones1 = ones1.cuda()

            G_content_loss = content_loss_func(fake_input, real_img.float())
            G_adv_loss = adv_loss_func(output_D, ones1.float())

            '''###############################################################################################'''

            print('G loss functions: ', G_content_loss, G_adv_loss)
            print('validation loss: ', valid_loss_G)
            actual_G_loss = G_content_loss + 0.01 * G_adv_loss

            '''###############################################################################################'''

            actual_G_loss.backward()
            G_optim.step()
            valid_loss_G, train_loss_G, psnr_G = evaluate_valid(val_loader, actual_G_loss.item(), fake_input, content_loss_func, adv_loss_func, train_loss_G, G, D)


            #print("Epoch" + str(epoch) + "Ended")
            torch.save(D.state_dict(), "Model Checkpoints/{}.pt".format(D.name))
            torch.save(G.state_dict(), "Model Checkpoints/{}.pt".format(G.name))
            print("Fake image accuracy(Discriminator)", corr_fake_D / batch_size)
            print("Real Image Accuracy (Discriminator)", correct_D / batch_size)
            print("Combined Accuracy (Discriminator)", ((corr_fake_D + correct_D) / (2 * batch_size)))
            corr_fake_D, correct_D = 0, 0

            '''save real image here'''
            temp = torch.transpose(real_img[0], 0, 2).numpy()
            temp = Image.fromarray(temp)
            filename = 'real_Epoch_' + str(epoch) + '_batch_' + str(i) + '.png'
            temp.save('Model Checkpoints/' + filename)

            '''save fake image here'''
            fake11 = G(noise1.float())
            temp = torch.transpose(fake11[0].detach(), 0, 2)
            temp = torch.transpose(temp.detach(), 1, 0)
            max_num = max(temp.flatten())
            min_num = min(temp.flatten())
            print('before sig', max_num, min_num)
            temp = (temp - min_num)/(max_num-min_num)

            # temp = torch.sigmoid(temp).numpy()
            # print('after sig', max(temp.flatten()), min(temp.flatten()))
            # print(type(temp), temp.shape, type(temp[0][0][0]), temp[0][0][0])
            # print(temp.shape)
            # temp = Image.fromarray(temp)
            # temp.save('Model Checkpoints/fake_images_Epoch_%03d.png' % epoch)

            plt.imshow(temp)
            #plt.show()
            filename = 'fake_Epoch_' + str(epoch) + '_batch_' + str(i) + '.png'
            plt.savefig('Model Checkpoints/' + filename)
            train_loss_G = np.array(train_loss_G)
            valid_loss_G = np.array(valid_loss_G)
            psnr_G.append(np.array(psnr_G))

            np.save("Training_loss_G", train_loss_G)
            np.save("Validation_loss_g", valid_loss_G)
            np.save("psnr_G", psnr_G)

            torch.save(D.state_dict(), "{}.pt".format(D.name))
            torch.save(G.state_dict(), "{}.pt".format(G.name))


# Evaluation on validation dataset
def evaluate_valid(valid_loader, actual_G_loss1, fake_input1, content_loss_func, adv_loss_func, training_loss_G, G, D):
  psnr_G = []
  valid_loss_G = []
  for i, batch in enumerate(valid_loader):
    if i > 0 :
      break

    low_img, real_img = batch
    #print(low_img.shape, real_img.shape)

    low_img = torch.transpose(low_img, 1, 3)
    real_img = torch.transpose(real_img, 1, 3)

    noise1 = Variable(low_img)
    #print(noise1.shape)
    #noise1 = torch.transpose(noise1, 1, 4).squeeze()
    #print(noise1.shape)
    #noise1 = torch.transpose(noise1, 1, 3)
    #print(noise1.shape)
    if torch.cuda.is_available():
      noise1 = noise1.cuda()
    noise_output = G(noise1.float())

    noise_output1 = noise_output.squeeze(1)
    noise_output1 = np.array(noise_output1.detach())
    real_img1 = np.array(real_img.detach())

    total_psnr_G= evaluate(noise_output1, real_img1)
    psnr_G.append(total_psnr_G)

    print('psnr: ' ,psnr_G)

    #training_loss_G.append(actual_G_loss1)
    np.append(training_loss_G, actual_G_loss1)

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
    total_loss_G = G_content_l + 1e-3 * G_adv_l
    valid_loss_G.append(total_loss_G.item())
    return valid_loss_G, training_loss_G, psnr_G


def evaluate(noise_output, real_img):
    batch_size = real_img.shape[0]
    psnr_G_np = np.zeros(batch_size)

    for i in range(batch_size):
        psnr_G_np[i] = compare_psnr(real_img[i], noise_output[i])
    total_psnr_G = np.sum(psnr_G_np)

    return total_psnr_G



if __name__ == "__main__":
    # load iterator
    LR_train = []
    for i in range(1, 1201):
        for j in range(6):
            name = str(i) + '_' + str(j) + '.png'
            im = Image.open('new/ds_train/ds_train_pic/' + name)
            LR_train.append(np.array(im))
    print(len(LR_train))
    HR_train = []
    for i in range(1, 1201):
        for j in range(6):
            name = str(i) + '.png'
            im = Image.open('new/hr_train/hr_train_pic/' + name)
            HR_train.append(np.array(im))
    print(len(HR_train))

    LR_valid = []
    for i in range(1201, 1551):
        for j in range(6):
            name = str(i) + '_' + str(j) + '.png'
            im = Image.open('new/ds_valid/ds_valid_pic/' + name)
            LR_valid.append(np.array(im))
    print(len(LR_valid))
    HR_valid = []
    for i in range(1201, 1551):
        for j in range(6):
            name = str(i) + '.png'
            im = Image.open('new/hr_valid/hr_valid_pic/' + name)
            HR_valid.append(np.array(im))
    print(len(HR_valid))

    LR_test = []
    for i in range(1551, 1601):
        for j in range(6):
            name = str(i) + '_' + str(j) + '.png'
            im = Image.open('new/ds_test/ds_test_pic/' + name)
            LR_test.append(np.array(im))
    print(len(LR_test))
    HR_test = []
    for i in range(1551, 1601):
        for j in range(6):
            name = str(i) + '.png'
            im = Image.open('new/hr_test/hightest/hr_test_pic/' + name)
            HR_test.append(np.array(im))
    print(len(HR_test))

    # resize data
    for i in range(len(LR_train)):
        LR_train[i] = np.array(LR_train[i])[0:162, 0:139, :]
    for i in range(len(LR_valid)):
        LR_valid[i] = np.array(LR_valid[i])[0:162, 0:139, :]
    for i in range(len(LR_test)):
        LR_test[i] = np.array(LR_test[i])[0:162, 0:139, :]
    for i in range(len(HR_train)):
        HR_train[i] = np.array(HR_train[i])[0:162 * 4, 0:139 * 4, :]
    for i in range(len(HR_valid)):
        HR_valid[i] = np.array(HR_valid[i])[0:162 * 4, 0:139 * 4, :]
    for i in range(len(HR_test)):
        HR_test[i] = np.array(HR_test[i])[0:162 * 4, 0:139 * 4, :]

    LR_train = np.array(LR_train)[0:int(len(LR_train)/6)]
    HR_train = np.array(HR_train)[0:int(len(HR_train)/6)]

    print('Resize done')

    batch_size1 = 16
    train_loader, val_loader, test_loader = load_data(HR_train, HR_valid, HR_test, LR_train, LR_valid, LR_test, batch_size1)

    training_GAN(batch_size=batch_size1, gen_lr=1e-3, dis_lr=0.01, epochs=1000, resid_block_num=18, num_channel=20, kernel_size=3,
                 gen_weights='Generator.pt', dis_weights='', cuda1=False, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
