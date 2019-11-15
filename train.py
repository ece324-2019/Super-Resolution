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

torch.manual_seed(1000)

# GAN Training code

def training_GAN(batch_size, gen_lr, dis_lr, epochs, resid_block_num, num_channel, kernel_size, gen_weights,
                 dis_weights, cuda1, train_loader, val_loader, test_loader):
    torch.manual_seed(1000)

    sample_fac = 4

    G, D, G_loss_func, D_loss_func, G_optim, D_optim = load_model(gen_lr, dis_lr, resid_block_num, num_channel,
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
            real_img = Variable(label)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            real_img = torch.transpose(real_img, 1, 3)
            output_fake_img = D(real_img.float())
            output_fake_img = output_fake_img.view(-1)
            correct_D += int(sum(output_fake_img > 0.5))
            ones1 = Variable(torch.ones(output_fake_img.size()[0]))
            if torch.cuda.is_available():
                ones1 = ones1.cuda()
            D_loss_real_img = D_loss_func(output_fake_img, ones1)
            print(i)
            noise1 = Variable(data1)
            noise1 = noise1.unsqueeze(1)
            if torch.cuda.is_available():
                noise1 = noise1.cuda()
            #print(noise1.shape)
            noise1 = torch.transpose(noise1, 1, 4).squeeze()
            #noise1 = torch.transpose(noise1, 2, 3)
            #print(noise1.shape)
            ''''''''''''''''''''''''
            fake_input = G(noise1.float())
            #print('world')
            fake_input = torch.transpose(fake_input,2,3)

            #print('generated: ', fake_input.shape)

            output_D = D(fake_input)

            #print(fake_input.shape)

            output_D = output_D.view(-1)
            #print('!!!')

            corr_fake_D += int(sum(output_D < 0.5))

            #print(fake_input.shape)

            zeros1 = Variable(torch.zeros(real_img.size()[0]))
            if torch.cuda.is_available():
                zeros1 = zeros1.cuda()

            D_loss_fake_img = D_loss_func(output_D, zeros1)
            #print(fake_input.shape)
            error_D = D_loss_real_img + D_loss_fake_img
            error_D.backward(retain_graph=True)
            D_optim.step()
            #print('yeah')
            # Training the Generator
            G.zero_grad()
            #print(fake_input.shape, real_img.shape)
            #print(noise1.shape)

            output_D = D(fake_input.detach()).view(-1)

            ones1 = Variable(torch.ones(real_img.size()[0]))
            if torch.cuda.is_available():
                ones1 = ones1.cuda()
            #print('almost')

            G_content_loss = content_loss_func(fake_input, real_img.float())
            G_adv_loss = adv_loss_func(output_D, ones1.float())

            '''###############################################################################################'''

            print('loss functions: ', G_content_loss, G_adv_loss)
            actual_G_loss = G_content_loss + G_adv_loss  # We should probably change this equation

            '''###############################################################################################'''

            #print('almost there')
            #print(G_content_loss, G_adv_loss, G_content_loss.shape, G_adv_loss.shape)
            actual_G_loss.backward()
            G_optim.step()
            #print("here")

            valid_loss_G, train_loss_G, psnr_G, psnr_I = evaluate_valid(val_loader, actual_G_loss.item(), fake_input, content_loss_func, adv_loss_func, train_loss_G, G, D)
            #print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' %(epoch.item(), 25, i, len(train_loader), error_D.data[0].item(), actual_G_loss.data[0].item())) ############### --> idk

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
        temp = (temp - min_num)/(max_num-min_num)
        #temp = torch.sigmoid(temp).numpy()
        #print('after sig', max(temp.flatten()), min(temp.flatten()))
        #print(type(temp), temp.shape, type(temp[0][0][0]), temp[0][0][0])
        #print(temp.shape)
        #temp = Image.fromarray(temp)
        #temp.save('Model Checkpoints/fake_images_Epoch_%03d.png' % epoch)
        plt.imshow(temp)
        #plt.show()
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

    #print('shape', low_img.shape)
    for i in range(batch_size):
        #print(low_img[i].shape, real_img[i].shape)
        #low_img = torch.transpose(torch.tensor(low_img[i]), 0, 2)
        #real_img = torch.transpose(torch.tensor(real_img[i]), 0, 2)
        #print(low_img[i].shape, real_img[i].shape)
        #interpolated_img[i] = resize(low_img[i], (556, 648))

        interpolated_img[i] = np.repeat(np.repeat(low_img[i], 4, axis=1), 4, axis=2)

        #print(real_img[i].shape, interpolated_img[i].shape)
        psnr_I_np[i] = compare_psnr(real_img[i], interpolated_img[i])
        psnr_G_np[i] = compare_psnr(real_img[i], noise_output[i])
    total_psnr_I = np.sum(psnr_I_np)
    total_psnr_G = np.sum(psnr_G_np)

    return total_psnr_G, total_psnr_I



if __name__ == "__main__":
    # load iterator
    #print("loading datasets")
    HR_train = np.load('HR_train.npy')
    HR_valid = np.load('HR_valid.npy')
    HR_test = np.load('HR_test.npy')
    LR_train = np.load('LR_train.npy', allow_pickle=True)
    LR_valid = np.load('LR_valid.npy', allow_pickle=True)
    LR_test = np.load('LR_test.npy', allow_pickle=True)
    #print("Done loading datasets")

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

    #print("Doing datasets")

    LR_train = np.array(LR_train)[0:int(len(LR_train)/3600)]
    HR_train = np.array(HR_train)[0:int(len(HR_train)/3600)]

    #LR_train = np.repeat(LR_train, 100, axis=0)
    #HR_train = np.repeat(HR_train, 100, axis=0)

    print('Resize done')

    batch_size1 = 2
    train_loader, val_loader, test_loader = load_data(HR_train, HR_valid, HR_test, LR_train, LR_valid, LR_test, batch_size1)

    training_GAN(batch_size=batch_size1, gen_lr=0.001, dis_lr=0.1, epochs=1000, resid_block_num=18, num_channel=20, kernel_size=3,
                 gen_weights='', dis_weights='', cuda1=False, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
