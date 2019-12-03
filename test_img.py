import torch
import numpy as np
from matplotlib import pyplot as plt
from dataset import ImageDataset
import torchvision.utils as vutils
from torch.autograd import Variable
from skimage.transform import resize
from skimage import io
from skimage.measure import compare_psnr
import cv2
from models import Generator1

if __name__ == '__main__':
    img_LR = cv2.imread("test_img_to_present/LR/1_0.png")
    img_HR = cv2.imread("test_img_to_present/HR/1.png")

    img_HR1 = np.array(img_HR)[0:162 * 4, 0:139 * 4, :]
    img_LR1 = np.array(img_LR)[0:162, 0:139, :]

    img_present_LR = img_LR1
    img_present_HR = img_HR1

    #print('img_LR1 numpy', img_LR1.size) # 67554
    #print('img_HR1 numpy', img_HR1.size) # 1080864

    model = 'Generator.pt'

    G = Generator1(num_blocks=18, upsample_factor=4, out_channel_num=20, kernel_size=3, stride_num=1)
    G.load_state_dict(torch.load(model))
    
    img_LR1 = torch.from_numpy(img_LR1)

    #print('to torch LR1', img_LR1.shape) # torch.Size([162, 139, 3])

    img_LR11 = Variable(img_LR1)
    img_LR11 = img_LR11.unsqueeze(dim=0)
    #print('to torch LR11 variable unsqueeze', img_LR11.shape) # torch.Size([1, 162, 139, 3])
    img_LR11 = img_LR11.transpose(1, 3)
    #print('to torch LR11 variable transpose', img_LR11.shape) # torch.Size([1, 3, 139, 162])
    img_LR11 = img_LR11.transpose(2, 3)
    #print('to torch LR11 variable transpose', img_LR11.shape) # torch.Size([1, 3, 162, 139])
    
    fake_image = G(img_LR11.float())
    #print('fake image', fake_image.shape) # torch.Size([1, 3, 648, 556])
    fake_image = fake_image.squeeze(0)
    #print('fake image squeeze', fake_image.shape) # torch.Size([3, 648, 556])

    fake_imageA = np.array(fake_image.detach())
    #print('fake image numpy', fake_image.size) # 1080864

    img_HR11 = torch.from_numpy(img_HR1)
    #print('HR torch from numpy', img_HR11.shape) # torch.Size([648, 556, 3])
    img_HR11 = img_HR11.transpose(1, 2)
    #print('HR transpose', img_HR11.shape) # torch.Size([648, 3, 556])
    img_HR11 = img_HR11.transpose(1, 0)
    #print('HR transpose', img_HR11.shape) # torch.Size([3, 648, 556])

    img_HR11 = np.array(img_HR11.detach())
    #print('HR to numpy', img_HR11.size) # 1080864
    
    img_LR11 = img_LR11.squeeze()
    #print('LR11 squeeze', img_LR11.shape) # torch.Size([3, 162, 139])
    img_LR11 = resize(img_LR11, (3, 648, 556))
    #print('LR11 resized', img_LR11.shape) # (3, 648, 556)

    fake_imageA = resize(fake_imageA, (648, 556, 3))
    fake_image = fake_image.transpose(0, 2)
    fake_image = fake_image.transpose(0, 1)
    print('shape', fake_image.shape)
    fake_image = np.array(fake_image.detach())
    img_HR11 = resize(img_HR11, (648, 556, 3))
    img_LR11 = resize(img_LR11, (162, 139, 3))

    a = max(fake_image.flatten())
    fake_image1 = fake_image/a

    fig = plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img_present_LR)
    plt.title('LR_test_image')

    plt.subplot(1,3,2)
    plt.imshow(img_present_HR)
    plt.title('HR_test_image')

    plt.subplot(1,3,3)
    plt.imshow(fake_image1)
    plt.title('Generated_test_image')

    path_name = 'output_test_images/star_fish.png'
    fig.savefig(path_name, dpi=fig.dpi)

    print(img_HR11.size, fake_image1.size, img_LR11.size) # 1080864 1080864 67554

    psnr_G = compare_psnr(img_HR11, fake_image1)
    #psnr_G1 = compare_psnr(img_HR11, img_LR11)

    print("PSNR value for the generated image =", psnr_G)
    #print("PSNR value for the input LR image =", psnr_G1)
    

# green = 11.225663396419156
# bridge = 10.600961168964815
# hand = 10.806214229740815
# building1 = 11.049487792627723
# sign1 = 9.417731340624078
# star_fish = 12.478617139495515
