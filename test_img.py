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
    img_LR = cv2.imread("1580_2.png")
    img_HR = cv2.imread("1580.png")

    img_HR1 = np.array(img_HR)[0:162 * 4, 0:139 * 4, :]
    img_LR1 = np.array(img_LR)[0:162, 0:139, :]

    model = 'Model Checkpoints/Generator.pt'

    G = Generator1(num_blocks=18, upsample_factor=4, out_channel_num=20, kernel_size=3, stride_num=1)
    G.load_state_dict(torch.load(model))
    img_LR1 = torch.from_numpy(img_LR1)

    #print(img_LR1.shape) # torch.Size([162, 139, 3])
    #print(img_HR1.shape) # (648, 556, 3)

    img_LR11 = Variable(img_LR1)
    #print(type(img_LR11)) # --> tensor
    #print(img_LR11.shape) # torch.Size([163, 139, 3])
    img_LR11 = img_LR11.unsqueeze(dim=0)
    img_LR11 = img_LR11.transpose(1, 3)
    img_LR11 = img_LR11.transpose(2, 3)
    #print(img_LR11.shape)
    
    fake_image = G(img_LR11.float())
    fake_image = fake_image.squeeze(0)
    fake_image = fake_image.squeeze(0)

    fake_image = np.array(fake_image.detach())
    #print(np.shape(fake))
    #print(np.shape(real))

    print(fake_image.shape) # (3, 648, 556)
    print(img_HR1.shape) # (648, 556, 3)
    img_LR11 = img_LR11.squeeze()
    print(img_LR11.shape) # (3, 162, 139)

    img_HR11 = torch.from_numpy(img_HR1)
    img_HR11 = img_HR11.transpose(1, 2)
    img_HR11 = img_HR11.transpose(1, 0)
    print(img_HR11.shape) # torch.Size([3, 648, 556])

    img_HR11 = np.array(img_HR11.detach())

    img_LR11 = resize(img_LR11, (3, 648, 556))

    psnr_G = compare_psnr(img_HR11, fake_image)
    psnr_G1 = compare_psnr(img_HR11, img_LR11)

    fake_image = resize(fake_image, (648, 556, 3))
    img_HR11 = resize(img_HR11, (648, 556, 3))
    img_LR11 = resize(img_LR11, (162, 139, 3))

    a = max(fake_image.flatten())
    fake_image1 = fake_image/a

    fig = plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(fake_image1)
    plt.title('Generated_test_image')

    plt.subplot(1,3,2)
    plt.imshow(img_LR11)
    plt.title('LR_test_image')

    plt.subplot(1,3,3)
    plt.imshow(img_HR11)
    plt.title('HR_test_image')

    path_name = 'test_output_image.png'
    fig.savefig(path_name, dpi=fig.dpi)

    print("PSNR value for the generated image =", psnr_G)
    print("PSNR value for the input LR image =", psnr_G1)
    
