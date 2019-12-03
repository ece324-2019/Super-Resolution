import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch
import models
from baseline_model import *
from skimage.measure import compare_psnr

num = 49

LR_test_rec = []
for i in range(1551, 1601):
    for j in range(6):
        name = str(i)+'_'+str(j)+'.png'
        im = Image.open('new/ds_test/ds_test_pic/'+name)
        LR_test_rec.append(np.array(im))
print(len(LR_test_rec))
HR_test_rec = []
for i in range(1551, 1601):
    for j in range(6):
        name = str(i)+'.png'
        im = Image.open('new/hr_test/hightest/hr_test_pic/'+name)
        HR_test_rec.append(np.array(im))
print(len(HR_test_rec))

bl1 = average(average(LR_test_rec[num]))
max_num = max(bl1.flatten())
min_num = min(bl1.flatten())
bl1 = (bl1 - min_num)/(max_num-min_num)

plt.imshow(bl1)
plt.show()

for i in range(len(HR_test_rec)):
    HR_test_rec[i] = np.array(HR_test_rec[i])[0:162 * 4, 0:139 * 4, :]

for i in range(len(LR_test_rec)):
    LR_test_rec[i] = np.array(LR_test_rec[i])[0:162, 0:139, :]

model = 'Generator.pt'
G = models.Generator1(num_blocks=18, upsample_factor=4, out_channel_num=20, kernel_size=3, stride_num=1)
G.load_state_dict(torch.load(model))
lr = torch.FloatTensor(LR_test_rec)
lr = torch.transpose(lr, 1, 3)
fake = G(lr[num:num+2])
fake = torch.transpose(fake, 1, 3)
img = fake.detach().numpy()

temp = img[0]
max_num = max(temp.flatten())
min_num = min(temp.flatten())
temp = (temp - min_num)/(max_num-min_num)



fig = plt.figure()
plt.subplot(1,4,1)
plt.imshow(LR_test_rec[num])
plt.title('LR_test_image')

plt.subplot(1,4,2)
plt.imshow(HR_test_rec[num])
plt.title('HR_test_image')

plt.subplot(1,4,3)
plt.imshow(temp)
plt.title('Generated_test_image')

plt.subplot(1,4,4)
plt.imshow(bl1)
plt.title('Baseline_image')

path_name = 'img.png'
fig.savefig(path_name, dpi=fig.dpi)

plt.show()

fake_psnr = compare_psnr(HR_test_rec[num], temp)
base_psnr = compare_psnr(HR_test_rec[num], bl1)

print("fake_psnr: ", fake_psnr, "   base_psnr: ", base_psnr)
