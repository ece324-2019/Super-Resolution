'''
import numpy as np
import matplotlib.pyplot as plt
LR_train = np.load('LR_train.npy', allow_pickle=True)
HR_train = np.load('HR_train.npy', allow_pickle=True)

for i in range(len(LR_train)):
    LR_train[i] = np.array(LR_train[i])[0:162, 0:139, :]
for i in range(len(HR_train)):
    HR_train[i] = np.array(HR_train[i])[0:162 * 4, 0:139 * 4, :]

for i in range(5):
    plt.imshow(LR_train[6*i])
    plt.show()
    plt.imshow(HR_train[6*i])
    plt.show()
'''

import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob


#HR_train = torchvision.datasets.ImageFolder(root='new/hr_train')
#LR_train = torchvision.datasets.ImageFolder(root='new/ds_train')
#LR_test = torchvision.datasets.ImageFolder(root='new/ds_test')
#LR_valid = torchvision.datasets.ImageFolder(root='new/ds_valid')
#HR_test = torchvision.datasets.ImageFolder(root='new/hr_test')
#HR_valid = torchvision.datasets.ImageFolder(root='new/hr_valid')


LR_train_rec = []
for i in range(1, 1201):
    for j in range(6):
        name = str(i)+'_'+str(j)+'.png'
        im = Image.open('new/ds_train/ds_train_pic/'+name)
        LR_train_rec.append(np.array(im))
print(len(LR_train_rec))
HR_train_rec = []
for i in range(1, 1201):
    for j in range(6):
        name = str(i)+'.png'
        im = Image.open('new/hr_train/hr_train_pic/'+name)
        HR_train_rec.append(np.array(im))
print(len(HR_train_rec))

LR_valid_rec = []
for i in range(1201, 1551):
    for j in range(6):
        name = str(i)+'_'+str(j)+'.png'
        im = Image.open('new/ds_valid/ds_valid_pic/'+name)
        LR_valid_rec.append(np.array(im))
print(len(LR_valid_rec))
HR_valid_rec = []
for i in range(1201, 1551):
    for j in range(6):
        name = str(i)+'.png'
        im = Image.open('new/hr_valid/hr_valid_pic/'+name)
        HR_valid_rec.append(np.array(im))
print(len(HR_valid_rec))

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

'''
HR_test_rec = []
for j, item in enumerate(HR_test):
    for i in range(6):
        HR_test_rec.append(item)
print('1')
HR_valid_rec = []
for j, item in enumerate(HR_valid):
    for i in range(6):
        HR_valid_rec.append(item)
print('2')
HR_train_rec = []
for j, item in enumerate(HR_train):
    for i in range(6):
        HR_train_rec.append(item)
print('3')

LR_test_rec = []
for j, item in enumerate(LR_test):
    LR_test_rec.append(item)
print('1')
LR_valid_rec = []
for j, item in enumerate(LR_valid):
    LR_valid_rec.append(item)
print('2')
#LR_train_rec = []
#for j, item in enumerate(LR_train):
#    LR_train_rec.append(item)
#print('3')

for i in range(len(HR_test_rec)):
    HR_test_rec[i] = np.array(HR_test_rec[i][0])
for i in range(len(HR_valid_rec)):
    HR_valid_rec[i] = np.array(HR_valid_rec[i][0])
for i in range(len(HR_train_rec)):
    HR_train_rec[i] = np.array(HR_train_rec[i][0])

for i in range(len(LR_test_rec)):
    LR_test_rec[i] = np.array(LR_test_rec[i][0])
for i in range(len(HR_valid_rec)):
    LR_valid_rec[i] = np.array(LR_valid_rec[i][0])
print(len(LR_train_rec))
for i in range(len(LR_train_rec)):
    LR_train_rec[i] = np.array(LR_train_rec[i])


for i in range(len(LR_train_rec)):
    LR_train_rec[i] = np.array(LR_train_rec[i])[0:162, 0:139, :]
for i in range(len(HR_train_rec)):
    HR_train_rec[i] = np.array(HR_train_rec[i])[0:162 * 4, 0:139 * 4, :]
'''

for i in range(5):
    plt.imshow(LR_train_rec[6*i])
    plt.show()
    plt.imshow(HR_train_rec[6*i])
    plt.show()
