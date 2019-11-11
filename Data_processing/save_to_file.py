import torchvision
import numpy as np

'''
Due to the large size of the data, we will not push it onto git
This code transforms pictures into numpy files, which can later be used
'''

HR_train = torchvision.datasets.ImageFolder(root='new/hr_train')
LR_train = torchvision.datasets.ImageFolder(root='new/dstrain')
LR_test = torchvision.datasets.ImageFolder(root='new/ds_test')
LR_valid = torchvision.datasets.ImageFolder(root='new/ds_valid')
HR_test = torchvision.datasets.ImageFolder(root='new/hr_test/hightest')
HR_valid = torchvision.datasets.ImageFolder(root='new/hr_valid')

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

for i in range(len(HR_test_rec)):
    HR_test_rec[i] = np.array(HR_test_rec[i][0])
for i in range(len(HR_valid_rec)):
    HR_valid_rec[i] = np.array(HR_valid_rec[i][0])
for i in range(len(HR_train_rec)):
    HR_train_rec[i] = np.array(HR_train_rec[i][0])

LR_train_rec = []
for index, i in enumerate(LR_train):
    LR_train_rec.append(np.array(i[0]))
print('4')
LR_test_rec = []
for i in LR_test:
    LR_test_rec.append(np.array(i[0]))
print('5')
LR_valid_rec = []
for i in LR_valid:
    LR_valid_rec.append(np.array(i[0]))
print('6')

np.save('new/HR_train', HR_train_rec, allow_pickle=True)
print('7')
np.save('new/HR_valid', HR_valid_rec, allow_pickle=True)
print('8')
np.save('new/HR_test', HR_test_rec, allow_pickle=True)
print('9')
np.save('new/LR_train', LR_train_rec, allow_pickle=True)
print('10')
np.save('new/LR_valid', LR_valid_rec, allow_pickle=True)
print('11')
np.save('new/LR_test', LR_test_rec, allow_pickle=True)
print('12')
