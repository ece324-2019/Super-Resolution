import os
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data = torchvision.datasets.ImageFolder(root='/content/gdrive/My Drive/ECE324 Project/new/', transform=transform)     # size 9600 + size 1600 = 11200

hr_train = []
hr_valid = []
hr_test = []
ds_train = []
ds_valid = []
ds_test = []

name1 = '/content/gdrive/My Drive/ECE324 Project/new/hr/'
name2 = '/content/gdrive/My Drive/ECE324 Project/new/ds/'

for i in range(1, 1601):
  if i < 1201:
    hr_train.append(name1 + str(i) + '.png')
  elif i < 1551:
    hr_valid.append(name1 + str(i) + '.png')
  else:
    hr_test.append(name1 + str(i) + '.png')

for j in range(1, 1601):
  if j < 1201:
    ds_train.append(name2 + str(j) + '_0.png')
    ds_train.append(name2 + str(j) + '_1.png')
    ds_train.append(name2 + str(j) + '_2.png')
    ds_train.append(name2 + str(j) + '_3.png')
    ds_train.append(name2 + str(j) + '_4.png')
    ds_train.append(name2 + str(j) + '_5.png')
  elif j < 1551:
    ds_valid.append(name2 + str(j) + '_0.png')
    ds_valid.append(name2 + str(j) + '_1.png')
    ds_valid.append(name2 + str(j) + '_2.png')
    ds_valid.append(name2 + str(j) + '_3.png')
    ds_valid.append(name2 + str(j) + '_4.png')
    ds_valid.append(name2 + str(j) + '_5.png')
  else:
    ds_test.append(name2 + str(j) + '_0.png')
    ds_test.append(name2 + str(j) + '_1.png')
    ds_test.append(name2 + str(j) + '_2.png')
    ds_test.append(name2 + str(j) + '_3.png')
    ds_test.append(name2 + str(j) + '_4.png')
    ds_test.append(name2 + str(j) + '_5.png')

print(len(ds_train), len(ds_valid), len(ds_test), len(hr_train), len(hr_valid), len(hr_test))
# 7200 2100 300 1200 350 50

# Dividing down-scaled images into 3 folders
for i in range(len(ds_train)):
  new_name = ds_train[i].replace('ds', 'ds_train')
  print(ds_train[i], new_name)
  os.rename(ds_train[i], new_name)
for i in range(len(ds_valid)):
  new_name = ds_valid[i].replace('ds', 'ds_valid')
  os.rename(ds_valid[i], new_name)
for i in range(len(ds_test)):
  new_name = ds_test[i].replace('ds', 'ds_test')
  os.rename(ds_test[i], new_name)


# Dividing higher resolution images into 3 folders
from PIL import Image

for i in range(1, 1201):
  new_name = hr_train[i].replace('hr', 'hr_train')
  new_name = new_name.replace(str(i)+".png", str(i)+"_0.png")
  #print(hr_train[i], new_name)
  os.rename(hr_train[i], new_name)
  for q in range(1, 7):
    img1 = Image.open(new_name)
    img1.save(new_name.replace("_0.png", "_"+str(q)+".png"))
for i in range(len(hr_valid)):
  q = 1201
  new_name = hr_valid[i].replace('hr', 'hr_valid')
  new_name = new_name.replace(str(q)+".png", str(q)+"_0.png")
  os.rename(hr_valid[i], new_name)
  q += 1
  for u in range(1, 7):
    img1 = Image.open(new_name)
    img1.save(new_name.replace("_0.png", "_"+str(u)+".png"))
for i in range(len(hr_test)):
  u = 1551
  new_name = hr_test[i].replace('hr', 'hr_test')
  new_name = new_name.replace(str(u)+".png", str(u)+"_0.png")
  os.rename(hr_test[i], new_name)
  u += 1
  for q in range(1, 7):
    img1 = Image.open(new_name)
    img1.save(new_name.replace("_0.png", "_"+str(q)+".png"))
print("done")
