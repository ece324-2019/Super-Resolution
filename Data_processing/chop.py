import torchvision

data = torchvision.datasets.ImageFolder(root='./ds')

imgs = []

for pics in data:
    imgs += [pics[0]]

for i in range(len(imgs)/6):
    w, h = imgs[6*i].size
    w_edge = (w - 278) / 2
    h_edge = (h - 162) / 2
    for j in range(6):
        imgs[6*i + j].crop((w_edge, h_edge, w / 2, h - h_edge)).save('new/ds/' + str(2*i + 1) + '_' + str(j) + '.png')
        imgs[6*i + j].crop((w/2, h_edge, w - w_edge, h - h_edge)).save('new/ds/' + str(2*i + 2) + '_' + str(j) + '.png')
    print(i)

'''
for i in range(len(imgs)):
    w, h = imgs[6*i].size
    w_edge = (w - 1112) / 2
    h_edge = (h - 648) / 2
    imgs[i].crop((w_edge, h_edge, w / 2, h - h_edge)).save('new/hr/' + str(2*i + 1) + '.png')
    imgs[i].crop((w/2, h_edge, w - w_edge, h - h_edge)).save('new/hr/' + str(2*i + 2) + '.png')
    print(i)
'''
