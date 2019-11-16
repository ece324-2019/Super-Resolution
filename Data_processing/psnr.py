import matplotlib.pyplot as plt

psnr = []
f = open("psnr.rtf", "r")
for l in f:
    if l[0:8] == 'psnr:  [':
        temp = l[8:]
        temp = temp[0:-3]
        psnr.append(float(temp))

plt.plot(range(len(psnr)), psnr)
plt.title('generator psnr vs epoch')
plt.xlabel('epoch')
plt.ylabel('psnr')
plt.show()
