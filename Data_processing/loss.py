import matplotlib.pyplot as plt

loss = []
f = open("srgan.txt", "r")
for l in f:
    if l[0:24] == 'loss functions:  tensor(':
        temp = l[24:]
        index = temp.index(',')
        temp = temp[0:index]
        loss.append(float(temp))

plt.plot(range(len(loss)), loss)
plt.title('generator loss vs epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
