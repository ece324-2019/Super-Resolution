import numpy as np
import matplotlib.pyplot as plot


def average(img):
    # create an empty array with 4x size of the original image
    new_img = np.array([[[0., 0., 0.]] * 2 * len(img[0])] * 2 * len(img))
    img = img[:,:,0:3]

    # fill up the new image with the original one in alternating rows and columns
    for i in range(len(img)):
        for j in range(len(img[0])):
            #new_img[i * 2][np.array([j * 2])] = img[i, j, :]
            new_img[i * 2, j * 2, :] = img[i, j][0:3]

    # compute the average RGB of the neighbours for the empty ones
    for i in range(len(img) - 1):
        for j in range(len(img[0]) - 1):
            new_img[i * 2 + 1][np.array([j * 2])] = (img[i, j, :] + img[i + 1, j, :]) / 2
            new_img[i * 2][np.array([j * 2 + 1])] = (img[i, j, :] + img[i, j + 1, :]) / 2
            new_img[i * 2 + 1][np.array([j * 2 + 1])] = (img[i, j, :] + img[i + 1, j, :] +
                                                         img[i, j + 1, :] + img[i + 1, j + 1, :]) / 4

    # fill up the margins
    for i in range(len(img) - 1):
        new_img[i * 2 + 1][np.array([-2])] = (img[i, -1] + img[i + 1, -1]) / 2
        new_img[i * 2][np.array([-1])] = new_img[i * 2, -2]
        new_img[i * 2 + 1][np.array([-1])] = new_img[i * 2 + 1, -2]
    for i in range(len(img[0]) - 1):
        new_img[-2][np.array([2 * i + 1])] = (img[-1, i] + img[-1, i + 1]) / 2
        new_img[-1][np.array([i * 2])] = new_img[-2, i * 2]
        new_img[-1][np.array([i * 2 + 1])] = new_img[-2, i * 2 + 1]
    new_img[-1][np.array([-1])] = img[-1, -1]
    new_img[-1][np.array([-2])] = img[-1, -1]
    new_img[-2][np.array([-1])] = img[-1, -1]

    return new_img


def main():
    img_name = '16_0'
    ext = '.png'

    # load image
    print(img_name+ext)
    img = plot.imread(img_name + ext)

    img = average(average(img))

    plot.imsave('improved' + img_name + ext, img)
    plot.imshow(img)
    plot.show()


if __name__ == '__main__':
    main()
