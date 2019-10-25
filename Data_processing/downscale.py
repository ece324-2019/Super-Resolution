import numpy as np
import matplotlib.pyplot as plot
import random
import cv2
# need to install cv2 library
# pip install opencv-python

'''
for image reshaping:
method 1 is taking the pixel with the max sum
method 2 is taking the pixel with the max squared sum
method 3 is taking the pixel with the min sum
method 4 is taking the pixel with the min squared sum
method 5 is taking the average of the reduced pixels
method 6 is taking the RMS of the reduced pixels
method 7 is selecting a random reduced pixel

for image smoothing:
method 1 is averaging
method 2 is Gaussian filtering
'''


# take parameters: image name, downscale factor on each dimension and the downscale method
def downscale(img_name, factor, method):
    img = plot.imread(img_name + '.png')
    new_img = np.array([[[0., 0., 0.]] * int(len(img[0]) / factor)] * int(len(img) / factor))
    # max sum
    if method not in range(1, 8):
        raise Exception
    elif method == 1:
        for i in range(len(new_img)):
            for j in range(len(new_img[0])):
                candidates = np.array(img[factor * i:factor * (i + 1), factor * j:factor * (j + 1)])
                weights = [sum(candidates[p, q]) for p in range(factor) for q in range(factor)]
                index = weights.index(max(weights))
                new_img[i][np.array([j])] = candidates.flatten()[3 * index: 3 * index + 3]
        return new_img
    # max squared sum
    elif method == 2:
        for i in range(len(new_img)):
            for j in range(len(new_img[0])):
                candidates = np.array(img[factor * i:factor * (i + 1), factor * j:factor * (j + 1)])
                weights = [sum(map(lambda x: x * x, candidates[p, q])) for p in range(factor) for q in range(factor)]
                index = weights.index(max(weights))
                new_img[i][np.array([j])] = candidates.flatten()[3 * index: 3 * index + 3]
        return new_img
    # min sum
    elif method == 3:
        for i in range(len(new_img)):
            for j in range(len(new_img[0])):
                candidates = np.array(img[factor * i:factor * (i + 1), factor * j:factor * (j + 1)])
                weights = [sum(candidates[p, q]) for p in range(factor) for q in range(factor)]
                index = weights.index(min(weights))
                new_img[i][np.array([j])] = candidates.flatten()[3 * index: 3 * index + 3]
        return new_img
    # min squared sum
    elif method == 4:
        for i in range(len(new_img)):
            for j in range(len(new_img[0])):
                candidates = np.array(img[factor * i:factor * (i + 1), factor * j:factor * (j + 1)])
                weights = [sum(map(lambda x: x * x, candidates[p, q])) for p in range(factor) for q in range(factor)]
                index = weights.index(min(weights))
                new_img[i][np.array([j])] = candidates.flatten()[3 * index: 3 * index + 3]
        return new_img
    # average
    elif method == 5:
        for i in range(len(new_img)):
            for j in range(len(new_img[0])):
                candidates = np.array(img[factor * i:factor * (i + 1), factor * j:factor * (j + 1)])
                weights = sum(sum(candidates))
                new_img[i][np.array([j])] = weights / factor / factor
        return new_img
    # RMS
    elif method == 6:
        for i in range(len(new_img)):
            for j in range(len(new_img[0])):
                candidates = np.array(img[factor * i:factor * (i + 1), factor * j:factor * (j + 1)])
                weights = sum(sum(map(lambda x: x * x, candidates)))
                new_img[i][np.array([j])] = [np.sqrt(x / factor / factor) for x in weights]
        return new_img
    # randomly select
    elif method == 7:
        for i in range(len(new_img)):
            for j in range(len(new_img[0])):
                candidates = np.array(img[factor * i:factor * (i + 1), factor * j:factor * (j + 1)])
                index = random.randrange(factor * factor)
                new_img[i][np.array([j])] = candidates.flatten()[3 * index: 3 * index + 3]
        return new_img


def smoothing(img, method, smoothing_factor):
    if method not in [1, 2]:
        raise Exception
    # averaging
    elif method == 1:
        blur = cv2.blur(img, (smoothing_factor, smoothing_factor))
        return blur
    # Gaussian filter
    elif method == 2:
        blur = cv2.GaussianBlur(img, (smoothing_factor, smoothing_factor), 0)
        return blur


# generate a downscaled image from a HR image
def generate(img_name, ext, downscale_method, downscale_factor, smoothing_method, smoothing_factor):
    try:
        img = smoothing(downscale(img_name, downscale_factor, downscale_method), smoothing_method, smoothing_factor)
        plot.imsave(img_name + 'x' + str(downscale_factor) + '_' + str(downscale_method)
                    + str(smoothing_method) + ext, img)
        return img
    except Exception:
        print('An error occurred, and the process cannot be completed')
        return False


def main():
    img_name = '0001'
    ext = '.png'
    downscale_method = 6
    downscale_factor = 4
    smoothing_method = 2
    smoothing_factor = 5

    img = generate(img_name, ext, downscale_method, downscale_factor, smoothing_method, smoothing_factor)
    # plot.imshow(img)
    # plot.show()


if __name__ == '__main__':
    main()
