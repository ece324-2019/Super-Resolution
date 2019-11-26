from skimage.measure import compare_psnr
from PIL import Image
import cv2
import numpy 
from skimage.transform import resize

def psnr_baseline():
    #img1 = Image.open("Baseline_Model/1_0.png")
    img1 = cv2.imread("Baseline_Model/1_0.png")
    a = numpy.asarray(img1)
    print(a.shape)
    img2 = cv2.imread("Baseline_Model/improved1_0.png")
    b = numpy.asarray(img2)
    bb = resize(b, (163, 139))
    print(b.shape)
    print(bb.shape)
    value1 = compare_psnr(a, bb)
    return value1

if __name__ == '__main__':
    val1 = psnr_baseline()
    print("The PSNR value for the baseline model for image #1 is", val1)