# declarations
import cv2 as cv

def smooth_image(img_greyscale, sz_kern, std_dev):
    # Inputs #
    # img_greyscale is the input image, assumed to have already been converted to greyscale
    # sz_kern is the size of the Gaussian Kernel
    # sd is the standard deviation of the Gaussian Kernel

    img_smooth = cv.GaussianBlur(img_greyscale, (sz_kern, sz_kern), std_dev)  # GaussianBlur

    return img_smooth