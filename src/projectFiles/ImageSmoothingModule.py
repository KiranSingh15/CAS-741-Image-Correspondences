# declarations
import cv2 as cv


def convert_to_greyscale(img_colour):
    img_greyscale = cv.cvtColor(img_colour, cv.COLOR_BGR2GRAY)
    return img_greyscale


def smooth_image(mthd_img_smoothing, img_greyscale, sz_kern, std_dev):
    # Inputs #
    # mthd_img_smoothing is the key that defines what method of image smoothing to use
    # img_greyscale is the input image, assumed to have already been converted to greyscale
    # sz_kern is the size of the Gaussian Kernel
    # sd is the standard deviation of the Gaussian Kernel

    if mthd_img_smoothing == 1:
        img_smooth = cv.GaussianBlur(
            img_greyscale, (sz_kern, sz_kern), std_dev
        )  # GaussianBlur
        return img_smooth


    return None # Return explicitly when method is not supported
