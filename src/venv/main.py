# declarations
import cv2 as cv
import numpy as np


# image inputs
img = cv.imread("cybertruck.jpg", cv.IMREAD_COLOR)
cv.imshow("Unaltered", img)


# Convert to greyscale
img_gs = cv.imread("cybertruck.jpg", cv.IMREAD_GRAYSCALE)
cv.imshow("Greyscale", img_gs)

# Apply Gaussian Kernel
# https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
kern_sz = 5
kernel = np.ones((kern_sz,kern_sz),np.float32) / (kern_sz ^ 2)
# img_gk = cv.filter2D(img_gs,-1,kernel) # filter 2D
img_gk = cv.GaussianBlur(img_gs, (kern_sz,kern_sz),2)#GaussianBlur

cv.imshow("Gaussian Smoothed",img_gk)

# Identify Key Points


# Identify descriptors


# Match descriptors should be in another file


cv.waitKey(0)
cv.destroyAllWindows()