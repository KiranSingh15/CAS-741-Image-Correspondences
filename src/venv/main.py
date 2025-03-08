# declarations
import cv2 as cv
import numpy as np
import orbDetector as ORB
from smoothImage import smooth_image

############################
# System Inputs - to be updated
kern_sz = 5
sd = 0.25

bin_sz = 2000
patch_sz = 31
fast_thresh = 10



# import colour image
img = cv.imread("cybertruck.jpg", cv.IMREAD_COLOR)
cv.imshow("Original", img)
print( img.shape )

# import greyscale image
img_gs = cv.imread("cybertruck.jpg", cv.IMREAD_GRAYSCALE)
cv.imshow("Greyscale", img_gs)
print( img.shape )

# Apply Gaussian Kernel
# https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

kernel = np.ones((kern_sz,kern_sz),np.float32) / (kern_sz ^ 2)
# img_gk = cv.filter2D(img_gs,-1,kernel) # filter 2D
# img_gk = cv.GaussianBlur(img_gs, (kern_sz,kern_sz),2)#GaussianBlur
img_gk = smooth_image(img_gs, kern_sz, sd)

cv.imshow("Gaussian Smoothed",img_gk)

# initialize ORB Object
orb = ORB.create_orb_object(bin_sz, patch_sz, fast_thresh)

# Identify Key Points
kp = ORB.detect_keypoints_ofast(orb, img_gk)
print("keypoints identified")

# # Identify descriptors
# fd = ORB.detect_features_rbrief(orb, img_gk, kp)
print("features identified")




# Drawing the keypoints
kp_image = cv.drawKeypoints(img_gk, kp, None, color=(0, 255, 0), flags=0)
cv.imshow('ORB', kp_image)


# Match descriptors should be in another file


cv.waitKey(0)
cv.destroyAllWindows()