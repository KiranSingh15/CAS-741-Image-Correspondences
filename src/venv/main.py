# declarations
import cv2 as cv
import numpy as np
import orbDetector as orb
from config import get_head_directory
from smoothImage import smooth_image
import inputformatmodule as user_inputs
import directoryManagement as ifc_dir
import os
# User Methods
user_methods, user_params = user_inputs.run_input_format_module()

kern_sz = user_params[0]
sd = user_params[1]
fast_thresh = user_params[2]
bin_sz = user_params[3]
patch_sz = user_params[4]

# directory management
head_dir = ifc_dir.setHeadDirPath()
print("Head directory: ", head_dir)

input_img_dir, local_input_folder = ifc_dir.setInputImgPath(head_dir)   # set the location of input images
input_img_names = ifc_dir.getInputImgNames(input_img_dir)   # get the names of the images to be processed

gs_imagery_path = ifc_dir.createOutputDir(head_dir, 1)    # greyscale images
gk_imagery_path = ifc_dir.createOutputDir(head_dir, 2)    # smoothed Imagery
detected_keypoints_path = ifc_dir.createOutputDir(head_dir, 3)    # keypoints
feature_descriptor_path = ifc_dir.createOutputDir(head_dir, 4)    # feature descriptors
feature_match_path = ifc_dir.createOutputDir(head_dir, 5)    # feature matches

# head_dir = get_head_directory()
# print("the head_directory is: ",head_dir)

# initialize opencv objects
kernel = np.ones((kern_sz,kern_sz),np.float32) / (kern_sz ^ 2)
orb_obj = orb.create_orb_object(bin_sz, patch_sz, fast_thresh)



img_path = input_img_names[0][2]
print(input_img_dir)
img_path = os.path.join(input_img_dir, input_img_names[0][2])
print(img_path)



for i, img_id in enumerate(input_img_names):
    # print("iteration: ", i)
    # # # read the image
    # print(img_id[2])
    img_path = os.path.join(input_img_dir, img_id[2])
    # print(img_path)
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    if img is None:
        print(f"Error: {img_path} could not be read.")
        continue
    else:
        print("No detected errors with path for ", img_id[2])
    # cv.imshow(img_id[0], img)

    # convert to greyscale
    img_gs = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img_path = os.path.join(head_dir, gs_imagery_path, img_id[0])

    # perform smoothing operations
    img_gk = smooth_image(img_gs, kern_sz, sd)

    # search for keypoints
    kp = orb.detect_keypoints_ofast(orb_obj, img_gk)

    # assign feature descriptors
    fd = orb.detect_features_rbrief(orb_obj, img_gk, kp)

    ## Save the results for each image

    # Greyscale Imagery
    full_gs_path = os.path.join(head_dir, gs_imagery_path, img_id[2])
    cv.imwrite(full_gs_path, img_gs)

    # Smoothing
    full_gk_path = os.path.join(head_dir, gk_imagery_path, img_id[2])
    cv.imwrite(full_gk_path, img_gk)

    # Detected Keypoints
    kp_path = os.path.join(head_dir, detected_keypoints_path)
    orb.save_keypoints(kp, img_id[0], kp_path)

    # Feature Descriptors
    print("Feature Descriptors are of type: ", type(fd))
    fd_path = os.path.join(head_dir, feature_descriptor_path)
    orb.save_descriptors(kp, fd, img_id[0], fd_path)
# end loop


# search for matches across all images


# save results



# import colour image
img = cv.imread("cybertruck.jpg", cv.IMREAD_COLOR)
# cv.imshow("Original", img)
# print(img.shape)

# import greyscale image
img_gs = cv.imread("cybertruck.jpg", cv.IMREAD_GRAYSCALE)
cv.imshow("Greyscale", img_gs)
# print(img.shape)

# Apply Gaussian Kernel
# https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

# kernel = np.ones((kern_sz,kern_sz),np.float32) / (kern_sz ^ 2)
# img_gk = cv.filter2D(img_gs,-1,kernel) # filter 2D
# img_gk = cv.GaussianBlur(img_gs, (kern_sz,kern_sz),2)#GaussianBlur
img_gk = smooth_image(img_gs, kern_sz, sd)
cv.imshow("Gaussian Smoothed",img_gk)

# initialize ORB Object
orb_obj = orb.create_orb_object(bin_sz, patch_sz, fast_thresh)

# Identify Key Points
kp = orb.detect_keypoints_ofast(orb_obj, img_gk)
print("keypoints identified")

# Drawing the keypoints
kp_image = cv.drawKeypoints(img_gk, kp, None, color=(0, 255, 0), flags=0)
cv.imshow('ORB', kp_image)



# # Identify descriptors
fd = orb.detect_features_rbrief(orb_obj, img_gk, kp)
print("features identified")

img_with_keypoints = cv.drawKeypoints(img, kp, None, color=(0, 255, 0),
                                      flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv.imshow("ORB Keypoints", img_with_keypoints)


# Match descriptors should be in another file

cv.waitKey(0)
cv.destroyAllWindows()