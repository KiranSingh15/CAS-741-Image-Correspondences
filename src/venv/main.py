## Library and Function Declarations
# Standard Libraries
import os
import numpy as np
import cv2 as cv

# User-Defined Functions
import inputformatmodule as user_inputs
import directoryManagement as ifc_dir
from smoothImage import smooth_image
import orbDetector as orb
import featureMatch as feat_match

# User Methods
user_methods, user_params = user_inputs.run_input_format_module()

kern_sz = user_params[0]
sd = user_params[1]
fast_thresh = user_params[2]
bin_sz = user_params[3]
patch_sz = user_params[4]
match_dist_thresh = user_params[5]

# directory management
head_dir = ifc_dir.setHeadDirPath()
# print("Head directory: ", head_dir)

input_img_dir, local_input_folder = ifc_dir.setInputImgPath(head_dir)   # set the location of input images
input_img_names, num_images = ifc_dir.getInputImgNames(input_img_dir)   # get the names of the images to be processed

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
img_ID = []

# img_path = input_img_names[0][2]
# print(input_img_dir)
img_path = os.path.join(input_img_dir, input_img_names[0][2])
# print(img_path)   # uncomment only to support debugging

for i, img_id in enumerate(input_img_names):
    # print("iteration: ", i)
    # print(type(i))  # uncomment only to support debugging

    img_ID.append([i]) # assigns unique image IDs

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
    full_gs_path = os.path.join(head_dir, gs_imagery_path, img_id[2])
    cv.imwrite(full_gs_path, img_gs)

    # perform smoothing operations
    img_gk = smooth_image(img_gs, kern_sz, sd)
    full_gk_path = os.path.join(head_dir, gk_imagery_path, img_id[2])
    cv.imwrite(full_gk_path, img_gk)

    # search for keypoints
    kp = orb.detect_keypoints_ofast(orb_obj, img_gk)
    kp_path = os.path.join(head_dir, detected_keypoints_path)
    orb.save_keypoints(kp, img_id[0], kp_path)

    # assign feature descriptors
    fd = orb.detect_features_rbrief(orb_obj, img_gk, kp)
    fd_path = os.path.join(head_dir, feature_descriptor_path)
    orb.save_descriptors(kp, fd, img_id[0], fd_path)


    ## Save the results for each image
    # moved into each individual function

# end loop

## search for matches across all images
# create brute force matching object
bf = feat_match.create_brute_force_matcher()

# this indexing method ensures that no images are redundantly cross-checked
for i in range(num_images):
    # retrieve descriptors for image i, also known as the query image
    img_inst_1 = input_img_names[i][0]
    img_path_1 = os.path.join(input_img_dir, input_img_names[i][2])
    img_1 = cv.imread(img_path_1,cv.IMREAD_GRAYSCALE)

    kp1, fd1 = feat_match.load_orb_descriptors(img_inst_1, feature_descriptor_path)

    for j in range(i + 1, num_images):  # ensures that no image is compared with itself
        # retrieve descriptors for image j, also known as the training image
        # print(f"i: {i}, j: {j}")
        img_inst_2 = input_img_names[j][0]
        img_path_2 = os.path.join(input_img_dir, input_img_names[j][2])
        img_2 = cv.imread(img_path_2, cv.IMREAD_GRAYSCALE)

        kp2, fd2 = feat_match.load_orb_descriptors(img_inst_2, feature_descriptor_path)

        # execute feature matching
        best_matches = feat_match.sort_matches(feat_match.match_descriptors(bf, fd1, fd2))


        # save feature matches to csv
        images_under_comparison = img_inst_1 + "_" + img_inst_2
        feat_match.save_matches_to_csv(best_matches, kp1, kp2, images_under_comparison, feature_match_path)
        # NOTE: feature matches should be rounded to the nearest coordinate


        ## Need to use the image identifiers (img_inst_1, img_inst_2) and add it to the list of matches

        # if the dataframe has been created, append the new feature matches, otherwise

        # define the combined dataframe


        # visualize matches between images
        feat_match.display_screened_matches(img_1, kp1, img_2, kp2, best_matches, match_dist_thresh)

        cv.waitKey(0)
        cv.destroyAllWindows()

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
print("Cybertruck keypoints identified")

# Drawing the keypoints
kp_image = cv.drawKeypoints(img_gk, kp, None, color=(0, 255, 0), flags=0)
cv.imshow('Unadjusted ORB', kp_image)



# # Identify descriptors
fd = orb.detect_features_rbrief(orb_obj, img_gk, kp)
print("Cybertruck features identified")
img_with_keypoints = cv.drawKeypoints(img, kp, None, color=(0, 255, 0),
                                      flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv.imshow("ORB with Keypoints", img_with_keypoints)



cv.waitKey(0)
cv.destroyAllWindows()