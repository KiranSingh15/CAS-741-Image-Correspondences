import os

import cv2 as cv
import FeatureDescriptorModule as assignDescriptors
import FeatureMatchingModule as matchFeatures
import ImagePlotModule as plotImage
import ImageSmoothingModule as smoothImage
import InputFormatModule as config
import KeypointDetectionModule as detectKeypoints
import OutputFormatModule as formatOutput
import OutputVerificationModule as verifyOutput

## Run the Input Format Module
head_dir = config.get_head_directory()

# create Output folder if one does not exist
formatOutput.make_directory(head_dir, "Outputs")

# define Output subfolders
(
    greyscale_folder_nm,
    smoothed_imagery_folder_nm,
    keypoint_folder_nm,
    descriptor_folder_nm,
    matches_folder_nm,
) = formatOutput.define_output_folders()

# scan for user-defined features
mthd_img_smoothing, mthd_kp_detection, mthd_kp_description, mthd_ft_match = (
    config.get_active_methods()
)

k, sigma, t, b, p, d, n_matches_disp = config.get_chosen_parameters()

input_img_dir, local_input_folder = config.set_input_img_path(
    head_dir
)  # set the location of input images
input_image_names, num_images = config.get_img_IDs(head_dir)
print("Number of images identified: ", num_images)

# Cycle through each image to smooth, identify keypoints, and extract descriptors

# declare orb_object
orb_object = detectKeypoints.initialize_orb(
    mthd_kp_detection, mthd_kp_description, b, p, k
)

# declare bfm_object
bf_matcher_object = matchFeatures.create_BF_matcher(mthd_ft_match, cv.NORM_HAMMING)


img_IDs = []

for i, img_id in enumerate(input_image_names):
    img_IDs.append([i])  # assigns unique image IDs

    # read the raw image
    img_path = os.path.join(input_img_dir, img_id[2])
    img = cv.imread(img_path, cv.IMREAD_COLOR)

    # verify that the image is not corrupted
    config.verify_imported_image(img, img_path, img_id[2])

    # import greyscale image
    img_gs = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    plotImage.save_image(img_gs, head_dir, greyscale_folder_nm, img_id[2])

    # Image Smoothing Module
    img_gk = smoothImage.smooth_image(mthd_img_smoothing, img_gs, k, sigma)
    plotImage.save_image(img_gk, head_dir, smoothed_imagery_folder_nm, img_id[2])

    # Keypoint Detection Module
    keypoints = detectKeypoints.detect_keypoints_ofast(
        mthd_kp_detection, orb_object, img_gk
    )
    # output keypoints
    formatOutput.output_keypoints(keypoints, img_id[0], head_dir, keypoint_folder_nm)

    # generate kp image
    img_kp = plotImage.gen_kp_img(img_gk, keypoints, 0)
    # save kp image
    plotImage.save_image(img_kp, head_dir, keypoint_folder_nm, img_id[2])

    # Feature Descriptor Module
    descriptors = assignDescriptors.compute_descriptors(
        mthd_kp_description, orb_object, img, keypoints
    )
    # output descriptors
    formatOutput.output_descriptors(
        keypoints, descriptors, img_id[0], head_dir, descriptor_folder_nm
    )

    # generate desc image
    img_desc = plotImage.gen_kp_img(
        img_gk, keypoints, cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    # save desc image
    plotImage.save_image(img_desc, head_dir, descriptor_folder_nm, img_id[2])

print("GS conversion complete.")
print("Image smoothing complete.")
print("Keypoint detection complete.")
print("Feature assignment complete.")

# Execute Feature Matching Module
fd_path = config.get_descriptor_path(head_dir, descriptor_folder_nm)

for i in range(num_images):
    img1_name = input_image_names[i][0]
    img1_path = os.path.join(input_img_dir, input_image_names[i][2])
    img_1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    kp1, fd1 = config.load_orb_descriptors(img1_name, fd_path)

    for j in range(i + 1, num_images):
        img2_name = input_image_names[j][0]
        img2_path = os.path.join(input_img_dir, input_image_names[j][2])
        img_2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
        kp2, fd2 = config.load_orb_descriptors(img2_name, fd_path)

        # Perform feature matching and sorting
        # execute feature matching
        matches = matchFeatures.match_features(bf_matcher_object, fd1, fd2)
        matches = matchFeatures.sort_matches(matches)

        # check that matches originate from unique images
        verifyOutput.check_match_uniqueness(img1_name, img2_name, matches)

        match_images = img1_name + "_" + img2_name
        img_ext = match_images + ".png"
        formatOutput.output_matches(
            img1_name, img2_name, matches, kp1, kp2, head_dir, matches_folder_nm
        )

        img_matches = plotImage.gen_matched_features(
            img_1, img_2, kp1, kp2, matches, n_matches_disp, d
        )
        plotImage.save_image(img_matches, head_dir, matches_folder_nm, img_ext)

        cv.waitKey(0)
        cv.destroyAllWindows()
print("Feature matching complete.")

print("Program flow complete!")
