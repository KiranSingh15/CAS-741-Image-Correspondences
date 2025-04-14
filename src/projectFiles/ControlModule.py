import os

import cv2 as cv

try:
    # Relative imports for testing/packaging context
    from . import FeatureDescriptorModule as assignDescriptors
    from . import FeatureMatchingModule as matchFeatures
    from . import ImagePlotModule as plotImage
    from . import ImageSmoothingModule as smoothImage
    from . import InputFormatModule as config
    from . import KeypointDetectionModule as detectKeypoints
    from . import OutputFormatModule as formatOutput
    from . import OutputVerificationModule as verifyOutput

except ImportError:
    # Absolute imports for direct execution
    import FeatureDescriptorModule as assignDescriptors
    import FeatureMatchingModule as matchFeatures
    import ImagePlotModule as plotImage
    import ImageSmoothingModule as smoothImage
    import InputFormatModule as config
    import KeypointDetectionModule as detectKeypoints
    import OutputFormatModule as formatOutput
    import OutputVerificationModule as verifyOutput


def main():
    # Run the Input Format Module
    head_dir = config.get_head_directory()
    output_root = head_dir / "Outputs"
    formatOutput.make_directory(head_dir, "Outputs")

    (
        greyscale_folder_nm,
        smoothed_imagery_folder_nm,
        keypoint_folder_nm,
        descriptor_folder_nm,
        matches_folder_nm,
    ) = formatOutput.define_output_folders()

    mthd_img_smoothing, mthd_kp_detection, mthd_kp_description, mthd_ft_match = (
        config.get_active_methods()
    )

    k, sigma, t, b, p, d, n_matches_disp = config.get_chosen_parameters()

    input_img_dir, local_input_folder = config.set_input_img_path(head_dir)
    input_image_names, num_images = config.get_img_IDs(head_dir)
    print("Number of images identified: ", num_images)

    orb_object = detectKeypoints.initialize_orb(
        mthd_kp_detection, mthd_kp_description, b, p, k
    )
    bf_matcher_object = matchFeatures.create_BF_matcher(mthd_ft_match, cv.NORM_HAMMING)

    img_IDs = []

    for i, img_id in enumerate(input_image_names):
        img_IDs.append([i])
        img_path = os.path.join(input_img_dir, img_id[2])
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        config.verify_imported_image(img, img_path, img_id[2])

        # Greyscale conversion
        img_gs = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        plotImage.save_image(img_gs, output_root, greyscale_folder_nm, img_id[2])

        # Image smoothing
        img_gk = smoothImage.smooth_image(mthd_img_smoothing, img_gs, k, sigma)
        plotImage.save_image(img_gk, output_root, smoothed_imagery_folder_nm, img_id[2])

        # Keypoint detection
        keypoints = detectKeypoints.detect_keypoints_ofast(
            mthd_kp_detection, orb_object, img_gk
        )
        formatOutput.output_keypoints(
            keypoints, img_id[0], output_root, keypoint_folder_nm
        )

        img_kp = plotImage.gen_kp_img(img_gk, keypoints, 0)
        plotImage.save_image(img_kp, output_root, keypoint_folder_nm, img_id[2])

        # Descriptor assignment
        descriptors = assignDescriptors.compute_descriptors(
            mthd_kp_description, orb_object, img, keypoints
        )
        formatOutput.output_descriptors(
            keypoints, descriptors, img_id[0], output_root, descriptor_folder_nm
        )

        img_desc = plotImage.gen_kp_img(
            img_gk, keypoints, cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )
        plotImage.save_image(img_desc, output_root, descriptor_folder_nm, img_id[2])

    print("GS conversion complete.")
    print("Image smoothing complete.")
    print("Keypoint detection complete.")
    print("Feature assignment complete.")

    fd_path = config.get_descriptor_path(head_dir, descriptor_folder_nm)

    print("Number of comparisons: ", int((num_images - 1) * num_images / 2))
    comp_count = 0

    for i in range(num_images):
        img1_name = input_image_names[i][0]
        img1_path = os.path.join(input_img_dir, input_image_names[i][2])
        img_1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
        kp1, fd1 = config.load_orb_descriptors(img1_name, fd_path)

        for j in range(i + 1, num_images):
            # increase comparison counter
            comp_count += 1

            img2_name = input_image_names[j][0]
            img2_path = os.path.join(input_img_dir, input_image_names[j][2])
            img_2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
            kp2, fd2 = config.load_orb_descriptors(img2_name, fd_path)

            matches = matchFeatures.match_features(bf_matcher_object, fd1, fd2)
            matches = matchFeatures.sort_matches(matches)
            verifyOutput.check_match_uniqueness(img1_name, img2_name, matches)

            match_images = img1_name + "_" + img2_name
            img_ext = match_images + ".png"

            formatOutput.output_matches(
                img1_name,
                img2_name,
                matches,
                kp1,
                kp2,
                fd1,
                fd2,
                output_root,
                matches_folder_nm,
            )

            img_matches = plotImage.gen_matched_features(
                img_1, img_2, kp1, kp2, matches, n_matches_disp, d
            )
            plotImage.save_image(img_matches, output_root, matches_folder_nm, img_ext)

            cv.waitKey(0)
            cv.destroyAllWindows()

            if comp_count % 10 == 1:
                print("Completed", comp_count - 1, "comparisons.")

    print("Feature matching complete.")
    print("Program flow complete!")


if __name__ == "__main__":
    main()
