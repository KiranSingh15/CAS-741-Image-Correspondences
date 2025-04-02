import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import SpecificationParametersModule as specParams

head_directory = Path(os.getcwd())


def get_head_directory():
    # head_directory = Path(os.getcwd())
    return head_directory  # Convert to Path object


def get_active_methods():
    mthd_img_smoothing, mthd_kp_detection, mthd_kp_description, mthd_ft_match = (
        specParams.get_default_methods()
    )
    return mthd_img_smoothing, mthd_kp_detection, mthd_kp_description, mthd_ft_match


def get_chosen_parameters():
    k, sigma, t, b, p = specParams.get_default_parameters()  # where
    return k, sigma, t, b, p


def set_input_img_path(head_dir):
    local_folder = "Raw_Images"
    img_dir = head_dir / local_folder  # Use '/' for safe path joining
    # print(img_dir)    # uncomment for debugging only
    return img_dir, local_folder


def get_img_IDs(head_dir):
    local_folder = "Raw_Images"
    # print("Head directory: ", head_dir)
    img_dir = head_dir / local_folder  # Use '/' for safe path joining
    # print(img_dir)    # uncomment for debugging only
    input_img = [
        (file.stem, file.suffix, file.name)
        for file in img_dir.iterdir()
        if file.is_file()
    ]
    num_images = len(input_img)
    # print ("IFM: ", num_images)    # uncomment for debugging only
    return input_img, num_images


def check_limits(u_sz_kern, u_std_dev, u_fast_thr, u_bin_sz, u_patch_sz):
    # Gaussian Filtering
    kern_bounds = [3, 15]  # 3 and 11 inclusive to scale down the kernel
    sd_bounds = [0, 10]  # (0, 10], or 0 exclusive and 10 inclusive

    # Keypoint detection
    fast_bounds = [2, 254]  # inclusive

    # Feature Detector
    bin_bounds = [1, 2048]
    patch_sz = [5, 100]

    # kernel size
    assert (
        u_sz_kern > kern_bounds[0]
    ), f"badKernelSize. Kernel must be > 1. (Kernel Size = {u_sz_kern})"
    assert (
        u_sz_kern < kern_bounds[1]
    ), f"badKernelSize. Kernel must be <= 15. (Kernel Size = {u_sz_kern})"
    assert (
        u_sz_kern % 2 == 1
    ), f"badKernelSize. Kernel must be odd. (Kernel Size = {u_sz_kern})"
    assert isinstance(
        u_sz_kern, int
    ), f"badKernelSize. Kernel must be an integer. (Kernel Size = {u_sz_kern})"

    # Standard Deviation
    assert (
        u_std_dev > sd_bounds[0]
    ), f"badStandardDeviation. Standard deviation must be > {sd_bounds[0]} and < {sd_bounds[1]}. (Standard Deviation = {u_std_dev})"
    assert (
        u_std_dev <= sd_bounds[1]
    ), f"badStandardDeviation. Standard deviation must be > 0 and < 10. (Standard Deviation = {u_std_dev})"

    # Pixel Intensity
    assert (
        u_fast_thr >= fast_bounds[0]
    ), f"badFASTThreshold. Threshold must be >= {fast_bounds[0]} and <= {fast_bounds[1]}. (Threshold = {u_fast_thr})"
    assert (
        u_fast_thr <= fast_bounds[1]
    ), f"badFASTThreshold. Threshold must be >= {fast_bounds[0]} and <= {fast_bounds[1]}. (Threshold = {u_fast_thr})"
    assert isinstance(
        u_fast_thr, int
    ), f"badFASTThreshold. Threshold must be an integer. (Kernel Size = {u_fast_thr})"

    # Bin Size
    assert (
        u_bin_sz >= bin_bounds[0]
    ), f"badBinSize. Bin Size must be >= {bin_bounds[0]} and <= {bin_bounds[1]}. (Bin size = {u_bin_sz})"
    assert (
        u_bin_sz <= bin_bounds[1]
    ), f"badBinSize. Bin Size must be = {bin_bounds[0]} and <= {bin_bounds[1]}. (Bin size = {u_bin_sz})"
    assert isinstance(
        u_bin_sz, int
    ), f"badBinSize. Bin Size must be an integer. (Bin Size = {u_bin_sz})"

    # Patch Size
    assert (
        u_patch_sz >= patch_sz[0]
    ), f"badPatchSize. Patch size must be >= {patch_sz[0]} and <= {patch_sz[1]}. (Patch size = {u_patch_sz})"
    assert (
        u_patch_sz <= patch_sz[1]
    ), f"badPatchSize. Patch size must be >= {patch_sz[0]} and <= {patch_sz[1]}. (Patch size = {u_patch_sz})"
    assert isinstance(
        u_patch_sz, int
    ), f"badPatchSize. Patch Size must be an integer. (Patch Size = {u_patch_sz})"

    # if u_bin_sz < bin_bounds[0] or u_bin_sz > bin_bounds[1]:
    #     err_count += 1
    #     err_list.append(
    #         "Error: badBinSize.")
    #         # "Error: User-defined feature descriptor bin size is invalid. Update the bin size to fall within the allowable bounds before rerunning the program.")
    #
    # if u_patch_sz < patch_sz[0] or u_patch_sz > patch_sz[1]:
    #     err_count += 1
    #     err_list.append(
    #         "Error: badPatchSize.")
    #         # "User-defined patch size is invalid. Update the patch size to fall within the allowable bounds before rerunning the program.")
    #
    # # print(err_count) # uncomment only to support debugging
    # if err_count == 0:
    #     # print("No errors detected in user-specified parameters.") # uncomment only to support debugging
    #     a = 0 # dummy line to avoid throwing an error
    # else:
    #     print("Total errors detected: ", err_count) # uncomment only to support debugging
    #     print(type(err_list)) # uncomment only to support debugging
    #     for i in err_list:
    #         print(i)
    #
    # return err_count, err_list


def verify_imported_image(img, img_path, img_id):
    if img is None:
        print(f"ReadImageError: {img_path} could not be read.")

    else:
        print("No detected errors with path for ", img_id)


def load_orb_descriptors(filename, directory):
    """
    Loads keypoints and descriptors from a CSV file and reconstructs ORB keypoints and descriptors.

    :param filename: Name of the CSV file to load (without extension).
    :param directory: Directory where the CSV file is stored.
    :return: ORB keypoints and descriptors in correct format for OpenCV matcher.
    """
    # Construct the full file path
    file_path = os.path.join(directory, filename + "_fd.csv")

    # Load the CSV into a DataFrame
    df = pd.read_csv(file_path)

    # Check if the necessary columns exist
    if not all(
        col in df.columns
        for col in [
            "x",
            "y",
            "size",
            "angle",
            "response",
            "octave",
            "class_id",
            "descriptor",
        ]
    ):
        print("CSV file doesn't have the required columns.")
        return None, None

    # Reconstruct keypoints from the CSV columns
    keypoints = []
    descriptor_list = []

    for _, row in df.iterrows():
        # Convert the descriptor string back to a NumPy array
        descriptor_bits = np.array(
            [int(b) for b in row["descriptor"]], dtype=np.uint8
        )  # Convert bit string to uint8
        descriptor_bytes = np.packbits(descriptor_bits)  # Repack bits into bytes
        descriptor_list.append(descriptor_bytes)  # Append as a single descriptor row

        # Create a keypoint for each row
        kp = cv.KeyPoint(
            row["x"],
            row["y"],
            row["size"],
            row["angle"],
            row["response"],
            int(row["octave"]),
            int(row["class_id"]),
        )
        keypoints.append(kp)

        # Stack descriptors into a single NumPy array if any descriptors exist
        descriptors = (
            np.vstack(descriptor_list).astype(np.uint8) if descriptor_list else None
        )

    # print(f"Data loaded and converted to ORB descriptors and keypoints from {file_path}")
    return keypoints, descriptors


# module testing code
check_limits(5, 1, 15, 50, 5)
