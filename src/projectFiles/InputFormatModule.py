import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import SpecificationParametersModule as specParams


def get_head_directory():
    head_directory = Path(os.getcwd())
    return head_directory  # Convert to Path object


def check_method_limits(
    mthd_img_smoothing, mthd_kp_detection, mthd_kp_description, mthd_ft_match
):
    avail_mthd_count, avail_is, avail_kpd, avail_fd, avail_ftm = (
        specParams.get_available_methods()
    )

    # assess available methods
    # Image smoothing
    assert (
        mthd_img_smoothing <= avail_mthd_count[0]
    ), f"Method selection is out of scope. Number of available methods for image smoothing is {avail_mthd_count[0]}. (Input method = {mthd_img_smoothing})"
    assert (
        mthd_kp_detection <= avail_mthd_count[1]
    ), f"Method selection is out of scope. Number of available methods for keypoint detection is {avail_mthd_count[1]}. (Input method = {mthd_kp_detection})"
    assert (
        mthd_kp_description <= avail_mthd_count[2]
    ), f"Method selection is out of scope. Number of available methods for feature descriptors is {avail_mthd_count[2]}. (Input method = {mthd_kp_description})"
    assert (
        mthd_ft_match <= avail_mthd_count[3]
    ), f"Method selection is out of scope. Number of available methods for feature matching is {avail_mthd_count[3]}. (Input method = {mthd_img_smoothing})"


def get_active_methods():
    mthd_img_smoothing, mthd_kp_detection, mthd_kp_description, mthd_ft_match = (
        specParams.get_assigned_methods()
    )

    check_method_limits(
        mthd_img_smoothing, mthd_kp_detection, mthd_kp_description, mthd_ft_match
    )

    return mthd_img_smoothing, mthd_kp_detection, mthd_kp_description, mthd_ft_match


def check_parameter_limits(u_sz_kern, u_std_dev, u_fast_thr, u_bin_sz, u_patch_sz):
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


def get_chosen_parameters():
    k, sigma, t, b, p = specParams.get_assigned_parameters()
    check_parameter_limits(k, sigma, t, b, p)

    return k, sigma, t, b, p


def set_input_img_path(head_dir):
    local_folder = "Raw_Images"
    img_dir = head_dir / local_folder  # Use '/' for safe path joining
    return img_dir, local_folder


def get_img_IDs(head_dir):
    local_folder = "Raw_Images"
    img_dir = head_dir / local_folder  # Use '/' for safe path joining
    input_img = [
        (file.stem, file.suffix, file.name)
        for file in img_dir.iterdir()
        if file.is_file()
    ]
    num_images = len(input_img)
    return input_img, num_images


def verify_imported_image(img, img_path, img_id):
    if img is None:
        print(f"ReadImageError: {img_path} could not be read.")

    else:
        print("No detected errors with path for ", img_id)

def get_descriptor_path (head_dir, descriptor_folder_nm):
    descriptor_path = head_dir / "Outputs" / descriptor_folder_nm
    return descriptor_path

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

    return keypoints, descriptors


# module testing code
# check_method_limits(1, 2, 1, 2)
# check_parameter_limits(5, 1, 15, 50, 5)
