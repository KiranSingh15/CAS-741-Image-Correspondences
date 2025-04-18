import os
from pathlib import Path

import cv2 as cv


# Draw keypoints; suitable for either unscaled keypoints or scaled keypoints for descriptors
def gen_kp_img(image_in, keypoints, in_flags):
    # in_flags = 0 for just keypoint pixel, or in_flags = cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS for descriptors
    img_keypoints = cv.drawKeypoints(
        image_in, keypoints, None, color=(0, 255, 0), flags=in_flags
    )
    return img_keypoints


# generate a plot of candidate matches between features
def gen_matched_features(
    image1, image2, keypoint1, keypoint2, matches, max_num_matches, dist_thresh=100
):
    # Filter matches based on the minimum distance
    [m for m in matches if m.distance < dist_thresh]

    img_matches = cv.drawMatches(
        image1, keypoint1, image2, keypoint2, matches[:max_num_matches], None, flags=2
    )
    return img_matches


def make_directory(head_dir, target_name):
    # Ensure target parent directory exists
    folder_path = Path(head_dir) / target_name
    os.makedirs(folder_path, exist_ok=True)


# output a generated image
def save_image(
    image_in, parent_dir, target_folder, image_name, use_outputs_folder=True
):
    """
    Save image to the specified folder.
    :param image_in: The image to save.
    :param parent_dir: Base directory (e.g., head_dir).
    :param target_folder: Name of the subfolder to save into.
    :param image_name: Output file name (e.g., "img1.png").
    :param use_outputs_folder: If True, image is saved under Outputs/target_folder. If False, saved under target_folder.
    """
    if use_outputs_folder:
        output_head_dir = Path(parent_dir)
    else:
        output_head_dir = Path(parent_dir) / "Outputs"

    full_output_dir = output_head_dir / target_folder
    full_output_dir.mkdir(parents=True, exist_ok=True)

    img_out_path = full_output_dir / image_name
    cv.imwrite(str(img_out_path), image_in)
