import os
from pathlib import Path

import cv2 as cv


# Drawing the keypoints
def gen_kp_img(image_in, keypoints, in_flags):
    # in_flags = 0 for just keypoint pixel, or in_flags = cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS for descriptors
    img_keypoints = cv.drawKeypoints(
        image_in, keypoints, None, color=(0, 255, 0), flags=in_flags
    )
    return img_keypoints


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


def save_image(image_in, parent_dir, target_folder, image_name):
    output_head_dir = parent_dir / "Outputs"
    make_directory(output_head_dir, target_folder)
    img_out_path = output_head_dir / target_folder / image_name
    cv.imwrite(img_out_path, image_in)
