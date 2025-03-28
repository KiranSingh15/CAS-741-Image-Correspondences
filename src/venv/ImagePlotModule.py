import cv2 as cv
import os

# parent_dir = ""
#
# def set_parent_dir(dir):
#     parent_dir = dir
#
# def get_parent_dir():
#     return parent_dir

# Drawing the keypoints
def gen_kp_img (image_in, keypoints, in_flags):
    # in_flags = 0 for just keypoint pixe, or in_flags = cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS for descriptors
    img_keypoints = cv.drawKeypoints(image_in, keypoints, None, color=(0, 255, 0), flags = in_flags)

    # cv.imshow('Unadjusted ORB', kp_image)
    return img_keypoints


def gen_matched_features (image1, image2, keypoint1, keypoint2, matches, max_num_matches, dist_thresh = 100):
    # Filter matches based on the minimum distance
    good_matches = [m for m in matches if m.distance < dist_thresh]

    img_matches = cv.drawMatches(image1, keypoint1, image2, keypoint2, matches[:max_num_matches], None,  flags = 2)
    return img_matches


def save_image(image_in, parent_dir, target_folder, image_name):
    img_out_path = parent_dir / target_folder / image_name
    # print(img_out_path, " is of type ", type(img_out_path))   # uncomment for debugging
    cv.imwrite(img_out_path, image_in)
