import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

# Add 'src/projectFiles' to sys.path so InputFormatModule can import its sibling
# Ensure src/ is in the Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import projectFiles.ImagePlotModule as plotmod

# Test keypoint generation on images

""" Tests to ensure
- The image is returned
- The shape is preserved
- It works even when no keypoints are passed
"""


def test_gen_kp_img_with_none_keypoints():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = plotmod.gen_kp_img(img, None, 0)

    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape


def test_gen_kp_img_no_flag():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    kps = [cv.KeyPoint(10, 10, 5)]
    result = plotmod.gen_kp_img(img, kps, in_flags=0)
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape


def test_gen_kp_img_rich_keypoints():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    kps = [cv.KeyPoint(30, 30, 10)]
    result = plotmod.gen_kp_img(
        img, kps, in_flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape


def test_gen_kp_img_with_none_image():
    with pytest.raises(cv.error):
        plotmod.gen_kp_img(None, [], 0)


# Test match generation
def test_gen_matched_features_success():
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    kp1 = [cv.KeyPoint(10, 10, 5)]
    kp2 = [cv.KeyPoint(20, 20, 5)]
    match = cv.DMatch(_queryIdx=0, _trainIdx=0, _distance=25)
    matches = [match]
    result = plotmod.gen_matched_features(
        img1, img2, kp1, kp2, matches, max_num_matches=1
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] > 0  # height


def test_gen_matched_features_with_empty_matches():
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    kp1 = [cv.KeyPoint(10, 10, 5)]
    kp2 = [cv.KeyPoint(20, 20, 5)]
    matches = []  # No matches

    result = plotmod.gen_matched_features(img1, img2, kp1, kp2, matches, 1)
    assert isinstance(result, np.ndarray)  # Still returns a blank composite image


def test_make_directory_creates(tmp_path):
    folder_name = "plots"
    plotmod.make_directory(tmp_path, folder_name)
    assert (tmp_path / folder_name).exists()


def test_save_image_success(tmp_path):
    img = np.zeros((50, 50, 3), dtype=np.uint8)

    # Direct save using the given tmp_path
    plotmod.save_image(img, tmp_path, "figures", "test_image.jpg")

    # Since save_image creates target_folder directly under tmp_path, no Outputs/ subdir
    output_path = tmp_path / "figures" / "test_image.jpg"

    assert output_path.exists(), f"Expected file not found: {output_path}"
    assert output_path.stat().st_size > 0, f"Output file is empty: {output_path}"
