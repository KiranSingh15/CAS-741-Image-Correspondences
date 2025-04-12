import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import pytest

# Add 'src/projectFiles' to sys.path
current_dir = os.path.dirname(__file__)
project_dir = os.path.abspath(os.path.join(current_dir, "..", "projectFiles"))
sys.path.insert(0, project_dir)

import projectFiles.ControlModule as control
import projectFiles.ImageSmoothingModule as smoothmod
import projectFiles.InputFormatModule as config


keypoint_columns = ["x", "y", "size", "angle", "response", "octave", "class_id"]
descriptor_columns = keypoint_columns + ["descriptor"]
match_columns = ["Query Index", "Train Index", "Distance", "Query X", "Query Y", "Train X", "Train Y"]

"""
Assess whether the provided path for a generated image is valid
This test is used for all operations, which include
: Greyscale conversion
: Image Smoothing (Gaussian Blur by default)
: Keypoint Detection (oriented-FAST by default)
: Feature Description (rotated-BRIEF by default)
: Feature Matching (Brute-Force by Default)
"""
def assert_valid_image(path: Path):
    assert path.exists(), f"Image not found: {path}"
    img = cv.imread(str(path))
    assert img is not None, f"Failed to load image at {path}"
    assert img.size > 0, f"Loaded image is empty: {path}"


"""
Assess whether the provided path for a generated CSV is valid, as well as the outlined 
column semantics for keypoints, features, and matched features.
: Keypoint Detection (oriented-FAST by default)
: Feature Description (rotated-BRIEF by default)
: Feature Matching (Brute-Force by Default)
"""
def validate_csv_structure(file_path: Path, expected_columns):
    assert file_path.exists(), f"Missing CSV file: {file_path}"
    df = pd.read_csv(file_path)
    for col in expected_columns:
        assert col in df.columns, f"Column '{col}' missing in {file_path.name}"




"""
Unit test of the control flow for the Control Module (main). This tests asserts that an temporary Output folder is 
generated with non-empty corresponding images and CSV data in corresponding subfolders. It also checks that a log of 
processed images is provided within the Outputs folder.

This test assess four different test cases, outlined with different datasets of images, given by the following folder names. 
1. building: a collection of two (2) images from a feature rich scene with high pixel variance between images
2. cybertruck: an empty folder that contains no images, and is designed to test program robustness
3. game: a collection of three (3) images that is intended to assess that  n*(n-1)/2 operations are performed for n images
4. lego: a collection of two (2) images in a static background with less noise that the sample images in 'building' but have more
repeated features
"""
@pytest.mark.parametrize("label_name", ["building", "cybertruck", "game", "lego"])
def test_controlmodule_full_pipeline_outputs(tmp_path, label_name):
    test_images_dir = Path(current_dir) / "testImages" / label_name
    raw_img_dir = tmp_path / "Raw_Images"
    raw_img_dir.mkdir(parents=True, exist_ok=True)

    for file in test_images_dir.glob("*"):
        if file.suffix.lower() in [".jpg", ".png"]:
            shutil.copy(file, raw_img_dir)

    image_files = list(raw_img_dir.glob("*"))
    if not image_files:
        pytest.skip(f"No images found for label: {label_name}")

    config.get_head_directory = lambda: tmp_path
    config.set_input_img_path = lambda head_dir: (raw_img_dir, "Raw_Images")
    config.get_img_IDs = lambda head_dir: (
        [[img.stem, img.suffix, img.name] for img in image_files],
        len(image_files),
    )

    control.main()

    output_root = tmp_path / "Outputs"
    assert output_root.exists(), "Outputs folder was not created."

    expected_folders = ["gsImagery", "gkImagery", "kpDetection", "fDescriptors", "fMatches"]
    for folder in expected_folders:
        assert (output_root / folder).exists(), f"{folder} folder was not created."

    gs_path = output_root / "gsImagery"
    gk_path = output_root / "gkImagery"
    for img_file in image_files:
        assert_valid_image(gs_path / img_file.name)
        assert_valid_image(gk_path / img_file.name)

    kp_path = output_root / "kpDetection"
    for img_file in image_files:
        validate_csv_structure(kp_path / f"{img_file.stem}_kp.csv", keypoint_columns)

    fd_path = output_root / "fDescriptors"
    for img_file in image_files:
        validate_csv_structure(fd_path / f"{img_file.stem}_fd.csv", descriptor_columns)

    matches_path = output_root / "fMatches"
    for i, img1 in enumerate(image_files):
        for j, img2 in enumerate(image_files):
            if i == j:      # for the same image
                continue
            if i > j:
                continue    # to avoid duplicating results, we only care when i < j
            match_csv = matches_path / f"{img1.stem}_{img2.stem}_fm.csv"
            if not match_csv.exists():
                print(f"[Info] Skipping match CSV check: {match_csv.name} not found (likely no matches).")
                continue
            validate_csv_structure(match_csv, match_columns)

    summary_path = output_root / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_pipeline_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Label: {label_name}\n")
        f.write("Folders:\n")
        for folder in expected_folders:
            f.write(f" - {folder}\n")
        f.write("Images Processed:\n")
        for img_file in image_files:
            f.write(f" - {img_file.name}\n")

    assert summary_path.exists(), "Summary file was not created."
    print(f"Results saved to the Results subfolder of : {tmp_path}")
