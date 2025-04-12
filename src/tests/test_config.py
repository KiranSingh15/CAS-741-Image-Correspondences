import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import pytest

# Add 'src/projectFiles' to sys.path so InputFormatModule can import its sibling
current_dir = os.path.dirname(__file__)
project_dir = os.path.abspath(os.path.join(current_dir, "..", "projectFiles"))
sys.path.insert(0, project_dir)
import projectFiles.InputFormatModule as config


"""
# Test that the upper and lower bounds that are used to constrain the user-defined 
# parameters themselves conform to the data type and value constraints 
"""
@pytest.mark.parametrize(
    "param_name, bounds, expected",
    [
        ("kern_bounds", config.kern_bounds, (3, 15)),
        ("sd_bounds", config.sd_bounds, (0, 10)),
        ("fast_bounds", config.fast_bounds, (2, 254)),
        ("bin_bounds", config.bin_bounds, (1, 2048)),
        ("patch_sz", config.patch_sz, (5, 100)),
        ("match_distance_limits", config.match_distance_limits, (0, 150)),
        ("num_match_disp", config.num_match_disp, (1, 1000)),
    ],
)
def test_bounds_are_valid(param_name, bounds, expected):
    # Ensure bounds are a list or tuple with exactly two items
    assert isinstance(bounds, (list, tuple)), f"{param_name} should be a list or tuple."
    assert len(bounds) == 2, f"{param_name} should have exactly 2 elements."

    # Ensure both elements are integers
    assert all(
        isinstance(b, int) for b in bounds
    ), f"{param_name} bounds must be integers."

    # Check actual values match the expected reference
    assert bounds[0] == expected[0], f"{param_name} lower bound incorrect."
    assert bounds[1] == expected[1], f"{param_name} upper bound incorrect."


"""# 
# Kernel size bounds should both be odd integers — e.g., 3 to 15.
# 
"""
def test_kern_bounds_are_odd():
    assert config.kern_bounds[0] % 2 == 1, "Lower kernel size bound must be odd."
    assert config.kern_bounds[1] % 2 == 1, "Upper kernel size bound must be odd."


"""
Test that the system can successfully identify the current root of the directory. This is important for portability 
between different computers and operating systems
"""
def test_dir_path():
    # Test get_head_directory()
    head_dir = config.get_head_directory()
    assert isinstance(
        head_dir, Path
    ), "get_head_directory() should return a Path object"


# Test the check of the method limits
# Each test gets one bad value, others are valid
"""
Ensure that each method of Image Smoothing, Keypoint Detection, Feature Description, and Feature Matching 
fall between the specified number of allowable methods, or are disabled
"""
@pytest.mark.parametrize(
    "img, kp, desc, match",
    [
        (-1, 2, 1, 1),  # img smoothing too low
        (3, 2, 1, 1),  # img smoothing too high (max is 2)
        (1, -1, 1, 1),  # keypoint detection too low
        (1, 4, 1, 1),  # keypoint detection too high (max is 3)
        (1, 2, -1, 1),  # feature descriptor too low
        (1, 2, 3, 1),  # feature descriptor too high (max is 2)
        (1, 2, 1, -1),  # feature matching too low
        (1, 2, 1, 3),  # feature matching too high (max is 2)
    ],
)
def test_check_method_limits_invalid(monkeypatch, img, kp, desc, match):
    # Mock specParams.get_available_methods() to return max values
    monkeypatch.setattr(
        config.specParams,
        "get_available_methods",
        lambda: ([2, 3, 2, 2], [], [], [], []),
    )

    # Should raise AssertionError for all of the above
    with pytest.raises(AssertionError):
        config.check_method_limits(img, kp, desc, match)


def test_check_method_limits_valid(monkeypatch):
    monkeypatch.setattr(
        config.specParams,
        "get_available_methods",
        lambda: ([2, 3, 2, 2], [], [], [], []),
    )

    # All inputs valid
    config.check_method_limits(1, 2, 1, 2)


# Test the internal check for the allowable Parameter Limits
@pytest.mark.parametrize(
    "kernel, stddev, threshold, binsize, patchsize, matchdist, matchdisp",
    [
        # Invalid kernel sizes:
        (2, 1.0, 100, 100, 20, 50, 100),  # too small (must be > 3)
        (16, 1.0, 100, 100, 20, 50, 100),  # too large (max is 15)
        (4, 1.0, 100, 100, 20, 50, 100),  # even kernel (must be odd)
        (5.5, 1.0, 100, 100, 20, 50, 100),  # float (must be int)
        ("7", 1.0, 100, 100, 20, 50, 100),  # string (must be int)
    ],
)
def test_check_parameter_limits_invalid_kernel(
    kernel, stddev, threshold, binsize, patchsize, matchdist, matchdisp
):
    with pytest.raises(AssertionError):
        config.check_parameter_limits(
            kernel, stddev, threshold, binsize, patchsize, matchdist, matchdisp
        )


def test_check_parameter_limits_valid():
    # All values within allowed ranges
    config.check_parameter_limits(
        5,  # kernel size (odd, within bounds)
        1.5,  # std dev (between 0 and 10)
        100,  # threshold (between 2 and 254)
        128,  # bin size (between 1 and 2048)
        20,  # patch size (between 5 and 100)
        100,  # match distance (>0 and <=150)
        50,  # match display count (between 1 and 1000)
    )


# Test set_input_img_path()
def test_set_input_img_path(tmp_path):
    img_dir, folder_name = config.set_input_img_path(tmp_path)

    assert img_dir == tmp_path / "Raw_Images"
    assert folder_name == "Raw_Images"
    assert isinstance(img_dir, Path)
    assert isinstance(folder_name, str)


# Test get_img_IDs()
def create_dummy_files(folder: Path, filenames: list):
    for fname in filenames:
        (folder / fname).touch()


def test_get_img_IDs_with_images(tmp_path):  # Test for non-empty folder with files
    # Setup directory structure: head_dir/Raw_Images/
    raw_dir = tmp_path / "Raw_Images"
    raw_dir.mkdir()

    # Create dummy image files
    filenames = ["img1.jpg", "img2.png", "img3.tif"]
    create_dummy_files(raw_dir, filenames)

    image_ids, count = config.get_img_IDs(tmp_path)

    assert count == 3
    assert all(len(entry) == 3 for entry in image_ids)  # stem, suffix, name
    assert {name for _, _, name in image_ids} == set(filenames)


def test_get_img_IDs_empty_directory(tmp_path):  # Test of empty folder
    raw_dir = tmp_path / "Raw_Images"
    raw_dir.mkdir()

    image_ids, count = config.get_img_IDs(tmp_path)

    assert count == 0
    assert image_ids == []


def test_get_img_IDs_ignores_directories(tmp_path):
    raw_dir = tmp_path / "Raw_Images"
    raw_dir.mkdir()

    # Add an image and a subdirectory
    create_dummy_files(raw_dir, ["img.jpg"])
    (raw_dir / "subdir").mkdir()

    image_ids, count = config.get_img_IDs(tmp_path)

    assert count == 1
    assert image_ids[0][2] == "img.jpg"


# Test verify_imported_image() method
def test_verify_imported_image_none(capsys):
    config.verify_imported_image(None, "fake/path.jpg", "img001")

    captured = capsys.readouterr()
    assert "ReadImageError: fake/path.jpg could not be read." in captured.out


def test_verify_imported_image_valid(capsys):
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)  # a valid image-like object
    config.verify_imported_image(dummy_img, "any/path.jpg", "img002")

    captured = capsys.readouterr()
    assert "No detected errors with path for  img002" in captured.out


# Test get_descriptor_path()
def test_get_descriptor_path(tmp_path):
    descriptor_folder_name = "ORB_Descriptors"
    result_path = config.get_descriptor_path(tmp_path, descriptor_folder_name)

    expected_path = tmp_path / "Outputs" / descriptor_folder_name

    assert result_path == expected_path
    assert isinstance(result_path, Path)


def test_load_orb_descriptors_valid(tmp_path):
    # Create the directory and file path
    directory = tmp_path
    filename = "test_image"
    file_path = directory / f"{filename}_fd.csv"

    # Bit string for one descriptor: 32 bits (ORBs are 256 bits = 32 bytes)
    descriptor_bits = "0" * 256  # all zeros

    # Create dummy data for one keypoint
    df = pd.DataFrame(
        [
            {
                "x": 10.0,
                "y": 20.0,
                "size": 31.0,
                "angle": 45.0,
                "response": 0.8,
                "octave": 0,
                "class_id": 1,
                "descriptor": str(descriptor_bits),
            }
        ]
    )

    # Explicitly cast the column to string before saving
    df["descriptor"] = df["descriptor"].astype(str)

    df.to_csv(file_path, index=False)

    keypoints, descriptors = config.load_orb_descriptors(filename, str(directory))

    # Check types and content
    assert isinstance(keypoints, list)
    assert isinstance(keypoints[0], cv.KeyPoint)
    assert isinstance(descriptors, np.ndarray)
    assert descriptors.shape == (1, 32)
    assert descriptors.dtype == np.uint8


def test_load_orb_descriptors_missing_columns(tmp_path):
    directory = tmp_path
    filename = "missing_cols"
    file_path = directory / f"{filename}_fd.csv"

    # Missing 'descriptor' column
    df = pd.DataFrame(
        [
            {
                "x": 10.0,
                "y": 20.0,
                "size": 31.0,
                "angle": 45.0,
                "response": 0.8,
                "octave": 0,
                "class_id": 1,
            }
        ]
    )
    df.to_csv(file_path, index=False)

    keypoints, descriptors = config.load_orb_descriptors(filename, str(directory))

    assert keypoints is None
    assert descriptors is None


def test_orb_descriptor_roundtrip(tmp_path):
    # Create a dummy grayscale image
    img = np.zeros((100, 100), dtype=np.uint8)
    cv.circle(img, (50, 50), 10, 255, -1)  # Add a feature

    # Step 1: Detect ORB keypoints and descriptors
    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)

    assert descriptors is not None and len(keypoints) > 0

    # Step 2: Prepare DataFrame for saving
    records = []
    for kp, desc in zip(keypoints, descriptors):
        desc_bits = np.unpackbits(desc).astype(str)
        desc_str = "".join(desc_bits)
        records.append(
            {
                "x": kp.pt[0],
                "y": kp.pt[1],
                "size": kp.size,
                "angle": kp.angle,
                "response": kp.response,
                "octave": kp.octave,
                "class_id": kp.class_id,
                "descriptor": desc_str,
            }
        )

    df = pd.DataFrame.from_records(records)

    # Step 3: Save to CSV in expected format
    filename = "roundtrip_test"
    csv_path = tmp_path / f"{filename}_fd.csv"
    df.to_csv(csv_path, index=False)

    # Step 4: Load it back
    kps_loaded, desc_loaded = config.load_orb_descriptors(filename, str(tmp_path))

    assert kps_loaded is not None
    assert desc_loaded is not None
    assert len(kps_loaded) == len(keypoints)
    assert desc_loaded.shape == descriptors.shape
    assert desc_loaded.dtype == np.uint8
