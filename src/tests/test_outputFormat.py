import os
import sys
from pathlib import Path
import pytest
import cv2 as cv
import numpy as np
import pandas as pd


# Add 'src/projectFiles' to sys.path so InputFormatModule can import its sibling
current_dir = os.path.dirname(__file__)
project_dir = os.path.abspath(os.path.join(current_dir, '..', 'projectFiles'))
sys.path.insert(0, project_dir)
import projectFiles.OutputFormatModule as outmod
# Test make_directory()
def test_make_directory_creates_folder(tmp_path):
    folder_name = "test_folder"
    outmod.make_directory(tmp_path, folder_name)

    expected_path = tmp_path / folder_name
    assert expected_path.exists()
    assert expected_path.is_dir()



# Test output_keypoints()
@pytest.mark.parametrize("num_points", [1, 100, 1000])
def test_output_keypoints_variable_size(tmp_path, num_points):
    image_id = f"img_kp_{num_points}"
    keypoints = [cv.KeyPoint(float(i), float(i + 1), 5) for i in range(num_points)]

    outmod.output_keypoints(keypoints, image_id, tmp_path, outmod.keypoint_folder_nm)

    output_file = tmp_path / "Outputs" / outmod.keypoint_folder_nm / f"{image_id}_kp.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert df.shape[0] == num_points
    assert list(df.columns) == ["x", "y", "size", "angle", "response", "octave", "class_id"]


# Test output_descriptors()
@pytest.mark.parametrize("num_points", [1, 100, 1000])
def test_output_descriptors_variable_size(tmp_path, num_points):
    image_id = f"img_{num_points}"
    keypoints = [cv.KeyPoint(float(i), float(i + 1), 5)for i in range(num_points)]
    descriptors = np.random.randint(0, 256, (num_points, 32), dtype=np.uint8)

    outmod.output_descriptors(keypoints, descriptors, image_id, tmp_path, outmod.descriptor_folder_nm)

    output_file = tmp_path / "Outputs" / outmod.descriptor_folder_nm / f"{image_id}_fd.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert df.shape[0] == num_points
    assert "descriptor" in df.columns


@pytest.mark.parametrize("num_matches", [1, 100, 1000])
def test_output_matches_variable_size(tmp_path, num_matches):
    query_id = "imgQ"
    train_id = "imgT"

    kp1 = [cv.KeyPoint(float(i), float(i + 1), 5) for i in range(num_matches)]
    kp2 = [cv.KeyPoint(float(i + 2), float(i + 3), 5) for i in range(num_matches)]
    matches = [cv.DMatch(_queryIdx=i, _trainIdx=i, _distance=np.random.rand() * 100) for i in range(num_matches)]

    outmod.output_matches(query_id, train_id, matches, kp1, kp2, tmp_path, outmod.matches_folder_nm)

    output_file = tmp_path / "Outputs" / outmod.matches_folder_nm / f"{query_id}_{train_id}_fm.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert df.shape[0] == num_matches
    assert list(df.columns) == [
        "Query Index", "Train Index", "Distance",
        "Query X", "Query Y", "Train X", "Train Y"
    ]

