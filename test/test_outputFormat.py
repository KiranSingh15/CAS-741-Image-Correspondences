import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add 'src' to sys.path so 'projectFiles' becomes importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import projectFiles.OutputFormatModule as outmod


@pytest.mark.parametrize("num_points", [1, 100, 1000])
def test_output_keypoints_variable_size(tmp_path, num_points):
    image_id = f"img_kp_{num_points}"
    keypoints = [cv.KeyPoint(float(i), float(i + 1), 5) for i in range(num_points)]

    output_dir = tmp_path / "Outputs"
    outmod.output_keypoints(keypoints, image_id, output_dir, outmod.keypoint_folder_nm)

    output_file = output_dir / outmod.keypoint_folder_nm / f"{image_id}_kp.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert df.shape[0] == num_points
    assert list(df.columns) == [
        "x",
        "y",
        "size",
        "angle",
        "response",
        "octave",
        "class_id",
    ]


@pytest.mark.parametrize("num_points", [1, 100, 1000])
def test_output_descriptors_variable_size(tmp_path, num_points):
    image_id = f"img_{num_points}"
    keypoints = [cv.KeyPoint(float(i), float(i + 1), 5) for i in range(num_points)]
    descriptors = np.random.randint(0, 256, (num_points, 32), dtype=np.uint8)

    outmod.output_descriptors(
        keypoints, descriptors, image_id, tmp_path, outmod.descriptor_folder_nm
    )

    output_file = tmp_path / outmod.descriptor_folder_nm / f"{image_id}_fd.csv"
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
    matches = [
        cv.DMatch(_queryIdx=i, _trainIdx=i, _distance=np.random.rand() * 100)
        for i in range(num_matches)
    ]
    descriptors1 = np.random.randint(0, 256, (num_matches, 32), dtype=np.uint8)
    descriptors2 = np.random.randint(0, 256, (num_matches, 32), dtype=np.uint8)

    outmod.output_matches(
        query_id,
        train_id,
        matches,
        kp1,
        kp2,
        descriptors1,
        descriptors2,
        tmp_path,
        outmod.matches_folder_nm,
    )

    output_file = tmp_path / outmod.matches_folder_nm / f"{query_id}_{train_id}_fm.csv"

    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert df.shape[0] == num_matches
    assert all(
        col in df.columns
        for col in [
            "Query Index",
            "Train Index",
            "Distance",
            "Query X",
            "Query Y",
            "Train X",
            "Train Y",
            "Query Descriptor",
            "Train Descriptor",
            "Query Image ID",
            "Train Image ID",
        ]
    )
