import os
import shutil
import sys
from pathlib import Path

import pytest

# Add 'src/projectFiles' to sys.path so InputFormatModule can import its sibling
current_dir = os.path.dirname(__file__)
project_dir = os.path.abspath(os.path.join(current_dir, "..", "projectFiles"))
sys.path.insert(0, project_dir)

import projectFiles.ControlModule as control
import projectFiles.InputFormatModule as config


@pytest.mark.parametrize("label_name", ["building", "cybertruck", "game", "lego"])
def test_controlmodule_outputs_and_csv(tmp_path, label_name):
    """
    This test ensures that:
    - Images are saved in the correct output folders.
    - CSVs for keypoints, descriptors, and matches are created in their respective folders.
    """

    # Setup: Copy test images into Raw_Images
    test_images_dir = Path(__file__).parent / "testImages" / label_name
    raw_img_dir = tmp_path / "Raw_Images"
    raw_img_dir.mkdir(parents=True, exist_ok=True)

    for file in test_images_dir.glob("*"):
        if file.suffix.lower() in [".jpg", ".png"]:
            shutil.copy(file, raw_img_dir)

    image_files = list(raw_img_dir.glob("*"))
    if not image_files:
        pytest.skip(f"No images found for label: {label_name}")

    # Patch config methods to return simulated paths
    config.get_head_directory = lambda: tmp_path
    config.set_input_img_path = lambda head_dir: (raw_img_dir, "Raw_Images")
    config.get_img_IDs = lambda head_dir: (
        [[img.stem, img.suffix, img.name] for img in image_files],
        len(image_files),
    )

    # Run the ControlModule to process the images
    control.main()

    # Verify that images and CSVs are created in the correct folders
    output_root = tmp_path / "Outputs"
    assert output_root.exists(), "Outputs folder was not created."

    expected_folders = [
        "gsImagery",
        "gkImagery",
        "kpDetection",
        "fDescriptors",
        "fMatches",
    ]
    for folder in expected_folders:
        folder_path = output_root / folder
        assert folder_path.exists(), f"{folder} folder was not created."

    # Check that grayscale and smoothed images were saved
    gs_path = output_root / "gsImagery"
    gk_path = output_root / "gkImagery"
    for img_file in image_files:
        assert (
            gs_path / img_file.name
        ).exists(), f"{img_file.name} missing in gsImagery"
        assert (
            gk_path / img_file.name
        ).exists(), f"{img_file.name} missing in gkImagery"

    # Check that keypoint and descriptor CSVs were saved
    kp_path = output_root / "kpDetection"
    fd_path = output_root / "fDescriptors"
    for img_file in image_files:
        kp_csv = kp_path / f"{img_file.stem}_kp.csv"
        fd_csv = fd_path / f"{img_file.stem}_fd.csv"
        assert kp_csv.exists(), f"{kp_csv.name} not found in {kp_path}"
        assert fd_csv.exists(), f"{fd_csv.name} not found in {fd_path}"

    # Check match CSVs exist for all image pairs
    matches_path = output_root / "fMatches"
    for i, img1 in enumerate(image_files):
        for j, img2 in enumerate(image_files):
            if i == j:
                continue  # Skip matching image to itself

            match_csv = matches_path / f"{img1.stem}_{img2.stem}_fm.csv"
            assert match_csv.exists(), f"{match_csv.name} not found in {matches_path}"
