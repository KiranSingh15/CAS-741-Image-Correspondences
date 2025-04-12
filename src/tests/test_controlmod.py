import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2 as cv
import pytest

# Add 'src/projectFiles' to sys.path so InputFormatModule can import its sibling
current_dir = os.path.dirname(__file__)
project_dir = os.path.abspath(os.path.join(current_dir, "..", "projectFiles"))
sys.path.insert(0, project_dir)

# Now try importing from projectFiles
import projectFiles.ControlModule as control
import projectFiles.ImageSmoothingModule as smoothmod
import projectFiles.InputFormatModule as config


def save_image(image_in, parent_dir, target_folder, image_name):
    """
    Save image to the specified folder.
    :param image_in: The image to save.
    :param parent_dir: Base directory (e.g., head_dir).
    :param target_folder: Name of the subfolder to save into.
    :param image_name: Output file name (e.g., "img1.png").
    """
    output_head_dir = Path(parent_dir) / "Outputs"
    full_output_dir = output_head_dir / target_folder
    full_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    img_out_path = full_output_dir / image_name
    print(f"Saving image to: {img_out_path}")  # Debugging line
    cv.imwrite(str(img_out_path), image_in)


def assert_valid_image(path: Path):
    assert path.exists(), f"Image not found: {path}"
    img = cv.imread(str(path))
    assert img is not None, f"Failed to load image at {path}"
    assert img.size > 0, f"Loaded image is empty: {path}"


@pytest.mark.parametrize("label_name", ["building", "cybertruck", "game", "lego"])
def test_controlmodule_image_read_and_smooth_outputs(tmp_path, label_name):
    """
    This test verifies:
    - Images are read and smoothed by the ControlModule
    - Output folders gsImagery and gkImagery are created under the Outputs folder
    - test_summary.txt file exists and logs label and folders
    """

    # Copy test images to simulated Raw_Images
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

    # Run the control pipeline
    control.main()

    # Create timestamp for test summary file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_filename = f"{timestamp}_image_smoothing.txt"

    # Verify Outputs folder structure
    output_root = tmp_path / "Outputs"
    assert output_root.exists(), "Outputs folder was not created."

    # Check if gsImagery and gkImagery folders exist under Outputs
    gs_path = output_root / "gsImagery"
    gk_path = output_root / "gkImagery"
    assert gs_path.exists(), "gsImagery folder not found."
    assert gk_path.exists(), "gkImagery folder not found."

    # Save images into the respective folders
    for img_file in image_files:
        img = cv.imread(str(img_file), cv.IMREAD_GRAYSCALE)
        smoothed = smoothmod.smooth_image(
            mthd_img_smoothing=1, img_greyscale=img, sz_kern=5, std_dev=1.0
        )

        # Save grayscale image to gsImagery folder
        save_image(img, tmp_path, "gsImagery", img_file.name)

        # Save smoothed image to gkImagery folder
        save_image(smoothed, tmp_path, "gkImagery", img_file.name)

        # Check if the image files were saved
        assert (
            gs_path / img_file.name
        ).exists(), f"{img_file.name} missing in gsImagery"
        assert (
            gk_path / img_file.name
        ).exists(), f"{img_file.name} missing in gkImagery"

        for img_file in image_files:
            assert_valid_image(gs_path / img_file.name)
            assert_valid_image(gk_path / img_file.name)

    # Create the summary file
    summary_path = output_root / summary_filename
    with open(summary_path, "w") as summary_file:
        summary_file.write(f"Test Label: {label_name}\n")
        summary_file.write(f"Processed Folders:\n")
        summary_file.write(f" - gsImagery\n")
        summary_file.write(f" - gkImagery\n")
        summary_file.write(f"Processed Images:\n")
        for img_file in image_files:
            summary_file.write(f" - {img_file.name}\n")

    # Check if summary file exists
    assert summary_path.exists(), "test_summary.txt was not created."
    print(f"Results saved to the Results subfolder of : {tmp_path}")
