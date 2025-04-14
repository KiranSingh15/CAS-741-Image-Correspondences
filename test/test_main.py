import os
import sys
import cv2 as cv
import numpy as np
import pytest
from pathlib import Path

# Set up import path to projectFiles
current_dir = Path(__file__).parent
project_dir = (current_dir / ".." / "src" / "projectFiles").resolve()
sys.path.insert(0, str(project_dir))

import ControlModule as control
import InputFormatModule as config

def create_synthetic_image(output_path, filename):
    """Creates a synthetic image with distinguishable features."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv.circle(img, (50, 50), 20, (255, 255, 255), -1)
    cv.rectangle(img, (10, 10), (30, 30), (255, 255, 255), -1)
    cv.imwrite(str(output_path / filename), img)

@pytest.mark.parametrize("img_names", [["synthetic_01.png", "synthetic_02.png"]])
def test_pipeline_synthetic_images(tmp_path, img_names):
    """
    This test runs the full ControlModule pipeline using synthetic images.
    It verifies that:
    - Grayscale images are produced
    - Keypoint, descriptor, and match CSVs are generated
    """

    # Setup folders
    raw_img_dir = tmp_path / "Raw_Images"
    raw_img_dir.mkdir()
    output_dir = tmp_path / "Outputs"

    # Generate synthetic images
    for name in img_names:
        create_synthetic_image(raw_img_dir, name)

    # Patch config
    config.get_head_directory = lambda: tmp_path
    config.get_specified_parameters = lambda: {
        "sz_kern": 5,
        "std_dev": 1.0,
        "mthd_img_smoothing": 1,
        "mthd_kp_detection": 1,
        "mthd_kp_description": 1,
        "u_fast_thr": 40,
        "u_bin_sz": 500,
        "u_patch_sz": 31,
        "mthd_f_matching": 1,
        "u_hamming_dist": 30,
        "n_disp_matches": 20,
    }

    # Run pipeline
    control.main()

    # Prepare output folders
    gs_dir = output_dir / "gsImagery"
    gk_dir = output_dir / "gkImagery"
    kp_dir = output_dir / "kpDetection"
    fd_dir = output_dir / "fDescriptors"
    fm_dir = output_dir / "fMatches"

    results = {}

    # Grayscale and Smoothed Image Checks
    for name in img_names:
        results[f"Grayscale {name}"] = (gs_dir / name).exists()
        results[f"Smoothed {name}"] = (gk_dir / name).exists()

    # Keypoint CSV and Image Checks
    for name in img_names:
        stem = Path(name).stem
        results[f"Keypoint CSV {name}"] = (kp_dir / f"{stem}_kp.csv").exists()
        results[f"Keypoint IMG {name}"] = (kp_dir / name).exists()

    # Descriptor CSV and Image Checks
    for name in img_names:
        stem = Path(name).stem
        results[f"Descriptor CSV {name}"] = (fd_dir / f"{stem}_fd.csv").exists()
        results[f"Descriptor IMG {name}"] = (fd_dir / name).exists()

    # Match CSV and Image Checks
    img1, img2 = Path(img_names[0]).stem, Path(img_names[1]).stem
    results[f"Match CSV {img1}_{img2}"] = (fm_dir / f"{img1}_{img2}_fm.csv").exists()
    results[f"Match IMG {img1}_{img2}"] = (fm_dir / f"{img1}_{img2}.png").exists()

    # Assert all checks
    for item, status in results.items():
        assert status, f"{item} is missing."

    # # Generate summary log
    # summary_file = output_dir / "test_summary.txt"
    # with open(summary_file, "w", encoding="utf-8") as f:
    #     f.write("Functional Test Summary: Synthetic Pipeline Check\n")
    #     f.write(f"Image Inputs: {', '.join(img_names)}\n\n")
    #     f.write("Check Results:\n")
    #     for item, status in results.items():
    #         icon = "Pass" if status else "Fail"
    #         f.write(f"{icon} {item}\n")
