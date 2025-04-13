import sys
from pathlib import Path

import cv2 as cv

# Allow importing helper_functions
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import helper_functions as helper

"""
Kernel size and standard deviation represent the user-configurable parameters for 
the smoothing module, which uses Gaussian Blurring by default.

We use default parameters
"""


def test_image_smoothing(kernel_size=5, std_dev=1.0):
    # Parameters for smoothing

    # Clear project output directories
    raw_image_dir = current_dir.parent / "projectFiles" / "Raw_Images"
    output_dir = current_dir.parent / "projectFiles" / "Outputs"
    helper.clear_directory(raw_image_dir)
    helper.clear_directory(output_dir)

    # Copy test images
    in_img_dir = current_dir / "testImages" / "aruco"
    helper.copy_folder_contents(in_img_dir, raw_image_dir)

    # Prepare to run pipeline
    project_dir = current_dir.parent / "projectFiles"
    sys.path.insert(0, str(project_dir))
    import ControlModule as control
    import InputFormatModule as config

    # Patch the config
    config.get_head_directory = lambda: project_dir
    config.get_specified_parameters = lambda: {
        "sz_kern": kernel_size,
        "std_dev": std_dev,
        "mthd_img_smoothing": 1,
    }

    # Run the pipeline
    control.main()

    # Check dimensions and dtype of output images
    gs_path = output_dir / "gsImagery"
    gk_path = output_dir / "gkImagery"

    gs_errors, gk_errors = [], []
    for img_file in raw_image_dir.glob("*"):
        input_img = cv.imread(str(img_file), cv.IMREAD_GRAYSCALE)
        input_shape = input_img.shape
        input_dtype = input_img.dtype

        gs_img_path = gs_path / img_file.name
        gk_img_path = gk_path / img_file.name

        gs_img = cv.imread(str(gs_img_path), cv.IMREAD_GRAYSCALE)
        gk_img = cv.imread(str(gk_img_path), cv.IMREAD_GRAYSCALE)

        if gs_img.shape != input_shape:
            gs_errors.append(
                (img_file.name, f"Shape mismatch: {gs_img.shape} vs {input_shape}")
            )
        if gs_img.dtype != input_dtype:
            gs_errors.append(
                (img_file.name, f"Dtype mismatch: {gs_img.dtype} vs {input_dtype}")
            )

        if gk_img.shape != input_shape:
            gk_errors.append(
                (img_file.name, f"Shape mismatch: {gk_img.shape} vs {input_shape}")
            )
        if gk_img.dtype != input_dtype:
            gk_errors.append(
                (img_file.name, f"Dtype mismatch: {gk_img.dtype} vs {input_dtype}")
            )

    # Archive results
    archive_dir = helper.create_timestamped_output_dir("STFR-IS-01")
    helper.copy_selected_subfolders(output_dir, archive_dir, ["gsImagery", "gkImagery"])

    # Append summary
    summary_file = archive_dir / "summary.txt"
    with open(summary_file, "a") as f:
        f.write(f"\n--- Image Validation ---\n")
        f.write(f"Kernel Size (k): {kernel_size}\n")
        f.write(f"Standard Deviation (sigma): {std_dev}\n")
        f.write("\n")
        f.write(helper.summarize_image_check_results(gs_errors, "grayscale"))
        f.write("\n")
        f.write(helper.summarize_image_check_results(gk_errors, "smoothed"))

    print(f"Test STFR-IS-01 complete. Outputs archived to: {archive_dir}")
