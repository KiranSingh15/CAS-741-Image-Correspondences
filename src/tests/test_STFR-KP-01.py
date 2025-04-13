import csv
import sys
from pathlib import Path

# Setup current directory and import helper functions
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
import helper_functions as helper


def test_keypoint_detection(threshold=60):
    test_id = "STFR-KP-01"

    # Clear directories
    raw_image_dir = current_dir.parent / "projectFiles" / "Raw_Images"
    output_dir = current_dir.parent / "projectFiles" / "Outputs"
    helper.clear_directory(raw_image_dir)
    helper.clear_directory(output_dir)

    # Load test images
    in_img_dir = current_dir / "testImages" / "aruco"
    helper.copy_folder_contents(in_img_dir, raw_image_dir)

    # Prepare project modules
    project_dir = current_dir.parent / "projectFiles"
    sys.path.insert(0, str(project_dir))
    import ControlModule as control
    import InputFormatModule as config

    # Patch parameter configuration for keypoint detection
    config.get_head_directory = lambda: project_dir
    config.get_specified_parameters = lambda: {
        "fast_thr": threshold,
        "mthd_kp_detection": 1,
    }

    # Run full pipeline
    control.main()

    # Validate CSV output (first two columns should contain integers)
    kp_path = output_dir / "kpDetection"
    invalid_entries = []

    for csv_file in kp_path.glob("*_kp.csv"):
        with open(csv_file, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row_idx, row in enumerate(reader, start=2):  # Skip header
                try:
                    x = float(row[0])
                    y = float(row[1])
                    if not x.is_integer() or not y.is_integer():
                        invalid_entries.append((csv_file.name, row_idx, row[0], row[1]))
                except (ValueError, IndexError):
                    invalid_entries.append((csv_file.name, row_idx, row[0], row[1]))

    # Archive results
    archive_dir = helper.create_timestamped_output_dir(test_id)
    helper.copy_selected_subfolders(output_dir, archive_dir, ["kpDetection"])

    # Write summary
    summary_file = archive_dir / "summary.txt"
    with open(summary_file, "a") as f:
        f.write("\n--- Keypoint Detection Validation ---\n")
        f.write(f"Pixel Intensity Threshold (t): {threshold}\n")
        f.write("Keypoint detection method: FAST\n\n")

        if invalid_entries:
            f.write("Some keypoint CSVs contained non-integer x or y values:\n")
            for entry in invalid_entries:
                f.write(
                    f" - {entry[0]} (line {entry[1]}): x={entry[2]}, y={entry[3]}\n"
                )
        else:
            f.write("All keypoint CSVs passed: x and y values are all integers.\n")

    print(f"Test {test_id} complete. Outputs archived to: {archive_dir}")
