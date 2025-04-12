import sys
from pathlib import Path

import pandas as pd

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import helper_functions as helper


def test_descriptor_generation(bin_size=100, patch_size=31):
    test_id = "STFR-FD-01"

    # Clear relevant input/output folders
    raw_image_dir = current_dir.parent / "projectFiles" / "Raw_Images"
    output_dir = current_dir.parent / "projectFiles" / "Outputs"
    helper.clear_directory(raw_image_dir)
    helper.clear_directory(output_dir)

    # Import test images
    in_img_dir = current_dir / "testImages" / "lego"
    helper.copy_folder_contents(in_img_dir, raw_image_dir)

    # Set up path and import modules
    project_dir = current_dir.parent / "projectFiles"
    sys.path.insert(0, str(project_dir))
    import ControlModule as control
    import InputFormatModule as config

    config.get_head_directory = lambda: project_dir
    config.get_specified_parameters = lambda: {
        "u_bin_sz": bin_size,
        "u_patch_sz": patch_size,
        "mthd_kp_detection": 1,
        "mthd_kp_description": 1,
        "u_fast_thr": 40,
    }

    # Run pipeline
    control.main()

    # Archive and collect results
    archive_dir = helper.create_timestamped_output_dir(test_id)
    descriptor_path = output_dir / "fDescriptors"
    helper.copy_selected_subfolders(
        output_dir, archive_dir, folders_to_copy=["fDescriptors"]
    )

    # Check each descriptor CSV
    errors = []
    for csv_file in descriptor_path.glob("*_fd.csv"):
        try:
            df = pd.read_csv(csv_file)
            for idx, val in enumerate(df["descriptor"]):
                if (
                    not isinstance(val, str)
                    or len(val) != 256
                    or any(c not in "01" for c in val)
                ):
                    errors.append(
                        (csv_file.name, idx, "Invalid binary string in descriptor")
                    )
        except Exception as e:
            errors.append((csv_file.name, "file", f"Failed to load: {str(e)}"))

    # Log results to summary
    summary_path = archive_dir / "summary.txt"
    with open(summary_path, "a") as f:
        f.write("\n--- Descriptor Generation ---\n")
        f.write(f"Bin Size (b): {bin_size}\n")
        f.write(f"Patch Size (p): {patch_size}\n")
        f.write(
            f"Number of descriptor files checked: {len(list(descriptor_path.glob('*_fd.csv')))}\n"
        )
        f.write("Errors:\n")
        if errors:
            for e in errors:
                f.write(f" - {e}\n")
        else:
            f.write("No descriptor format issues found.\n")

    print(f"Test {test_id} complete. Archived to {archive_dir}")
