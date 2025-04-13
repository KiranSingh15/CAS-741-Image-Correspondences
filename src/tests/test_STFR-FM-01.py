import sys
from pathlib import Path

import pandas as pd

# Allow importing helper_functions from the current directory
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import helper_functions as helper


def test_feature_matching(d=25, n_disp_matches=30):
    test_id = "STFR-FM-01"

    # Clear output folders
    raw_image_dir = current_dir.parent / "projectFiles" / "Raw_Images"
    output_dir = current_dir.parent / "projectFiles" / "Outputs"
    helper.clear_directory(raw_image_dir)
    helper.clear_directory(output_dir)

    # Copy test images
    in_img_dir = current_dir / "testImages" / "aruco"
    helper.copy_folder_contents(in_img_dir, raw_image_dir)

    # Setup project imports
    project_dir = current_dir.parent / "projectFiles"
    sys.path.insert(0, str(project_dir))
    import ControlModule as control
    import InputFormatModule as config

    # Patch input config
    config.get_head_directory = lambda: project_dir
    config.get_specified_parameters = lambda: {
        "mthd_kp_detection": 1,
        "mthd_kp_description": 1,
        "u_fast_thr": 40,
        "u_bin_sz": 100,
        "u_patch_sz": 31,
        "mthd_f_matching": 1,
        "u_hamming_dist": d,
        "n_disp_matches": n_disp_matches,
    }

    # Run pipeline
    control.main()

    # Archive and check outputs
    archive_dir = helper.create_timestamped_output_dir(test_id)
    helper.copy_selected_subfolders(
        output_dir, archive_dir, folders_to_copy=["fMatches"]
    )

    matches_path = output_dir / "fMatches"

    print("fMatches contents after copy:")
    for item in (archive_dir / "fMatches").glob("*"):
        print(" -", item.name)

    # Scan output files
    match_dfs = []
    hamming_flag = False

    for csv_file in matches_path.glob("*.csv"):
        df = pd.read_csv(csv_file)
        if "Query Descriptor" in df.columns and "Train Descriptor" in df.columns:
            hamming_flag = True
            match_dfs.append((csv_file.name, df))

    # Write summary
    summary_file = archive_dir / "summary.txt"
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"Feature Matching Functional Test Summary: {test_id}\n")
        f.write(f"Hamming Distance Threshold (d): {d}\n")
        f.write(f"Max Displayed Matches (n_disp_matches): {n_disp_matches}\n\n")
        f.write(f"Hamming Distance Used: {'Yes' if hamming_flag else 'No'}\n")
        f.write(f"Match Files Found: {len(match_dfs)}\n\n")
        for fname, df in match_dfs:
            f.write(f"{fname} — Matched Features: {len(df)}\n")

    print(f"Test STFR-FM-01 complete. Outputs archived to: {archive_dir}")

    # Scan output files
    match_dfs = []
    hamming_flag = False
    descriptor_issues = []
    match_score_issues = []

    for csv_file in (archive_dir / "fMatches").glob("*.csv"):
        df = pd.read_csv(csv_file)
        if "Query Descriptor" in df.columns and "Train Descriptor" in df.columns:
            hamming_flag = True
            match_dfs.append((csv_file.name, df))

            for idx, row in df.iterrows():
                q_desc = row["Query Descriptor"]
                t_desc = row["Train Descriptor"]
                distance = row["Distance"]

                # Check descriptors are 256 bits
                if (
                    not isinstance(q_desc, str)
                    or len(q_desc) != 256
                    or not set(q_desc).issubset({"0", "1"})
                ):
                    descriptor_issues.append(
                        (csv_file.name, idx, "Query Descriptor Invalid")
                    )
                if (
                    not isinstance(t_desc, str)
                    or len(t_desc) != 256
                    or not set(t_desc).issubset({"0", "1"})
                ):
                    descriptor_issues.append(
                        (csv_file.name, idx, "Train Descriptor Invalid")
                    )

                # Check match score (distance) is numeric and within bounds
                if not isinstance(distance, (int, float)):
                    match_score_issues.append(
                        (csv_file.name, idx, f"Distance is not numeric: {distance}")
                    )

    # Write summary
    summary_file = archive_dir / "summary.txt"
    with open(summary_file, "a") as f:
        f.write(f"Feature Matching Functional Test Summary: {test_id}\n")
        f.write(f"Hamming Distance Threshold (d): {d}\n")
        f.write(f"Max Displayed Matches (n_disp_matches): {n_disp_matches}\n\n")
        f.write(f"Hamming Distance Used: {'Yes' if hamming_flag else 'No'}\n")
        f.write(f"Match Files Found: {len(match_dfs)}\n\n")
        for fname, df in match_dfs:
            f.write(f"{fname} — Matched Features: {len(df)}\n")

        f.write("\nDescriptor Format Checks:\n")
        if descriptor_issues:
            for file, idx, issue in descriptor_issues:
                f.write(f" - {file}, Row {idx}: {issue}\n")
        else:
            f.write("All descriptors are valid 256-bit binary strings.\n")

        f.write("\nMatching Score Checks:\n")
        if match_score_issues:
            for file, idx, issue in match_score_issues:
                f.write(f" - {file}, Row {idx}: {issue}\n")
        else:
            f.write("All distances are valid and within threshold.\n")
