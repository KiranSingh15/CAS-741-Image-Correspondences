import sys
from pathlib import Path
import pandas as pd
import shutil
import pytest
import cv2 as cv

# Allow local helper import
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
import helper_functions as helper

# ControlModule setup
project_dir = current_dir.parent / "projectFiles"
sys.path.insert(0, str(project_dir))
import ControlModule as control
import InputFormatModule as config
import OutputFormatModule as output

lego_img_ID = ["LG_02.png", "LG_03.png"]
lego_swap_ID = ["LG_SWAP_02.png", "LG_SWAP_01.png"]

def load_descriptor_pairs_with_coords(match_folder, img_ids):
    """Load match descriptor and (x,y) coords as unordered sets."""
    desc_pairs = set()
    for csv_file in match_folder.glob("*.csv"):
        if not any(img_id.replace('.png', '') in csv_file.name for img_id in img_ids):
            continue
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            try:
                qx, qy = float(row["Query X"]), float(row["Query Y"])
                tx, ty = float(row["Train X"]), float(row["Train Y"])
                dq = row.get("Query Descriptor", "").strip()
                dt = row.get("Train Descriptor", "").strip()
                if dq and dt:
                    desc_pairs.add(frozenset([
                        (round(qx, 1), round(qy, 1), dq),
                        (round(tx, 1), round(ty, 1), dt)
                    ]))
            except Exception:
                continue
    return desc_pairs

def copy_match_outputs(source_dir, target_dir):
    """Copy all match CSVs and PNGs to archive output."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for file in source_dir.glob("*"):
        if file.suffix in [".csv", ".png"]:
            shutil.copy(file, target_dir)

@pytest.mark.repeatability
def test_repeatability_controlmodule_output():
    """
    STNFR-RE-1: ControlModule output repeatability check for descriptors + coordinates.
    Match outputs (CSVs + PNGs) are also archived.
    """
    test_id = "STNFR-RE-1"
    raw_dir = project_dir / "Raw_Images"
    out_dir = project_dir / "Outputs"

    # LEGO phase
    lego_input = current_dir / "testImages" / "lego"
    helper.clear_directory(raw_dir)
    helper.clear_directory(out_dir)
    helper.copy_folder_contents(lego_input, raw_dir)
    config.get_head_directory = lambda: project_dir
    control.main()
    folder_a = out_dir / "fMatches"
    pairs_a = load_descriptor_pairs_with_coords(folder_a, lego_img_ID)

    archive_dir = helper.create_timestamped_output_dir(test_id)
    copy_match_outputs(folder_a, archive_dir / "lego" / "fMatches")

    # LEGO_SWAP phase
    lego_swap_input = current_dir / "testImages" / "lego_swap"
    helper.clear_directory(raw_dir)
    helper.clear_directory(out_dir)
    helper.copy_folder_contents(lego_swap_input, raw_dir)

    # Confirm all images are valid
    for name in lego_swap_ID:
        path = raw_dir / name
        print(f"[CHECK] {path}")
        assert path.exists(), f"Missing: {path}"
        img = cv.imread(str(path))
        assert img is not None and img.size > 0, f"Unreadable image: {path}"

    config.get_head_directory = lambda: project_dir
    control.main()
    folder_b = out_dir / "fMatches"
    pairs_b = load_descriptor_pairs_with_coords(folder_b, lego_swap_ID)

    copy_match_outputs(folder_b, archive_dir / "lego_swap" / "fMatches")

    # Compare
    unmatched_a = pairs_a - pairs_b
    unmatched_b = pairs_b - pairs_a

    summary_path = archive_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Repeatability Test Summary: {test_id}\n")
        f.write(f"Images from lego: {lego_img_ID}\n")
        f.write(f"Images from lego_swap: {lego_swap_ID}\n\n")
        f.write(f"Total matches in lego: {len(pairs_a)}\n")
        f.write(f"Total matches in lego_swap: {len(pairs_b)}\n")
        f.write(f"Unmatched in lego: {len(unmatched_a)}\n")
        f.write(f"Unmatched in lego_swap: {len(unmatched_b)}\n\n")

        if unmatched_a:
            f.write("[FAIL] Matches in lego not found in lego_swap:\n")
            for item in unmatched_a:
                f.write(f" - {item}\n")
        if unmatched_b:
            f.write("[FAIL] Matches in lego_swap not found in lego:\n")
            for item in unmatched_b:
                f.write(f" - {item}\n")

        if not unmatched_a and not unmatched_b:
            f.write("All descriptor + coordinate matches are repeatable.\n")

    assert not unmatched_a and not unmatched_b, "Mismatch in descriptor-coordinate matches across datasets"