import os
from pathlib import Path

import numpy as np
import pandas as pd

greyscale_folder_nm = "gsImagery"
smoothed_imagery_folder_nm = "gkImagery"
keypoint_folder_nm = "kpDetection"
descriptor_folder_nm = "fDescriptors"
matches_folder_nm = "fMatches"


def define_output_folders():
    greyscale_folder_nm = "gsImagery"
    smoothed_imagery_folder_nm = "gkImagery"
    keypoint_folder_nm = "kpDetection"
    descriptor_folder_nm = "fDescriptors"
    matches_folder_nm = "fMatches"

    return (
        greyscale_folder_nm,
        smoothed_imagery_folder_nm,
        keypoint_folder_nm,
        descriptor_folder_nm,
        matches_folder_nm,
    )


def make_directory(parent_dir, target_name):
    # Ensure target directory exists
    folder_path = Path(parent_dir) / target_name
    os.makedirs(folder_path, exist_ok=True)


def output_keypoints(keypoints, image_id, parent_dir, target_folder):
    # check to see if the keypoint folder exists, and create it if it does not
    output_head_dir = parent_dir / "Outputs"
    make_directory(output_head_dir, keypoint_folder_nm)

    # Generate a unique file name
    file_name = f"{image_id}_kp.csv"
    file_path = os.path.join(output_head_dir, target_folder, file_name)
    file_path = output_head_dir / target_folder / file_name

    # Convert keypoints to a list of tuples
    keypoint_list = [
        (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in keypoints
    ]

    # Create a DataFrame
    df = pd.DataFrame(
        keypoint_list,
        columns=["x", "y", "size", "angle", "response", "octave", "class_id"],
    )

    # Name the file
    # unique_file_name = "PoseID_" + str(pose_id) + "_" + "CameraID_" + str(camera_id) + "_" + "kp"
    # unique_file_name = parent_dir + image_id + "_kp" + ".csv"
    # full_file_name = unique_file_name + ".csv"

    # full_file_name = parent_dir + image_id + "_kp" + ".csv"
    # file_path = os.path.join(target_folder, full_file_name)
    df.to_csv(file_path, index=False)
    # print("Keypoints saved to keypoints.csv")


def output_descriptors(keypoints, descriptors, image_id, parent_dir, target_folder):
    """
    Saves keypoints and their associated descriptors to a CSV file.

    :param keypoints: List of keypoints detected in the image.
    :param descriptors: Descriptors associated with the keypoints (as numpy array).
    :param image_id: Identifier for the image (used in the CSV file name).
    :param target_folder: Directory where the CSV file will be saved.
    """

    # check to see if the descriptors folder exists, and create it if it does not
    output_head_dir = parent_dir / "Outputs"
    make_directory(output_head_dir, descriptor_folder_nm)
    file_name = f"{image_id}_fd.csv"
    # file_path = os.path.join(parent_dir, target_folder, file_name)
    file_path = output_head_dir / target_folder / file_name

    # unique_file_name = image_id + "_fd"
    # full_file_name = unique_file_name + ".csv"
    # file_path = os.path.join(target_folder, full_file_name)

    # Ensure descriptors is the second element in the tuple
    if isinstance(descriptors, tuple):
        descriptors = descriptors[
            1
        ]  # Access the second element of the tuple which contains descriptors

    if descriptors is None:
        print(f"No descriptors found for {image_id}. Skipping save.")
        return

    # Convert keypoints to a list of tuples (x, y, size, angle, response, octave, class_id)
    keypoint_list = [
        (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in keypoints
    ]

    # Ensure descriptors is a numpy array of the correct type
    if isinstance(descriptors, np.ndarray) and descriptors.dtype == np.uint8:
        # Convert descriptors to a list (convert binary descriptors to a string representation of bits)
        descriptor_list = [
            "".join([str(int(b)) for b in np.unpackbits(np.uint8(desc))])
            for desc in descriptors
        ]
    else:
        print(
            f"Invalid descriptor type. Expected np.uint8 array, got {type(descriptors)}"
        )
        return

    # Combine the keypoints and descriptors into a single list of tuples
    combined_data = [
        keypoint + (descriptor,)
        for keypoint, descriptor in zip(keypoint_list, descriptor_list)
    ]

    # Create a DataFrame
    df = pd.DataFrame(
        combined_data,
        columns=[
            "x",
            "y",
            "size",
            "angle",
            "response",
            "octave",
            "class_id",
            "descriptor",
        ],
    )

    # Save the DataFrame to CSV
    df.to_csv(file_path, index=False)

    # print(f"Keypoints and descriptors saved to {file_path}")


# def output_matches(matches, kp1, kp2, image_id, target_folder):
def output_matches(
    query_img_ID, train_imd_ID, matches, kp1, kp2, parent_dir, target_dir
):
    """
    Saves brute-force matching results to a CSV file using pandas.
    :param matches: List of cv2.DMatch objects containing feature matches.
    :param kp1: List of cv2.KeyPoint objects from the first image.
    :param kp2: List of cv2.KeyPoint objects from the second image.
    :param image_id: Identifier for the image pair (used in the CSV file name).
    :param target_folder: Directory where the CSV file will be saved.
    """

    # check to see if the matches folder exists, and create it if it does not
    output_head_dir = parent_dir / "Outputs"
    make_directory(output_head_dir, matches_folder_nm)
    file_name = f"{query_img_ID}_{train_imd_ID}_fm.csv"
    file_path = os.path.join(output_head_dir, matches_folder_nm, file_name)

    # Extract match data
    match_data = []
    for match in matches:
        q_idx = match.queryIdx
        t_idx = match.trainIdx
        q_kp = kp1[q_idx].pt  # (x, y)
        t_kp = kp2[t_idx].pt  # (x, y)

        # round match coordinates
        match_data.append(
            [
                q_idx,
                t_idx,
                match.distance,
                round(q_kp[0]),
                round(q_kp[1]),
                round(t_kp[0]),
                round(t_kp[1]),
            ]
        )

    # Create a DataFrame
    df = pd.DataFrame(
        match_data,
        columns=[
            "Query Index",
            "Train Index",
            "Distance",
            "Query X",
            "Query Y",
            "Train X",
            "Train Y",
        ],
    )

    # Save DataFrame to CSV
    df.to_csv(file_path, index=False)

    # print(f"Matches saved to {file_path}")
