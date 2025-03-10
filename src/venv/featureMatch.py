import cv2 as cv
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# create BFMatcher object
def create_brute_force_matcher():
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    return bf

def load_orb_descriptors(filename, directory):
    """
    Loads keypoints and descriptors from a CSV file and reconstructs ORB keypoints and descriptors.

    :param filename: Name of the CSV file to load (without extension).
    :param directory: Directory where the CSV file is stored.
    :return: ORB keypoints and descriptors in correct format for OpenCV matcher.
    """
    # Construct the full file path
    file_path = os.path.join(directory, filename + "_fd.csv")

    # Load the CSV into a DataFrame
    df = pd.read_csv(file_path)

    # Check if the necessary columns exist
    if not all(col in df.columns for col in ["x", "y", "size", "angle", "response", "octave", "class_id", "descriptor"]):
        print("CSV file doesn't have the required columns.")
        return None, None

    # Reconstruct keypoints from the CSV columns
    keypoints = []
    descriptor_list = []

    for _, row in df.iterrows():
        # Convert the descriptor string back to a NumPy array
        descriptor_bits = np.array([int(b) for b in row['descriptor']], dtype=np.uint8)  # Convert bit string to uint8
        descriptor_bytes = np.packbits(descriptor_bits)  # Repack bits into bytes
        descriptor_list.append(descriptor_bytes)  # Append as a single descriptor row

        # Create a keypoint for each row
        kp = cv.KeyPoint(row['x'], row['y'], row['size'], row['angle'],
                          row['response'], int(row['octave']), int(row['class_id']))
        keypoints.append(kp)

        # Stack descriptors into a single NumPy array if any descriptors exist
        descriptors = np.vstack(descriptor_list).astype(np.uint8) if descriptor_list else None

    # print(f"Data loaded and converted to ORB descriptors and keypoints from {file_path}")
    return keypoints, descriptors


# Match descriptors.
def match_descriptors(brute_force_matcher, des1, des2):
    matches = brute_force_matcher.match(des1, des2)
    return matches

# Sort them in the order of their distance.
def sort_matches(matches):
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


# function displaying the output image with the feature matching
def display_output(pic1, kpt1, pic2, kpt2, best_match):
    # drawing the feature matches using drawMatches() function
    output_image = cv.drawMatches(pic1, kpt1, pic2, kpt2, best_match, None, flags=2)
    # Naming a window
    cv.namedWindow('Output image', cv.WINDOW_NORMAL)

    # Using resizeWindow()
    cv.resizeWindow('Output image', 800, 800)
    cv.imshow('Output image', output_image)

# alternative display function, where matches must be equal less than the allowable hamming distance
def display_screened_matches(pic1, kpt1, pic2, kpt2, matches, dist_thresh):
    # Filter matches based on the minimum distance
    good_matches = [m for m in matches if m.distance < dist_thresh]

    # drawing the feature matches using drawMatches() function
    output_image = cv.drawMatches(pic1, kpt1, pic2, kpt2, matches, None, flags=2)
    # Naming a window
    cv.namedWindow('Output image', cv.WINDOW_NORMAL)

    # Using resizeWindow()
    cv.resizeWindow('Output image', 800, 800)
    cv.imshow('Output image', output_image)

# Draw first 10 matches.
def draw_matches(img1, kp1, img2, kp2, matches, max_disp_matches):
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:max_disp_matches], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()

def save_matches_to_csv(matches, kp1, kp2, image_identifier, target_folder):
    """
    Saves brute-force matching results to a CSV file using pandas.

    :param matches: List of cv2.DMatch objects containing feature matches.
    :param kp1: List of cv2.KeyPoint objects from the first image.
    :param kp2: List of cv2.KeyPoint objects from the second image.
    :param image_identifier: Identifier for the image pair (used in the CSV file name).
    :param target_folder: Directory where the CSV file will be saved.
    """
    # Extract match data
    match_data = []
    for match in matches:
        q_idx = match.queryIdx
        t_idx = match.trainIdx
        q_kp = kp1[q_idx].pt  # (x, y)
        t_kp = kp2[t_idx].pt  # (x, y)

        ## unrounded match coordinates
        # match_data.append([q_idx, t_idx, match.distance,
        #                    q_kp[0], q_kp[1], t_kp[0], t_kp[1]])

        # round match coordinates
        match_data.append([q_idx, t_idx, match.distance,
                           round(q_kp[0]), round(q_kp[1]),
                           round(t_kp[0]), round(t_kp[1])])

    # Create a DataFrame
    df = pd.DataFrame(match_data, columns=["Query Index", "Train Index", "Distance",
                                           "Query X", "Query Y", "Train X", "Train Y"])

    # Ensure target directory exists
    os.makedirs(target_folder, exist_ok=True)

    # Generate a unique file name
    file_name = f"{image_identifier}_matches.csv"
    file_path = os.path.join(target_folder, file_name)

    # Save DataFrame to CSV
    df.to_csv(file_path, index=False)

    # print(f"Matches saved to {file_path}")