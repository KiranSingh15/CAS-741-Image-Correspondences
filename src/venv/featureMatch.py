import cv2 as cv
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

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

    print(f"Data loaded and converted to ORB descriptors and keypoints from {file_path}")
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
    cv.imshow('Output image', output_image)


# Draw first 10 matches.
def draw_matches(img1, kp1, img2, kp2, matches, max_disp_matches):
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:max_disp_matches], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()