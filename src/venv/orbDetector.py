import cv2 as cv
import pandas as pd
import os
import numpy as np

def create_orb_object(bin_sz, patch_sz, fast_thresh):
    orb = cv.ORB.create(nfeatures=bin_sz, patchSize=patch_sz,
                        fastThreshold = fast_thresh)
    return orb


def detect_keypoints_ofast(orb, img):
    # Keypoint Detection
    kp = orb.detect(img, None)
    return kp


def detect_features_rbrief(orb, img, kp):
    # Feature Descriptor
    features = orb.compute(img, kp, None)
    return features

def save_keypoints(keypoints, image_identifier, target_folder):
    # Convert keypoints to a list of tuples
    keypoint_list = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

    # Create a DataFrame
    df = pd.DataFrame(keypoint_list, columns=["x", "y", "size", "angle", "response", "octave", "class_id"])

    # Name the file
    # unique_file_name = "PoseID_" + str(pose_id) + "_" + "CameraID_" + str(camera_id) + "_" + "kp"
    unique_file_name = image_identifier + "_kp"
    full_file_name = unique_file_name + ".csv"
    file_path = os.path.join(target_folder, full_file_name)
    df.to_csv(file_path, index=False)

    # print("Keypoints saved to keypoints.csv")


def save_descriptors(keypoints, descriptors, image_identifier, target_folder):
    """
    Saves keypoints and their associated descriptors to a CSV file.

    :param keypoints: List of keypoints detected in the image.
    :param descriptors: Descriptors associated with the keypoints (as numpy array).
    :param image_identifier: Identifier for the image (used in the CSV file name).
    :param target_folder: Directory where the CSV file will be saved.
    """
    # Ensure descriptors is the second element in the tuple
    if isinstance(descriptors, tuple):
        descriptors = descriptors[1]  # Access the second element of the tuple which contains descriptors

    if descriptors is None:
        print(f"No descriptors found for {image_identifier}. Skipping save.")
        return

    # Convert keypoints to a list of tuples (x, y, size, angle, response, octave, class_id)
    keypoint_list = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

    # Ensure descriptors is a numpy array of the correct type
    if isinstance(descriptors, np.ndarray) and descriptors.dtype == np.uint8:
        # Convert descriptors to a list (convert binary descriptors to a string representation of bits)
        descriptor_list = [''.join([str(int(b)) for b in np.unpackbits(np.uint8(desc))]) for desc in descriptors]
    else:
        print(f"Invalid descriptor type. Expected np.uint8 array, got {type(descriptors)}")
        return

    # Combine the keypoints and descriptors into a single list of tuples
    combined_data = [keypoint + (descriptor,) for keypoint, descriptor in zip(keypoint_list, descriptor_list)]

    # Create a DataFrame
    df = pd.DataFrame(combined_data,
                      columns=["x", "y", "size", "angle", "response", "octave", "class_id", "descriptor"])

    # Generate a unique file name based on the image identifier
    unique_file_name = image_identifier + "_fd"
    full_file_name = unique_file_name + ".csv"
    file_path = os.path.join(target_folder, full_file_name)

    # Save the DataFrame to CSV
    df.to_csv(file_path, index=False)

    print(f"Keypoints and descriptors saved to {file_path}")

