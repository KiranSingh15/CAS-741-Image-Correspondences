import cv2 as cv
import pandas as pd
import os

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

def save_keypoints(keypoints, pose_id, camera_id, target_folder):
    # Convert keypoints to a list of tuples
    keypoint_list = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

    # Create a DataFrame
    df = pd.DataFrame(keypoint_list, columns=["x", "y", "size", "angle", "response", "octave", "class_id"])

    # Name the file
    unique_file_name = "PoseID_" + str(pose_id) + "_" + "CameraID_" + str(camera_id) + "_" + "keypoints"
    full_file_name = unique_file_name + ".csv"
    file_path = os.path.join(target_folder, full_file_name)
    # df.to_csv(file_path, index=False)

    # print("Keypoints saved to keypoints.csv")