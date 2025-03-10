# This file serves as the configuration file for the Image Feature Correspondence (IFC) software
# this file should be checked against
from pathlib import Path
import os

# file paths
def get_head_directory():
    return Path(os.getcwd())  # Convert to Path object

# Activate functions, where inactive = 0, and all other permutations are defined by [1,n], where n is the last method
def get_active_functions():
    active_fxns = {
        "gs_conversion": 1,
        "img_smoothing": 1,
        "keypoint_detection": 1,
        "descriptor_assignment": 1,
        "feature_matching": 1
    }

    # uncomment for visualization purposes only.
    # for i, (k, v) in enumerate(active_fxns.items()):  # k=key, v=value
    #     print(i, k, v)

    return active_fxns

def get_chosen_parameters():

    # Input Parameters
    in_params = {
    "kernel size": int(9), # where kernel size must be positive, odd, and greater or equal to 3
    "standard deviation": 1,# where the standard deviation must be a positive, real number
    "fast threshold": int(5),
    "bin size": int(31),
    "patch size": int(35),
    "distance_threshold": int(20)
    }

    return in_params



