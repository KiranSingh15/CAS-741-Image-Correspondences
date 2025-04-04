# This file outlines the software constraints and limits for the input data

# Exported Constants
# Tuning Parameters
"""USER INPUT HERE"""
k = int(5)  # gaussian kernel size
sigma = 1  # gaussian blur standard deviation
t = int(15)  # pixel intensity threshold
b = int(1000)  # descriptor bin size
p = int(31)  # descriptor patch search size
d = int(25)  # upper bound for Hamming distance between features
n_disp_matches = int(30)  # max number of displayed feature matches

# available methods of processing
mthd_is = ["Gaussian Kernel"]  # methods of image smoothing
mthd_kpd = ["FAST", "Harris"]  # methods of keypoint detection
mthd_fd = ["Binary Descriptors"]  # methods of assigning feature descriptors
mthd_ftm = [
    "Hamming Distance - Brute Force",
    "Hamming Distance - FLANN",
]  # methods of comparing descriptors

# Selected Method
""" USER INPUT HERE """
mthd_img_smoothing = int(1)
mthd_kp_detection = int(1)
mthd_kp_description = int(1)
mthd_ft_match = int(1)


def get_available_methods():
    method_count = [len(mthd_is), len(mthd_kpd), len(mthd_fd), len(mthd_ftm)]
    return method_count, mthd_is, mthd_kpd, mthd_fd, mthd_ftm


def get_assigned_parameters():
    # def_params = {
    # "kernel_size": 3,  # where kernel size must be positive, odd, and greater or equal to 3
    # "standard deviation": 1,  # where the standard deviation must be a positive, real number
    # "fast threshold": 15,
    # "bin size": 2000,
    # "patch size": 31,
    # }
    return k, sigma, t, b, p, d, n_disp_matches


def get_assigned_methods():
    # TO BE USED IN SUBSEQUENT UPDATE
    # mthd_is= ["Gaussian Kernel"] # methods of image smoothing
    # mthd_kpd = ["FAST","Harris"] # methods of keypoint detection
    # mthd_fd = ["Binary Descriptors"] # methods of assigning feature descriptors
    # mthd_ftm = ["Hamming Distance"]  # methods of comparing descriptors

    ifc_def_mthds = [
        mthd_img_smoothing,
        mthd_kp_detection,
        mthd_kp_description,
        mthd_ft_match,
    ]

    return mthd_img_smoothing, mthd_kp_detection, mthd_kp_description, mthd_ft_match
