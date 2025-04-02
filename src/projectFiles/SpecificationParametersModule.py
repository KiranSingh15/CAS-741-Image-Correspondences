# This file outlines the software constraints and limits for the input data

# Exported Constants
# tuning parameters
k = int(5)  # gaussian kernel size
sigma = 3  # gaussian blur standard deviation
t = int(15)  # pixel intensity threshold
b = int(2000)  # descriptor bin size
p = int(31)  # descriptor patch search size

# available methods of processing
mthds_is= ["Gaussian Kernel"] # methods of image smoothing
mthd_kpd = ["FAST","Harris"] # methods of keypoint detection
mthd_fd = ["Binary Descriptors"] # methods of assigning feature descriptors
mthd_ftm = ["Hamming Distance - Brute Force", "Hamming Distance - FLANN"]  # methods of comparing descriptors

# methods
mthd_img_smoothing = int(1)
mthd_kp_detection = int(1)
mthd_kp_description = int(1)
mthd_ft_match = int(1)

def get_available_methods ():
    method_count = [len(mthds_is), len(mthd_kpd), len(mthd_fd), len(mthd_ftm)]
    return method_count, mthds_is, mthd_kpd, mthd_fd, mthd_ftm

def get_default_parameters():
    # def_params = {
    # "kernel_size": 3,  # where kernel size must be positive, odd, and greater or equal to 3
    # "standard deviation": 1,  # where the standard deviation must be a positive, real number
    # "fast threshold": 15,
    # "bin size": 2000,
    # "patch size": 31,
    # }
    return k, sigma, t, b, p


def get_default_methods():
    # TO BE USED IN SUBSEQUENT UPDATE
    # mthds_is= ["Gaussian Kernel"] # methods of image smoothing
    # mthd_kpd = ["FAST","Harris"] # methods of keypoint detection
    # mthd_fd = ["Binary Descriptors"] # methods of assigning feature descriptors
    # mthd_ftm = ["Hamming Distance"]  # methods of comparing descriptors
    # method_limits = [len(mthds_is), len(mthd_kpd), len(mthd_fd), len(mthd_ftm)]

    ifc_def_mthds = [
        mthd_img_smoothing,
        mthd_kp_detection,
        mthd_kp_description,
        mthd_ft_match,
    ]

    return mthd_img_smoothing, mthd_kp_detection, mthd_kp_description, mthd_ft_match
    # return ifc_def_mthds, method_limits
