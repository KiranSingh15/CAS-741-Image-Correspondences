# This file outlines the software constraints and limits for the input data

# Exported Constants
# function parameters
k = int(5)  # gaussian kernel size
sigma = 1  # gaussian blur standard deviation
t = int(15)  # pixel intensity threshold
b = int(2000)  # descriptor bin size
p = int(31)  # descriptor patch search size

# methods
mthd_img_smoothing = int(1)
mthd_kp_detection = int(1)
mthd_kp_description = int(1)
mthd_ft_match = int(1)


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


# Check user inputs
# def check_limits(u_sz_kern, u_std_dev, u_fast_thr, u_bin_zs, u_patch_sz):
#     err_count = 0
#     err_list = []
#
#     # Gaussian Filtering
#     kern_bounds = [3, 15] # 3 and 11 inclusive to scale down the kernel
#     sd_bounds = [0, 10] # (0, 10], or 0 exclusive and 10 inclusive
#
#     # Keypoint detection
#     fast_bounds = [2, 254] # inclusive
#
#     # Feature Detector
#     bin_bounds = [1, 2048]
#     patch_sz = [5, 100]
#
#
#     if u_sz_kern < kern_bounds[0] or u_sz_kern > kern_bounds[1]:
#         err_count += 1
#         err_list.append(
#             "Error: Gaussian Kernel size is invalid. Update the kernel size to fall within the allowable bounds before rerunning the program.")
#     elif u_sz_kern % 2 == 0 or u_sz_kern / 1 == 1:
#         err_count += 1
#         err_list.append(
#             "Error: Gaussian Kernel size is invalid. Kernel size should be an odd, positive integer, excluding 1.")
#
#     if u_std_dev <= 0 or u_std_dev > sd_bounds[1]:
#         err_count += 1
#         err_list.append(
#             "Error: Standard deviation is invalid. Update the standard deviation to fall within the allowable bounds before rerunning the program.")
#
#     if u_fast_thr < fast_bounds[0] or u_fast_thr > fast_bounds[1]:
#         err_count += 1
#         err_list.append(
#             "Error: User-defined FAST threshold is invalid. Update the threshold to fall within the allowable bounds before rerunning the program.")
#
#     if u_bin_zs < bin_bounds[0] or u_bin_zs > bin_bounds[1]:
#         err_count += 1
#         err_list.append(
#             "Error: User-defined feature descriptor bin size is invalid. Update the bin size to fall within the allowable bounds before rerunning the program.")
#
#     if u_patch_sz < patch_sz[0] or u_patch_sz > patch_sz[1]:
#         err_count += 1
#         err_list.append(
#             "User-defined patch size is invalid. Update the patch size to fall within the allowable bounds before rerunning the program.")
#
#     # print(err_count) # uncomment only to support debugging
#     if err_count == 0:
#         # print("No errors detected in user-specified parameters.") # uncomment only to support debugging
#         a = 0 # dummy line to avoid throwing an error
#     else:
#         print("Total errors detected: ", err_count) # uncomment only to support debugging
#         print(type(err_list)) # uncomment only to support debugging
#         for i in err_list:
#             print(i)
#
#     return err_count, err_list
