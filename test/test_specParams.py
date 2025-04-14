import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import projectFiles.SpecificationParametersModule as specParams

lim_kern_bounds = [3, 11]  # 3 and 11 inclusive to scale down the kernel
lim_sd_bounds = [0, 10]  # (0, 10], or 0 exclusive and 10 inclusive
lim_fast_bounds = [2, 254]  # inclusive
lim_bin_bounds = [1, 2048]  # exclusive
lim_patch_sz = [5, 100]  # inclusive
lim_match_distance_limits = [0, 150]  # exclusive
lim_num_match_disp = [1, 1000]  # inclusive


method_count, mthd_is, mthd_kpd, mthd_fd, mthd_ftm = specParams.get_available_methods()
mthd_img_smoothing, mthd_kp_detection, mthd_kp_description, mthd_ft_match = (
    specParams.get_assigned_methods()
)
test_k, test_sigma, test_t, test_b, test_p, test_d, test_displayed_matches = (
    specParams.get_assigned_parameters()
)


# Gaussian Kernel
def test_kernel_bounds():
    """
    Test that the assigned Gaussian kernel size is:
    - an integer,
    - an odd number,
    - within the defined bounds (inclusive).

    Fails if:
    - the type is not int,
    - the kernel size is even,
    - the size falls outside of [3, 15].
    """
    assert isinstance(test_k, int), "Lower Gaussian kernel limit must be an integer"
    assert test_k % 2 == 1, f"kern_bounds must be odd but is equal to {test_k}"
    assert test_k >= lim_kern_bounds[0], "Expected lower Gaussian kernel limit to be 3"
    assert test_k <= lim_kern_bounds[1], "Expected upper Gaussian kernel limit to be 15"


# Gaussian Standard Deviation
def test_standard_deviation():  # if std dev is not a natural number, then need to account for floating point error
    """
    Verifies that the Gaussian standard deviation:
    - is a float or int,
    - is strictly greater than 0,
    - is at most 10.

    Uses epsilon for float rounding.

    Fails if:
    - the type is incorrect,
    - it is out of the interval (0, 10].
    """
    assert isinstance(test_sigma, (float, int)), f"sigma is of type {type(test_sigma)}."
    assert (
        test_sigma > lim_sd_bounds[0] - 1e-6
    ), f"Test Gaussian standard deviation  failed. (Standard Deviation = {test_sigma})"
    assert (
        test_sigma <= lim_sd_bounds[1] + 1e-6
    ), f"Test Gaussian standard deviation  failed. (Standard Deviation = {test_sigma})"


# Test FAST intensity threshold
def test_fast_bounds():  # pixel intensity
    """
    Validates the FAST threshold parameter:
    - must be an integer,
    - in the range [2, 254].

    Fails if:
    - the type is incorrect,
    - the value is outside the allowable threshold range.
    """
    assert isinstance(test_t, int)
    assert (
        test_t >= lim_fast_bounds[0]
    ), f"Test FAST threshold failed. (Threshold = {test_t})"
    assert (
        test_t <= lim_fast_bounds[1]
    ), f"Test FAST threshold failed. (Threshold = {test_t})"


# Bin Size
def test_bin_bounds():
    """
    Ensures the descriptor bin size:
    - is an integer,
    - falls within the allowed range [1, 2048].

    Fails if:
    - not an integer,
    - value is out of range.
    """
    assert isinstance(test_b, int)
    assert test_b >= lim_bin_bounds[0], f"Test bin size failed. (Bin size = {test_b})"
    assert test_b <= lim_bin_bounds[1], f"Test bin size failed. (Bin size = {test_b})"


# Patch Size
def test_patch_size_bounds():
    """
    Verifies the patch size for feature descriptors:
    - is an integer,
    - is between 5 and 100, inclusive.

    Fails if:
    - not an integer,
    - value out of range.
    """
    assert isinstance(test_p, int)

    assert test_p >= lim_patch_sz[0], f"Test patch size failed. (Patch size = {test_p})"
    assert test_p <= lim_patch_sz[1], f"Test patch size failed. (Patch size = {test_p})"


# Match Distance Limit
def test_match_distance_limits():
    """
    Confirms that the maximum match distance:
    - is an integer,
    - is within (0, 150].

    Fails if:
    - not an integer,
    - value is nonpositive or above limit.
    """
    assert isinstance(test_d, int)
    assert (
        test_d > lim_match_distance_limits[0]
    ), f"badMatchDistance. (Match distance = {test_d})"
    assert (
        test_d <= lim_match_distance_limits[1]
    ), f"badMatchDistance. (Match distance = {test_d})"


# Displayed Matches Count
def test_num_match_disp():
    """
    Validates the number of displayed matches:
    - is an integer,
    - is in the range [1, 1000].

    Fails if:
    - not an integer,
    - value is below 1 or above 1000.
    """
    assert isinstance(test_displayed_matches, int)
    assert (
        test_displayed_matches >= lim_num_match_disp[0]
    ), f"Test of the specified number of displayed matches failed. (Selected match distance = {test_displayed_matches})"
    assert (
        test_displayed_matches <= lim_num_match_disp[1]
    ), f"Test of the specified number of displayed matches failed. (Selected match distance = {test_displayed_matches})"
    assert isinstance(
        test_displayed_matches, int
    ), f"Number of displayed matches must be an integer. (Max limit of displayed matches = {test_displayed_matches})"


# Test available methods of processing
def test_avail_methods():
    """
    Checks availability of at least one method in each category:
    - Image smoothing,
    - Keypoint detection,
    - Feature description,
    - Feature matching.

    Fails if:
    - any method list is not a list or tuple,
    - any list is empty.
    """
    assert isinstance(
        specParams.mthd_is, (list, tuple)
    ), f"The format of available image smoothing methods cannot be read."
    assert isinstance(
        specParams.mthd_kpd, (list, tuple)
    ), f"The format of available keypoint detection methods cannot be read."
    assert isinstance(
        specParams.mthd_fd, (list, tuple)
    ), f"The format of available feature descriptor methods cannot be read."
    assert isinstance(
        specParams.mthd_ftm, (list, tuple)
    ), f"The format of available feature matching methods cannot be read."
    assert (
        len(specParams.mthd_is) > 0
    ), "At least one method of Image Smoothing must be available to define the default method"
    assert (
        len(specParams.mthd_kpd) > 0
    ), "At least one method of keypoint detection must be available to define the default method"
    assert (
        len(specParams.mthd_fd) > 0
    ), "At least one method of at feature detection must be available to define the default method"
    assert (
        len(specParams.mthd_ftm) > 0
    ), "At least one method of feature matching must be available to define the default method"


# Test User selected methods
def test_selected_methods():
    """
    Ensures that user-specified methods for each processing stage:
    - are integers,
    - are non-negative (0 means disabled),
    - align with valid enumerated options.

    Fails if:
    - not an int,
    - value < 0.
    """
    assert isinstance(
        specParams.mthd_img_smoothing, int
    ), f"The selected method of available image smoothing must be enumerated."
    assert isinstance(
        specParams.mthd_kp_detection, int
    ), f"The selected method of available keypoint detection must be enumerated."
    assert isinstance(
        specParams.mthd_kp_description, int
    ), f"The selected method of available feature description must be enumerated."
    assert isinstance(
        specParams.mthd_ft_match, int
    ), f"The selected method of available feature matching must be enumerated."
    assert (
        specParams.mthd_img_smoothing >= 0
    ), f"Selected Image Smoothing method must be 0 to disable, or 1 through n, where n is the number of available methods."
    assert (
        specParams.mthd_kp_detection >= 0
    ), f"Selected Keypoint Detection method must be 0 to disable, or 1 through n, where n is the number of available methods."
    assert (
        specParams.mthd_img_smoothing >= 0
    ), f"Selected Feature Description method must be 0 to disable, or 1 through n, where n is the number of available methods."
    assert (
        specParams.mthd_kp_detection >= 0
    ), f"Selected Feature Matching method must be 0 to disable, or 1 through n, where n is the number of available methods."
