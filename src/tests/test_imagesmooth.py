import pytest
import numpy as np
import cv2 as cv
import projectFiles.ImageSmoothingModule as smoothmod

# ------------------------------
# Tests for greyscale conversion
# ------------------------------
def test_convert_to_greyscale_output_shape():
    # Test that a 3-channel image is correctly converted to a single-channel grayscale image
    img_colour = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White BGR image
    img_gray = smoothmod.convert_to_greyscale(img_colour)

    assert img_gray.shape == (100, 100)
    assert img_gray.dtype == np.uint8


def test_convert_to_greyscale_actual_conversion():
    # Test that conversion from a blue-only image results in a dark grayscale image
    img_colour = np.zeros((10, 10, 3), dtype=np.uint8)
    img_colour[..., 0] = 255  # Set only blue channel

    img_gray = smoothmod.convert_to_greyscale(img_colour)
    assert img_gray.mean() < 100  # Blue has less impact on luminance


def test_convert_to_greyscale_invalid_input():
    # Test that passing a single-channel (already grayscale) image raises an error
    with pytest.raises(cv.error):
        _ = smoothmod.convert_to_greyscale(np.zeros((10, 10), dtype=np.uint8))  # Not BGR


# ------------------------------
# Tests for smooth_image
# ------------------------------

def test_smooth_image_valid_gaussian():
    # Test that Gaussian blur with valid parameters returns a same-sized, blurred image
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    blurred = smoothmod.smooth_image(1, img, sz_kern=5, std_dev=1.0)

    assert blurred.shape == img.shape
    assert blurred.dtype == img.dtype
    assert not np.array_equal(img, blurred)  # Should differ from original


def test_smooth_image_invalid_method_returns_nothing():
    # Test that an unrecognized smoothing method (e.g., 0) returns None
    img = np.ones((50, 50), dtype=np.uint8) * 127
    result = smoothmod.smooth_image(0, img, sz_kern=5, std_dev=1.0)

    assert result is None


def test_smooth_image_invalid_kernel_size():
    # Even kernel sizes are not valid for GaussianBlur
    img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

    with pytest.raises(cv.error):
        _ = smoothmod.smooth_image(1, img, sz_kern=4, std_dev=1.0)

