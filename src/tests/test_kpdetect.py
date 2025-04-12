import pytest
import cv2 as cv
import numpy as np
import projectFiles.KeypointDetectionModule as kpmod


# --------------------------
# Tests for initialize_orb()
# --------------------------

def test_initialize_orb_valid_config():
    orb = kpmod.initialize_orb(
        mthd_kp_detection=1,
        mthd_kp_description=1,
        bin_sz=500,
        patch_sz=31,
        fast_thresh=20
    )
    assert isinstance(orb, cv.ORB), "Should return a cv2.ORB instance"


def test_initialize_orb_invalid_config_detection():
    orb = kpmod.initialize_orb(
        mthd_kp_detection=0,
        mthd_kp_description=1,
        bin_sz=500,
        patch_sz=31,
        fast_thresh=20
    )
    assert orb is None


def test_initialize_orb_invalid_config_description():
    orb = kpmod.initialize_orb(
        mthd_kp_detection=1,
        mthd_kp_description=0,
        bin_sz=500,
        patch_sz=31,
        fast_thresh=20
    )
    assert orb is None


def test_initialize_orb_invalid_both():
    orb = kpmod.initialize_orb(
        mthd_kp_detection=0,
        mthd_kp_description=0,
        bin_sz=500,
        patch_sz=31,
        fast_thresh=20
    )
    assert orb is None


# -------------------------------
# Tests for detect_keypoints_ofast()
# -------------------------------

def test_detect_keypoints_with_valid_orb():
    img = np.zeros((100, 100), dtype=np.uint8)
    orb = cv.ORB.create()
    keypoints = kpmod.detect_keypoints_ofast(1, orb, img)

    assert isinstance(keypoints, (list, tuple))
    assert all(isinstance(kp, cv.KeyPoint) for kp in keypoints)


def test_detect_keypoints_invalid_method():
    img = np.zeros((100, 100), dtype=np.uint8)
    orb = cv.ORB.create()
    keypoints = kpmod.detect_keypoints_ofast(0, orb, img)

    assert keypoints is None


def test_detect_keypoints_with_none_orb():
    img = np.zeros((100, 100), dtype=np.uint8)
    # Intentionally passing None to simulate misconfiguration
    with pytest.raises(AttributeError):
        _ = kpmod.detect_keypoints_ofast(1, None, img)
