import cv2 as cv
import numpy as np
import pytest


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import projectFiles.FeatureDescriptorModule as fdmod


def test_compute_descriptors_valid():
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    orb = cv.ORB.create()
    keypoints = orb.detect(img, None)  # Let ORB pick real keypoints

    result = fdmod.compute_descriptors(1, orb, img, keypoints)

    assert isinstance(result, tuple)
    keypoints_out, descriptors = result
    assert isinstance(keypoints_out, (list, tuple))
    assert all(isinstance(kp, cv.KeyPoint) for kp in keypoints_out)
    assert descriptors is not None and isinstance(descriptors, np.ndarray)


def test_compute_descriptors_invalid_method():
    img = np.zeros((100, 100), dtype=np.uint8)
    orb = cv.ORB.create()
    keypoints = [cv.KeyPoint(30, 30, 10)]

    result = fdmod.compute_descriptors(0, orb, img, keypoints)

    # Function does not return anything explicitly
    assert result is None


def test_compute_descriptors_with_no_keypoints():
    img = np.zeros((100, 100), dtype=np.uint8)
    orb = cv.ORB.create()
    keypoints = []

    result = fdmod.compute_descriptors(1, orb, img, keypoints)

    assert isinstance(result, tuple)
    keypoints_out, descriptors = result

    # No keypoints, so output should reflect that
    assert len(keypoints_out) == 0
    assert descriptors is None


def test_compute_descriptors_with_none_orb():
    img = np.zeros((100, 100), dtype=np.uint8)
    keypoints = [cv.KeyPoint(30, 30, 10)]

    with pytest.raises(AttributeError):
        _ = fdmod.compute_descriptors(1, None, img, keypoints)
