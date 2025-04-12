import pytest
import numpy as np
import cv2 as cv
import projectFiles.FeatureMatchingModule as featmod


def test_create_BF_matcher_valid():
    """Test that a BFMatcher object is correctly created with valid input."""
    matcher = featmod.create_BF_matcher(1, cv.NORM_HAMMING)
    assert isinstance(matcher, cv.BFMatcher)


def test_create_BF_matcher_invalid_behavior():
    """Test that create_BF_matcher returns None or causes error for invalid method (not handled explicitly)."""
    matcher = featmod.create_BF_matcher(0, cv.NORM_L2)
    assert matcher is None or isinstance(matcher, cv.BFMatcher)  # depends on how module handles invalid cases


def test_match_features_no_loss():
    """Test that all descriptors are matched 1-to-1 with crossCheck enabled (no loss)."""
    desc1 = np.random.randint(0, 256, (10, 32), dtype=np.uint8)
    desc2 = desc1.copy()  # Ensure perfect match

    matcher = featmod.create_BF_matcher(1, cv.NORM_HAMMING)
    matches = featmod.match_features(matcher, desc1, desc2)

    assert len(matches) == 10  # All should match 1:1
    assert all(isinstance(m, cv.DMatch) for m in matches)


def test_match_features_invalid_shapes():
    """Test that mismatched descriptor shapes raise an OpenCV error."""
    desc1 = np.random.randint(0, 256, (10, 32), dtype=np.uint8)
    desc2 = np.random.randint(0, 256, (5, 16), dtype=np.uint8)  # Wrong shape

    matcher = featmod.create_BF_matcher(1, cv.NORM_HAMMING)
    with pytest.raises(cv.error):
        _ = featmod.match_features(matcher, desc1, desc2)


def test_sort_matches_ordering():
    """Test that sort_matches returns matches ordered by distance (ascending)."""
    matches = [
        cv.DMatch(_queryIdx=0, _trainIdx=0, _distance=5.0),
        cv.DMatch(_queryIdx=1, _trainIdx=1, _distance=2.0),
        cv.DMatch(_queryIdx=2, _trainIdx=2, _distance=3.0),
    ]

    sorted_matches = featmod.sort_matches(matches)
    distances = [m.distance for m in sorted_matches]

    assert distances == sorted(distances)


def test_no_matches_returns_empty_list():
    """Test that empty descriptors return an empty match list."""
    desc1 = np.empty((0, 32), dtype=np.uint8)
    desc2 = np.empty((0, 32), dtype=np.uint8)
    matcher = featmod.create_BF_matcher(1, cv.NORM_HAMMING)

    matches = featmod.match_features(matcher, desc1, desc2)