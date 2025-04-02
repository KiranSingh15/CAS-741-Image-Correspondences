import cv2 as cv


def create_BF_matcher(mthd_ft_match, norm_method):
    if mthd_ft_match == 1:
        # bfm_object = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        bfm_object = cv.BFMatcher(norm_method, crossCheck=True)
    return bfm_object


def match_features(bfm_object, desc1, desc2):
    matches = bfm_object.match(desc1, desc2)
    return matches


def sort_matches(matches):
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    return sorted_matches
