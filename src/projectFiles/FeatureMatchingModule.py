import cv2 as cv


# initial OpenCV feature matching object
def create_BF_matcher(mthd_ft_match, norm_method):
    if mthd_ft_match == 1:
        bfm_object = cv.BFMatcher(norm_method, crossCheck=True)
        return bfm_object

    return None


# compare match objects
def match_features(bfm_object, desc1, desc2):
    matches = bfm_object.match(desc1, desc2)
    return matches


# sort match objects by Hamming Distance
def sort_matches(matches):
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    return sorted_matches
