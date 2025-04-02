import cv2 as cv

def initialize_orb(mthd_kp_detection, mthd_kp_description, bin_sz, patch_sz, fast_thresh):
    if mthd_kp_detection == 1 and mthd_kp_description == 1:
        orb_object = cv.ORB.create(nfeatures=bin_sz, scoreType=cv.ORB_FAST_SCORE, patchSize=patch_sz,
                                   fastThreshold=fast_thresh)
        return orb_object

    return None


def detect_keypoints_ofast(mthd_kp_detection, orb_object, img):
    # Keypoint Detection
    if mthd_kp_detection == 1:
        kp = orb_object.detect(img, None)
        return kp

    return None