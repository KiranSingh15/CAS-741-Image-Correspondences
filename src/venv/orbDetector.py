import cv2 as cv

def create_orb_object(bin_sz, patch_sz, fast_thresh):
    orb = cv.ORB.create(nfeatures=bin_sz, patchSize=patch_sz,
                        fastThreshold = fast_thresh)
    return orb


def detect_keypoints_ofast(orb, img):
    # Keypoint Detection
    kp = orb.detect(img, None)
    return kp


def detect_features_rbrief(orb, img, kp):
    # Feature Descriptor
    features = orb.compute(img, kp, None)
    return features



