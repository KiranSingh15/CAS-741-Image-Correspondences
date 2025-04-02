

def compute_descriptors(mthd_kp_description, orb_object, img, keypoints):
    if mthd_kp_description == 1:
        descriptors = orb_object.compute(img, keypoints)

    return descriptors
