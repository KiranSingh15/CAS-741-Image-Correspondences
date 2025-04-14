import warnings

def check_match_uniqueness(query_img_id, train_img_id, matches):
    """
    Issues a warning if feature matches are found between the same image ID.
    """
    if query_img_id == train_img_id:
        warnings.warn(
            f"Non-unique image match detected: query and train image IDs are both '{query_img_id}'.",
            category=UserWarning
        )
    return True