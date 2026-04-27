import cv2


def detect_sift_features(gray_image):
    """
    Detect SIFT keypoints and descriptors.

    Args:
        gray_image: Grayscale image

    Returns:
        keypoints, descriptors
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors


def draw_keypoints(image, keypoints):
    """
    Draw SIFT keypoints on the image.
    """
    return cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


def match_features(des1, des2):
    """
    Match features using BFMatcher + Lowe's ratio test.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    return good_matches


def draw_matches(img1, kp1, img2, kp2, matches):
    """
    Draw matches between two images.
    """
    return cv2.drawMatchesKnn(
        img1, kp1,
        img2, kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )