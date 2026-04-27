# import numpy as np
# import cv2

# def extract_features(gray, mask):
#     roi = cv2.bitwise_and(gray, gray, mask=mask)

#     pixels = roi[mask == 255]

#     if len(pixels) == 0:
#         return [0, 0, 0]

#     mean = np.mean(pixels)
#     std = np.std(pixels)
#     area = len(pixels)

#     return [mean, std, area]


import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern

def extract_features(gray, mask):

    roi = cv2.bitwise_and(gray, gray, mask=mask)
    pixels = roi[mask == 255]

    if len(pixels) == 0:
        return np.zeros(8)

    # ===== Statistical =====
    mean = np.mean(pixels)
    std = np.std(pixels)
    sk = skew(pixels)
    kurt = kurtosis(pixels)
    area = len(pixels)

    # ===== LBP =====
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_pixels = lbp[mask == 255]

    lbp_mean = np.mean(lbp_pixels)
    lbp_std = np.std(lbp_pixels)

    # ===== Edges =====
    edges = cv2.Canny(roi, 100, 200)
    edge_pixels = edges[mask == 255]
    edge_density = np.sum(edge_pixels > 0) / len(pixels)

    # ===== FINAL VECTOR 🔥🔥
    features = np.array([
        mean,
        std,
        sk,
        kurt,
        area,
        lbp_mean,
        lbp_std,
        edge_density
    ])

    return features