# import cv2
# import numpy as np


# def threshold_segmentation(gray_image):
#     """
#     Apply Gaussian blur + Otsu threshold (inverted).
#     """
#     blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
    

#     _, thresh = cv2.threshold(
#         blurred,
#         0,
#         255,
#         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#     )

#     return thresh


# def extract_lung_region(binary_mask):
#     """
#     Extract lung regions using morphology + top 2 contours.
#     """

#     # =========================
#     # 1. Focus on chest region
#     # =========================
#     h, w = binary_mask.shape
#     chest_mask = np.zeros_like(binary_mask)

#     cv2.rectangle(
#         chest_mask,
#         (int(w * 0.2), int(h * 0.1)),
#         (int(w * 0.8), int(h * 0.95)),
#         255,
#         -1
#     )

#     binary_mask = cv2.bitwise_and(binary_mask, chest_mask)

#     # =========================
#     # 2. Morphology
#     # =========================
#     kernel = np.ones((5, 5), np.uint8)

#     morphed = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
#     morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel, iterations=2)

#     # =========================
#     # 3. Find contours
#     # =========================
#     contours, _ = cv2.findContours(
#         morphed,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE
#     )

#     lung_mask = np.zeros_like(binary_mask)

#     if contours:
#         # Sort by area
#         sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

#         # =========================
#         # 4. Take top 2 contours (lungs)
#         # =========================
#         for cnt in sorted_contours[:2]:
#             cv2.drawContours(lung_mask, [cnt], -1, 255, -1)

#     return lung_mask


# def segmentation_pipeline(original_bgr, gray_image):
#     """
#     Full segmentation pipeline with visualization.
#     """

#     # Step 1
#     thresh = threshold_segmentation(gray_image)

#     # Step 2
#     final_mask = extract_lung_region(thresh)

#     # Step 3: Overlay
#     overlay = original_bgr.copy()
#     overlay[final_mask == 255] = [0, 0, 255]

#     visualization = cv2.addWeighted(overlay, 0.4, original_bgr, 0.6, 0)

#     return {
#         "mask": final_mask,
#         "thresholded": thresh,
#         "visualization": visualization
#     }

import cv2
import numpy as np


def threshold_segmentation(gray_image):

    # تحسين الإضاءة 🔥
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)

    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

    # Otsu
    _, thresh1 = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Adaptive 🔥
    thresh2 = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # دمج الاتنين
    thresh = cv2.bitwise_or(thresh1, thresh2)

    # تنظيف
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return thresh


def extract_lung_region(binary_mask):

    h, w = binary_mask.shape

    # Chest focus
    chest_mask = np.zeros_like(binary_mask)
    cv2.rectangle(
        chest_mask,
        (int(w * 0.2), int(h * 0.1)),
        (int(w * 0.8), int(h * 0.95)),
        255, -1
    )

    binary_mask = cv2.bitwise_and(binary_mask, chest_mask)

    # Morphology
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Contours
    contours, _ = cv2.findContours(
        morphed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lung_mask = np.zeros_like(binary_mask)

    if contours:

        img_area = h * w

        # فلترة بالمساحة 🔥
        valid = [cnt for cnt in contours if cv2.contourArea(cnt) > 0.05 * img_area]

        # fallback 🔥🔥
        if len(valid) == 0:
            return binary_mask

        sorted_contours = sorted(valid, key=cv2.contourArea, reverse=True)

        for cnt in sorted_contours[:2]:
            cv2.drawContours(lung_mask, [cnt], -1, 255, -1)

    # fallback لو فاضي جدًا
    if np.sum(lung_mask) < 5000:
        return binary_mask

    return lung_mask


def segmentation_pipeline(original_bgr, gray_image):

    thresh = threshold_segmentation(gray_image)
    final_mask = extract_lung_region(thresh)

    overlay = original_bgr.copy()
    overlay[final_mask == 255] = [0, 0, 255]

    visualization = cv2.addWeighted(overlay, 0.4, original_bgr, 0.6, 0)

    return {
        "mask": final_mask,
        "thresholded": thresh,
        "visualization": visualization
    }