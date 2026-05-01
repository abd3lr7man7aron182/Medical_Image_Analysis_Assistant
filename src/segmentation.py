import cv2
import numpy as np


def threshold_segmentation(gray_image):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)

    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

    # Otsu
    _, thresh1 = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Adaptive 
    thresh2 = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    
    thresh = cv2.bitwise_or(thresh1, thresh2)

    
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

        
        valid = [cnt for cnt in contours if cv2.contourArea(cnt) > 0.05 * img_area]

        # fallback
        if len(valid) == 0:
            return binary_mask

        sorted_contours = sorted(valid, key=cv2.contourArea, reverse=True)

        for cnt in sorted_contours[:2]:
            cv2.drawContours(lung_mask, [cnt], -1, 255, -1)

    
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


def kmeans_segmentation(image, k=2):

    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        Z,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(image.shape)

    return segmented



def kmeans_mask(gray):

    Z = gray.reshape((-1, 1))
    Z = np.float32(Z)

    _, labels, centers = cv2.kmeans(
        Z, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(gray.shape)

    _, mask = cv2.threshold(segmented, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return mask