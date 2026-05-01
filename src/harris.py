import cv2
import numpy as np


def harris_corner_detection(gray_image, block_size=2, ksize=3, k=0.04):
    
    
    gray = np.float32(gray_image)

    response = cv2.cornerHarris(gray, block_size, ksize, k)

   
    response = cv2.dilate(response, None)

    return response


def get_corner_mask(response, threshold=0.01):
    """
    Generate a boolean mask for strong corners.

    Args:
        response (np.ndarray): Harris response matrix
        threshold (float): Threshold ratio

    Returns:
        np.ndarray: Boolean mask of detected corners
    """
    return response > (threshold * response.max())


def draw_corners(original_image, response, threshold=0.01):

    image_with_corners = original_image.copy()

    mask = get_corner_mask(response, threshold)

    
    image_with_corners[mask] = [0, 0, 255]

    return image_with_corners


def count_corners(response, threshold=0.01):

    mask = get_corner_mask(response, threshold)
    return np.sum(mask)


def harris_pipeline(original_image, gray_image, threshold=0.01):
    
    response = harris_corner_detection(gray_image)
    image_with_corners = draw_corners(original_image, response, threshold)
    num_corners = count_corners(response, threshold)

    return {
        "image": image_with_corners,
        "response": response,
        "num_corners": num_corners
    }