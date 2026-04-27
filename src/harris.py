import cv2
import numpy as np


def harris_corner_detection(gray_image, block_size=2, ksize=3, k=0.04):
    """
    Apply Harris Corner Detection.

    Args:
        gray_image (np.ndarray): Grayscale image
        block_size (int): Neighborhood size
        ksize (int): Sobel kernel size
        k (float): Harris detector free parameter

    Returns:
        np.ndarray: Corner response matrix
    """
    
    gray = np.float32(gray_image)

    # Harris response
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
    """
    Draw detected corners on the original image.

    Args:
        original_image (np.ndarray): Original BGR image
        response (np.ndarray): Harris response matrix
        threshold (float): Threshold for strong corners

    Returns:
        np.ndarray: Image with corners marked in red
    """
    image_with_corners = original_image.copy()

    mask = get_corner_mask(response, threshold)

    
    image_with_corners[mask] = [0, 0, 255]

    return image_with_corners


def count_corners(response, threshold=0.01):
    """
    Count number of detected corners.

    Args:
        response (np.ndarray): Harris response matrix
        threshold (float): Threshold ratio

    Returns:
        int: Number of detected corners
    """
    mask = get_corner_mask(response, threshold)
    return np.sum(mask)


def harris_pipeline(original_image, gray_image, threshold=0.01):
    """
    Full Harris pipeline.

    Args:
        original_image (np.ndarray): Original BGR image
        gray_image (np.ndarray): Grayscale image
        threshold (float): Threshold for corner detection

    Returns:
        dict: Contains:
            - image_with_corners
            - response
            - num_corners
    """
    response = harris_corner_detection(gray_image)
    image_with_corners = draw_corners(original_image, response, threshold)
    num_corners = count_corners(response, threshold)

    return {
        "image": image_with_corners,
        "response": response,
        "num_corners": num_corners
    }