import cv2
import numpy as np

def build_gaussian_pyramid(image, levels=3):
    """
    Generates a Gaussian pyramid by progressively downsampling the image.
    
    Args:
        image (np.ndarray): Input image.
        levels (int): Number of downsampling steps.
        
    Returns:
        list: A list of images from original resolution to the smallest level.
    """
    pyramid = [image]
    temp_img = image.copy()
    
    for i in range(levels):
        temp_img = cv2.pyrDown(temp_img)
        pyramid.append(temp_img)
        
    return pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    """
    Builds a Laplacian pyramid from a pre-calculated Gaussian pyramid.
    Captures the edges and details at each scale.
    
    Args:
        gaussian_pyramid (list): List of images from a Gaussian pyramid.
        
    Returns:
        list: List of detail images (edges).
    """
    laplacian_pyramid = []
    
    # Iterate from the smallest image back up to the largest
    for i in range(len(gaussian_pyramid) - 1, 0, -1):
        # 1. Expand the smaller image to the size of the larger one
        expanded = cv2.pyrUp(gaussian_pyramid[i])
        
        # 2. Ensure dimensions match exactly (PyrUp can occasionally differ by 1px)
        upper_level = gaussian_pyramid[i-1]
        expanded = cv2.resize(expanded, (upper_level.shape[1], upper_level.shape[0]))
        
        # 3. Subtract the expanded image from the higher resolution version
        laplacian = cv2.subtract(upper_level, expanded)
        laplacian_pyramid.append(laplacian)
        
    return laplacian_pyramid

def display_pyramid(pyramid_list, name="Pyramid Level"):
    """
    Displays each level of a pyramid in a separate window.
    """
    for i, img in enumerate(pyramid_list):
        window_name = f"{name} {i}"
        cv2.imshow(window_name, img)

def run_pyramid_pipeline(image, levels=3):
    """
    Full pipeline to generate both Gaussian and Laplacian pyramids.
    
    Returns:
        dict: Lists containing images for both pyramid types.
    """
    gauss_pyr = build_gaussian_pyramid(image, levels)
    lap_pyr = build_laplacian_pyramid(gauss_pyr)
    
    return {
        "gaussian": gauss_pyr,
        "laplacian": lap_pyr
    }

