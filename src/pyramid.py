import cv2
import numpy as np

def build_gaussian_pyramid(image, levels=3):
   
    pyramid = [image]
    temp_img = image.copy()
    
    for i in range(levels):
        temp_img = cv2.pyrDown(temp_img)
        pyramid.append(temp_img)
        
    return pyramid

def build_laplacian_pyramid(gaussian_pyramid):
   
    laplacian_pyramid = []
    
  
    for i in range(len(gaussian_pyramid) - 1, 0, -1):
        
        expanded = cv2.pyrUp(gaussian_pyramid[i])
        
        
        upper_level = gaussian_pyramid[i-1]
        expanded = cv2.resize(expanded, (upper_level.shape[1], upper_level.shape[0]))
        
        
        laplacian = cv2.subtract(upper_level, expanded)
        laplacian_pyramid.append(laplacian)
        
    return laplacian_pyramid

def display_pyramid(pyramid_list, name="Pyramid Level"):
  
    for i, img in enumerate(pyramid_list):
        window_name = f"{name} {i}"
        cv2.imshow(window_name, img)

def run_pyramid_pipeline(image, levels=3):
   
    gauss_pyr = build_gaussian_pyramid(image, levels)
    lap_pyr = build_laplacian_pyramid(gauss_pyr)
    
    return {
        "gaussian": gauss_pyr,
        "laplacian": lap_pyr
    }

