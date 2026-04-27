import cv2
import numpy as np
import os

def load_and_resize(image_path, size=(256, 256)):
    """
    Loads an image from a path and resizes it to a fixed dimension.
    
    Args:
        image_path (str): Path to the image file.
        size (tuple): Target width and height.
        
    Returns:
        np.ndarray: The resized BGR image, or None if loading fails.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def convert_to_grayscale(image):
    """
    Converts a BGR image to Grayscale.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_gaussian_filter(image, kernel_size=(5, 5), sigma=0):
    """
    Applies Gaussian Blur to the image to reduce high-frequency noise.
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_median_filter(image, kernel_size=5):
    """
    Applies Median filtering, highly effective against salt-and-pepper noise.
    """
    return cv2.medianBlur(image, kernel_size)

def calculate_mse(image_a, image_b):
    """
    Computes the Mean Squared Error (MSE) between two images.
    Lower values indicate higher similarity.
    """
    # Images must have the same dimension
    return np.mean((image_a.astype("float") - image_b.astype("float")) ** 2)

def preprocess_image(image_path):
    """
    Full pipeline: Loads, resizes, grayscales, and filters an image.
    
    Returns:
        dict: A dictionary containing all stages of the image and calculated metrics.
    """
    # 1. Load and Resize
    original = load_and_resize(image_path)
    if original is None:
        return None
    
    # 2. Grayscale
    gray = convert_to_grayscale(original)
    
    # 3. Filtering
    gaussian = apply_gaussian_filter(gray)
    median = apply_median_filter(gray)
    
    # 4. Metrics
    mse_gauss = calculate_mse(gray, gaussian)
    mse_median = calculate_mse(gray, median)
    
    return {
        "original": original,
        "grayscale": gray,
        "gaussian": gaussian,
        "median": median,
        "mse_gauss": mse_gauss,
        "mse_median": mse_median
    }

def show_results(results):
    cv2.imshow("Original", results["original"])
    cv2.imshow("Grayscale", results["grayscale"])
    cv2.imshow("Gaussian", results["gaussian"])
    cv2.imshow("Median", results["median"])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test Example
    # Ensure you have a sample image in your data folder
    test_path = "data/chest_xray/train/NORMAL/IM-0115-0001.jpeg"
    
    if os.path.exists(test_path):
        print(f"Starting preprocessing for: {test_path}")
        results = preprocess_image(test_path)
        
        if results:
            print("Pipeline completed successfully.")
            print(f"MSE (Original vs Gaussian): {results['mse_gauss']:.2f}")
            print(f"MSE (Original vs Median): {results['mse_median']:.2f}")
            
            # Note: In a production environment, you would save these 
            # or pass them to the next module (Harris/SIFT).
            show_results(results)
    else:
        print(f"Test file not found at {test_path}. Please check the path.")

