import os
import cv2
import numpy as np
from src.preprocessing import preprocess_image
from src.segmentation import segmentation_pipeline, kmeans_mask
from src.features import extract_features, extract_sift_features
from src.classification import train_model, predict_image, train_naive_bayes, train_adaboost


# =========================
# Build Dataset
# =========================
def build_dataset(base_dir, method="threshold", feature_type="old"):

    X = []
    y = []

    for label_name, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:

        folder = os.path.join(base_dir, label_name)

        for img_name in os.listdir(folder):

            path = os.path.join(folder, img_name)

            res = preprocess_image(path)
            if res is None:
                continue

            # =========================
            # Segmentation
            # =========================
            if method == "threshold":
                seg = segmentation_pipeline(res["original"], res["grayscale"])
                mask = seg["mask"]

            elif method == "kmeans":
                mask = kmeans_mask(res["grayscale"])

            else:
                raise ValueError("Unknown segmentation method")

            # =========================
            # Feature Extraction
            # =========================
            if feature_type == "old":
                features = extract_features(res["grayscale"], mask)

            elif feature_type == "sift":
                 features = extract_sift_features(res["grayscale"], mask)

            elif feature_type == "combined":
                  f1 = extract_features(res["grayscale"], mask)
                  f2 = extract_sift_features(res["grayscale"], mask)

                  features = np.concatenate([f1, f2])
           
            else:
                raise ValueError("Unknown feature type")

            X.append(features)
            y.append(label)

    return X, y


# =========================
# MAIN
# =========================
base_dir = "data/chest_xray/train"


#  Threshold Segmentation
print("\n===== THRESHOLD SEGMENTATION =====")
X, y = build_dataset(base_dir, method="threshold", feature_type="old")
print("Dataset size:", len(X))

print("Training Threshold Model...")
model, scaler = train_model(X, y)


#  KMeans Segmentation
print("\n===== KMEANS SEGMENTATION =====")
X_k, y_k = build_dataset(base_dir, method="kmeans", feature_type="old")
print("Dataset size:", len(X_k))

print("Training KMeans Model...")
model_k, scaler_k = train_model(X_k, y_k)


# =========================
# FEATURE COMPARISON 
# =========================

#  Old Features
print("\n===== OLD FEATURES (Threshold) =====")
X_old, y_old = build_dataset(base_dir, method="threshold", feature_type="old")
model_old, scaler_old = train_model(X_old, y_old)

#  SIFT Features
print("\n===== SIFT FEATURES (Threshold) =====")
X_sift, y_sift = build_dataset(base_dir, method="threshold", feature_type="sift")
model_sift, scaler_sift = train_model(X_sift, y_sift)

#  COMBINED Features
print("\n===== COMBINED FEATURES (Threshold) =====")

X_comb, y_comb = build_dataset(base_dir, method="threshold", feature_type="combined")

model_comb, scaler_comb = train_model(X_comb, y_comb)



# =========================
# NAIVE BAYES 
# =========================

print("\n===== NAIVE BAYES (Threshold + Old Features) =====")

X_nb, y_nb = build_dataset(base_dir, method="threshold", feature_type="old")

model_nb, scaler_nb = train_naive_bayes(X_nb, y_nb)


# =========================
# ADABOOST 
# =========================

print("\n===== ADABOOST (Threshold + Old Features) =====")

X_ab, y_ab = build_dataset(base_dir, method="threshold", feature_type="old")

model_ab, scaler_ab = train_adaboost(X_ab, y_ab)
# =========================
# Visualization
# =========================
def show_pipeline(path):

    res = preprocess_image(path)
    if res is None:
        print("Error loading image")
        return

    original = res["original"]
    gray = res["grayscale"]

    # Threshold
    seg = segmentation_pipeline(original, gray)

    # KMeans
    k_mask = kmeans_mask(gray)

    cv2.imshow("Original", original)
    cv2.imshow("Gaussian", res["gaussian"])
    cv2.imshow("Median", res["median"])

    cv2.imshow("Threshold Mask", seg["mask"])
    cv2.imshow("Threshold Overlay", seg["visualization"])

    cv2.imshow("KMeans Mask", k_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# Test Image
# =========================
test_img = r"F:\Third-Year-Second-Term\CV\Project\Medical_Image_ Analysis _Assistant\Data\chest_xray\val\NORMAL\NORMAL2-IM-1442-0001.jpeg "

show_pipeline(test_img)


# =========================
# Final Prediction (BEST MODEL)
# =========================
print("\n===== FINAL PREDICTION (Threshold + Old Features) =====")

result = predict_image(
    model,       
    scaler,
    test_img,
    preprocess_image,
    segmentation_pipeline,
    extract_features   
)

print("Final Prediction:", result)