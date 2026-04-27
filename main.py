import os
import cv2

# from src.preprocessing import preprocess_image
# from src.harris import harris_pipeline
# from src.pyramid import run_pyramid_pipeline, display_pyramid
# from src.sift import detect_sift_features, draw_keypoints, match_features, draw_matches


# # =========================
# # Paths
# # =========================
# base_dir = os.path.dirname(__file__)

# path1 = os.path.join(
#     base_dir,
#     "data",
#     "chest_xray",
#     "train",
#     "NORMAL",
#     "IM-0115-0001.jpeg"
# )

# path2 = os.path.join(
#     base_dir,
#     "data",
#     "chest_xray",
#     "train",
#     "PNEUMONIA",
#     "person1_bacteria_1.jpeg"
# )


# # =========================
# # Preprocessing Image 1
# # =========================
# res1 = preprocess_image(path1)

# if res1 is None:
#     print("Failed to load NORMAL image")
#     exit()

# original = res1["original"]
# gray = res1["grayscale"]
# gaussian = res1["gaussian"]
# median = res1["median"]


# # =========================
# # Harris
# # =========================
# harris_result = harris_pipeline(original, gray, threshold=0.02)
# print("Number of corners:", harris_result["num_corners"])

# cv2.imshow("Original", original)
# cv2.imshow("Grayscale", gray)
# cv2.imshow("Gaussian", gaussian)
# cv2.imshow("Median", median)
# cv2.imshow("Harris Corners", harris_result["image"])

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # =========================
# # Pyramid
# # =========================
# pyramid_result = run_pyramid_pipeline(original, levels=3)

# print("Pyramid done")

# display_pyramid(pyramid_result["gaussian"], "Gaussian")
# display_pyramid(pyramid_result["laplacian"], "Laplacian")

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # =========================
# # Preprocessing Image 2 (SIFT)
# # =========================
# res2 = preprocess_image(path2)

# if res2 is None:
#     print("Failed to load PNEUMONIA image")
#     exit()

# img1 = res1["original"]
# gray1 = res1["grayscale"]

# img2 = res2["original"]
# gray2 = res2["grayscale"]


# # =========================
# # SIFT
# # =========================
# kp1, des1 = detect_sift_features(gray1)
# kp2, des2 = detect_sift_features(gray2)

# print("Keypoints Image1:", len(kp1))
# print("Keypoints Image2:", len(kp2))

# img_kp1 = draw_keypoints(img1, kp1)
# img_kp2 = draw_keypoints(img2, kp2)


# # =========================
# # Matching
# # =========================
# matches = match_features(des1, des2)
# print("Good matches:", len(matches))

# match_img = draw_matches(img1, kp1, img2, kp2, matches)


# # =========================
# # Display SIFT
# # =========================
# cv2.imshow("SIFT Keypoints 1", img_kp1)
# cv2.imshow("SIFT Keypoints 2", img_kp2)
# cv2.imshow("Matches", match_img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



# from src.segmentation import segmentation_pipeline

# # # =========================
# # # Segmentation
# # # =========================
# # seg_result = segmentation_pipeline(original, gray)

# # print("Segmentation done")

# # cv2.imshow("Threshold", seg_result["threshold"])
# # cv2.imshow("ROI (Largest Region)", seg_result["roi"])
# # cv2.imshow("KMeans Segmentation", seg_result["kmeans"])

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# import cv2



# def main():
#     sample_path = path2

#     img = cv2.imread(sample_path)

#     if img is None:
#         print(f"File not found: {sample_path}")
#         return

#     # Resize
#     img_resized = cv2.resize(img, (512, 512))

#     # Convert to gray
#     gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

#     # Run pipeline
#     results = segmentation_pipeline(img_resized, gray)

#     # Display results
#     cv2.imshow("Original", img_resized)
#     cv2.imshow("Threshold", results["thresholded"])
#     cv2.imshow("Mask", results["mask"])
#     cv2.imshow("Overlay", results["visualization"])

#     print("Segmentation complete.")

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()

from src.preprocessing import preprocess_image
from src.segmentation import segmentation_pipeline
from src.features import extract_features

import os

def build_dataset(base_dir):

    X = []
    y = []

    for label_name, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:

        folder = os.path.join(base_dir, label_name)

        for img_name in os.listdir(folder):

            path = os.path.join(folder, img_name)

            res = preprocess_image(path)
            if res is None:
                continue

            seg = segmentation_pipeline(res["original"], res["grayscale"])

            features = extract_features(res["grayscale"], seg["mask"])

            X.append(features)
            y.append(label)

    return X, y

from src.classification import train_model

base_dir = "data/chest_xray/train"

print("Building dataset...")
X, y = build_dataset(base_dir)

print("Dataset size:", len(X))

print("Training model...")
model ,scaler= train_model(X, y)

print("Done...start the prediction")



from src.classification import predict_image

def show_pipeline(path):

    res = preprocess_image(path)
    if res is None:
        print("Error loading image")
        return

    original = res["original"]
    gray = res["grayscale"]
    gaussian = res["gaussian"]
    median = res["median"]

    seg = segmentation_pipeline(original, gray)

    # عرض
    cv2.imshow("Original", original)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Gaussian", gaussian)
    cv2.imshow("Median", median)

    cv2.imshow("Threshold", seg["thresholded"])
    cv2.imshow("Mask", seg["mask"])
    cv2.imshow("Overlay", seg["visualization"])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
test_img = r"F:\Third-Year-Second-Term\CV\Project\Medical_Image_ Analysis _Assistant\Data\chest_xray\val\PNEUMONIA\person1946_bacteria_4875.jpeg"

show_pipeline(test_img)

result = predict_image(
    model,
    scaler, 
    test_img,
    preprocess_image,
    segmentation_pipeline,
    extract_features
)

print("Final Prediction:", result)



