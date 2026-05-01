# Medical Image Analysis Assistant

### Pneumonia Detection from Chest X-rays using Machine Learning

---

## Overview

This project presents an AI-based system for analyzing chest X-ray images to detect Pneumonia.
It follows a complete computer vision pipeline including preprocessing, segmentation, feature extraction, and classification.

The system acts as a supportive diagnostic tool to assist doctors in early detection.

---

## Objectives

* Enhance medical images (noise removal and contrast improvement)
* Extract lung region (ROI)
* Analyze texture and structural patterns
* Classify images into:

  * Normal
  * Pneumonia
* Compare multiple techniques and models

---

## Project Pipeline

```
Image → Preprocessing → Segmentation → Feature Extraction → Classification → Evaluation
```

---

## 1. Preprocessing

Techniques used:

* Resize images (256×256)
* Convert to grayscale
* Gaussian Filtering
* Median Filtering
* CLAHE (Contrast Enhancement)

Results:

* Noise reduction
* Improved contrast
* Better input for segmentation

---

## 2. Segmentation

Two segmentation methods were implemented and compared:

### Threshold-Based Segmentation (Selected)

* Otsu Thresholding
* Adaptive Thresholding
* Morphological operations
* Contour filtering (top 2 lung regions)

### K-Means Segmentation

* Pixel clustering based on intensity

### Comparison

| Method    | Performance |
| --------- | ----------- |
| Threshold | Better      |
| KMeans    | Lower       |

The threshold-based approach was selected for the final pipeline.

---

## 3. Feature Extraction

### Statistical Features

* Mean
* Standard Deviation
* Skewness
* Kurtosis
* Area

### Texture Features (LBP)

* LBP Mean
* LBP Std

### Edge Features

* Edge Density (Canny)

### SIFT Features

* Number of keypoints
* Descriptor statistics

---

## Feature Strategy Comparison

| Feature Type                 | Performance                |
| ---------------------------- | -------------------------- |
| Statistical + Texture + Edge | Best                       |
| SIFT Only                    | Weak                       |
| Combined                     | No significant improvement |

Texture-based features showed the highest importance.

---

## 4. Classification Models

Models implemented and compared:

| Model         | Performance |
| ------------- | ----------- |
| Random Forest | Best        |
| AdaBoost      | Good        |
| Naive Bayes   | Lower       |

---

## 5. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve (AUC)

---

## Confusion Matrix Analysis

Special focus was placed on False Negatives (FN):

* False Negative = Pneumonia case predicted as Normal

Minimizing FN is critical in medical diagnosis.
Random Forest achieved the lowest FN among all models.

---

## ROC Curve

* AUC = 0.94
* Strong class separation
* High True Positive Rate with controlled False Positive Rate

---

## Results

### Best Model

Threshold Segmentation + Statistical/Texture Features + Random Forest

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 91%   |
| Precision | 95%   |
| Recall    | 93%   |
| AUC       | 0.94  |

---

## Feature Analysis

Key findings:

* Texture features (LBP) are the most important
* Edge density contributes to detecting infection patterns
* Features are correlated, which affects some models

### Insight

Pneumonia detection depends more on texture changes than raw intensity.

---

## Model Behavior

* Random Forest handles correlated features effectively
* Naive Bayes assumes independence, leading to lower performance
* SIFT is not suitable for this classification task

---

## Project Structure

```
project/
│
├── data/
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── features.py
│   ├── classification.py
│   ├── sift.py
│   ├── harris.py
│   ├── pyramid.py
│
├── notebooks/
│   └── main.ipynb
│
├── main.py
└── README.md
```

---

## How to Use the Project

### 1. Clone the Repository

```
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
```

---

### 2. Install Requirements

```
pip install -r requirements.txt
```

---

### 3. Prepare Dataset

Place dataset in:

```
data/chest_xray/
    ├── train/
    ├── val/
    └── test/
```

---

### 4. Run the Project

#### Option 1 (Recommended)

```
jupyter notebook notebooks/main.ipynb
```

This will display:

* preprocessing steps
* segmentation results
* model evaluation
* visualizations (ROC, confusion matrix)

---

#### Option 2 (Script)

```
python main.py
```

---

### 5. Test on New Image

Modify the image path:

```
test_img = "path_to_image"
```

Then run prediction.

---

## Outputs

The project produces:

* Preprocessing visualizations
* Segmentation comparison
* Feature analysis
* Model evaluation metrics
* Confusion matrix
* ROC curve
* Final prediction

---

## Future Improvements

* Deep learning models (CNN, ResNet)
* U-Net segmentation
* Data augmentation
* Larger datasets

---

## Key Takeaways

* Segmentation quality impacts model performance
* Texture features are critical in medical imaging
* Random Forest performs well with complex features
* Recall is essential in medical applications
