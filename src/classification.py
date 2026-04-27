from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

import numpy as np
# def train_model(X, y):

#     # =========================
#     # Split
#     # =========================
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y
#     )

#     print("Train size:", len(X_train))
#     print("Test size :", len(X_test))

#     # =========================
#     # Scaling 
#     # =========================
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # =========================
#     # Train 
#     # =========================
#     model = RandomForestClassifier(
#         n_estimators=100,
#         class_weight="balanced",
#         random_state=42,
#         n_jobs=-1   # يستخدم كل الـ CPU
#     )

#     model.fit(X_train, y_train)

#     # =========================
#     # Predict
#     # =========================
#     y_pred = model.predict(X_test)

#     # =========================
#     # Metrics
#     # =========================
#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred)
#     rec = recall_score(y_test, y_pred)

#     print("\n===== Evaluation =====")
#     print(f"Accuracy : {acc:.4f}")
#     print(f"Precision: {prec:.4f}")
#     print(f"Recall   : {rec:.4f}")

#     # =========================
#     # Confusion Matrix
#     # =========================
#     cm = confusion_matrix(y_test, y_pred)
#     print("\nConfusion Matrix:\n", cm)

#     # =========================
#     # Classification Report
#     # =========================
#     print("\nClassification Report:\n", classification_report(y_test, y_pred))

#     return model



# def predict_image(model, path, preprocess_image, segmentation_pipeline, extract_features):

#     # 1. preprocessing
#     res = preprocess_image(path)
#     if res is None:
#         return "Error loading image"

#     # 2. segmentation
#     seg = segmentation_pipeline(res["original"], res["grayscale"])

#     # 3. feature extraction
#     feat = extract_features(res["grayscale"], seg["mask"])

#     # 4. prediction
#     pred = model.predict([feat])

#     return "Pneumonia" if pred[0] == 1 else "Normal"

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 🔥 prediction
    y_pred = model.predict(X_test)

    # 🔥 metrics
    print("\n===== Evaluation =====")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, scaler



def predict_image(model, scaler, path, preprocess_image, segmentation_pipeline, extract_features):
    res = preprocess_image(path)
    if res is None: return "Error loading image"

    seg = segmentation_pipeline(res["original"], res["grayscale"])
    feat = extract_features(res["grayscale"], seg["mask"])

    # تحويل الميزات إلى مصفوفة وتطبيق الـ Scaler (مهم جداً!)
    feat_array = np.array([feat])
    feat_scaled = scaler.transform(feat_array)

    pred = model.predict(feat_scaled)

    return "Pneumonia" if pred[0] == 1 else "Normal"