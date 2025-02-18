# Schizophrenia Diagnosis using KNN

## Overview
This project applies **K-Nearest Neighbors (KNN)** for schizophrenia diagnosis using a dataset containing patient medical records. The dataset undergoes preprocessing, feature selection, and balancing using **SMOTE** before classification.

## Dataset
- **File**: `schizophrenia_dataset.csv`
- **Features**: Various medical and demographic attributes
- **Target**: `Diagnosis` (Binary classification)

## Workflow
1. Load and preprocess the dataset (handle missing values, one-hot encoding, standardization).
2. Compute correlation to select high-impact features.
3. Apply **SMOTE** to handle class imbalance.
4. Train and evaluate a **KNN classifier**.
5. Generate classification metrics and a confusion matrix.

## Installation & Usage
```bash
pip install -r requirements.txt  # Install dependencies
python knn.py  # Run the model
```

## Results
- Displays feature correlations with the diagnosis.
- Outputs **accuracy, classification report, and confusion matrix**.
- Acheives an accuracy of 99.6%

