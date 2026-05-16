# Telco Churn Prediction using K-Nearest Neighbors (KNN)

This repository contains a Jupyter Notebook implementing a **K-Nearest Neighbors (KNN)** classification model to predict customer churn using a telecommunications dataset. 

## Overview

Predicting customer churn is vital for telecom companies to identify at-risk customers and implement retention strategies. This project walks through a complete machine learning workflow—from data cleaning and feature engineering to model training and evaluation—using the distance-based KNN algorithm.

---

## Machine Learning Workflow

The notebook follows a systematic machine learning pipeline:

### 1. Library Dependencies
The project uses standard Python data science and machine learning libraries:
* `pandas` for data manipulation and loading.
* `scikit-learn` for preprocessing, data splitting, modeling, and evaluation metrics.

### 2. Data Preprocessing
* **Target Encoding**: Converts the categorical target variable `Churn` from text (`Yes`/`No`) to binary numeric format (`1`/`0`).
* **Categorical Feature Encoding**: Utilizes **One-Hot Encoding** (`pd.get_dummies`) to transform all categorical features into numeric columns.
* **Missing Value Handling**: Drops rows containing null values to prevent model training failures.

### 3. Data Splitting
The processed dataset is split into training and testing subsets using an **80/20 split ratio** to ensure reliable performance validation.
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### 4. Feature Scaling
Because KNN computes distance metrics to locate neighbors, all independent features are standardized using `StandardScaler` to ensure they share a common scale.

### 5. KNN Model Implementation
* **Hyperparameter Selection**: The model is initialized with $K = 5$ neighbors.
* **Training Characteristic**: KNN is a lazy learner; the `.fit()` step simply stores the training dataset vectors without solving explicit algebraic equations.

### 6. Performance Evaluation
The model is evaluated on the 20% holdout test set across multiple metrics:
* Accuracy
* Precision
* Recall
* Confusion Matrix

---

## Baseline Results

The model yields the following performance metrics on the test data:

| Metric | Score / Output |
| :--- | :--- |
| **Accuracy** | 86.66% |
| **Precision** | 50.00% |
| **Recall** | 1.12% |
| **Confusion Matrix** | `[[577, 1], [88, 1]]` |

**Critical Insight:** While the overall Accuracy looks high (86.66%), the **Recall is catastrophically low (1.12%)**. Out of 89 actual churning customers in the test set, the model only successfully caught **1**. This behavior is driven by a severe class imbalance in the underlying dataset (2,850 non-churners vs. 483 churners).

## Author

**Pereruan Nabalala**

Software Engineer | Machine Learning Enthusiast

GitHub:  
https://github.com/pereruannabaala

---
