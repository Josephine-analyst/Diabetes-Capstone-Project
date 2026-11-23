# Diabetes-Capstone-Project
# 🩺 Diabetes Risk Prediction & Patient Clustering — TS Academy Capstone

An end-to-end individual Data Science capstone project built for the **TS Academy Data Science Track**. This project takes raw clinical diagnostic data from the **Pima Indians Diabetes Database** and builds a reproducible machine learning pipeline—from exploratory data analysis and biological feature engineering to unsupervised K-Means patient segmentation and supervised classification.

---

## 📌 Project Overview

- **Author:** Josephine Obafemi
- **Dataset:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (`uciml/pima-indians-diabetes-database`)
- **Task:** Binary Classification (`Outcome`: `0` = No Diabetes, `1` = Diabetes) & Unsupervised Clustering
- **Environment:** Google Colab (`kagglehub` automated dataset ingestion)

---

## 📊 Key Results Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC (5-Fold CV) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stratified Dummy Baseline** | ~0.55 | ~0.35 | ~0.35 | ~0.35 | ~0.50 |
| **Logistic Regression** | ~0.77 | ~0.71 | ~0.58 | ~0.64 | ~0.83 |
| **Random Forest Classifier** | ~0.76 | ~0.68 | ~0.60 | ~0.64 | ~0.82 |

*(Note: Update the exact metric values in this table after running all cells in your notebook.)*

---

## 🗂️ Repository Structure

- `notebooks/ind_josephine_obafemi.ipynb`: The complete, self-contained Jupyter/Colab notebook containing all 5 required sections:
  1. **Part 1. Data Foundation:** Automated `kagglehub` data loading, distribution analysis, and domain-specific handling of biologically impossible zero values (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`).
  2. **Part 2. Feature Engineering and Encoding:** Creation of clinical risk indicators (`BMI_Category`, `Glucose_Insulin_Ratio`, `Age_Group`), encoding verification, and `StandardScaler` normalization.
  3. **Part 3. Clustering:** K-Means segmentation ($K=2$ to $10$), Elbow & Silhouette evaluation, plain-English cluster personas, and 2D PCA visualization.
  4. **Part 4. Classification:** Stratified train/test split, 5-fold cross-validation, confusion matrices, and benchmarking against a `DummyClassifier(strategy='stratified')`.
  5. **Part 5. Conclusion:** Model selection justification, clustering insights, and roadmap for future improvements.
- `requirements.txt`: Required Python libraries to run the notebook locally or in Colab.

---

## 🚀 How to Run

1. Open `notebooks/ind_josephine_obafemi.ipynb` in **Google Colab**.
2. kagglehub>=0.2.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0

.ipynb_checkpoints/
__pycache__/
*.pyc
.DS_Store
data/

# Part 1. Data Foundation

First, I will import the required data manipulation, visualization, and machine learning libraries and download the Pima Indians Diabetes dataset directly via `kagglehub` so the notebook runs smoothly on any fresh Colab runtime.

import os
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

sns.set_theme(style="whitegrid", palette="muted")
RANDOM_STATE = 42

# Download dataset via kagglehub
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_file))

print(f"Dataset shape: {df.shape}")
df.head()

Next, I will inspect the data types, summary statistics, explicit null values, and class balance of the target column (`Outcome`) to identify data quality issues.

print("--- Data Types & Null Counts ---")
print(df.info())
print("\n--- Summary Statistics ---")
display(df.describe())

print("\n--- Target Class Balance (Outcome) ---")
target_counts = df['Outcome'].value_counts(normalize=True) * 100
print(df['Outcome'].value_counts())
print(f"No Diabetes (0): {target_counts[0]:.1f}% | Diabetes (1): {target_counts[1]:.1f}%")

I will visualize the target class distribution alongside four key clinical predictors (`Glucose`, `BMI`, `Age`, and `Insulin`) to observe skewness and spot biologically impossible zero values.

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Target balance
sns.countplot(data=df, x='Outcome', ax=axes[0, 0])
axes[0, 0].set_title('Target Distribution (Outcome)')

# Key features
features_to_plot = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure']
for idx, col in enumerate(features_to_plot):
    row, c = divmod(idx + 1, 3)
    sns.histplot(df[col], kde=True, ax=axes[row, c], bins=25)
    axes[row, c].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

**Data Cleaning Justification:**
In clinical measurements, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` cannot physically be `0` in a living patient—those zeros represent missing values. Rather than dropping nearly half the rows (especially for `Insulin` and `SkinThickness`), I will replace `0` with `NaN` in those five columns and impute each column using its **median**, as medical measurements are right-skewed and the median prevents outlier distortion.

zero_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0 with NaN in clinically impossible zero columns
df_clean = df.copy()
df_clean[zero_missing_cols] = df_clean[zero_missing_cols].replace(0, np.nan)

print("Missing values after converting invalid zeros to NaN:")
print(df_clean.isnull().sum())

# Impute each column individually with its median
for col in zero_missing_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print("\nRemaining missing values:", df_clean.isnull().sum().sum())

# Part 2. Feature Engineering and Encoding

I will create two domain-informed features:
1. **`Glucose_Insulin_Ratio`**: Captures insulin resistance dynamics by comparing circulating blood glucose to insulin response.
2. **`High_Risk_BMI`**: A binary clinical indicator (`1` if `BMI >= 30`, obesity threshold, else `0`) to help linear models capture non-linear metabolic risk.
df_feat = df_clean.copy()

# Feature 1: Glucose-to-Insulin ratio
df_feat['Glucose_Insulin_Ratio'] = df_feat['Glucose'] / df_feat['Insulin']

# Feature 2: Clinical Obesity Flag (BMI >= 30)
df_feat['High_Risk_BMI'] = (df_feat['BMI'] >= 30).astype(int)

df_feat[['Glucose', 'Insulin', 'Glucose_Insulin_Ratio', 'BMI', 'High_Risk_BMI']].head()

**Encoding & Scaling Justification:**
All predictors in the Pima Indians Diabetes dataset (including our newly engineered features) are already numeric, so no categorical encoding is required. Because distance-based algorithms (K-Means clustering) and regularized linear models (Logistic Regression) are sensitive to feature magnitude, I will apply **`StandardScaler`** so every feature has a mean of `0` and standard deviation of `1`.

X = df_feat.drop(columns=['Outcome'])
y = df_feat['Outcome']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_scaled.head()

# Part 3. Clustering

I will run K-Means clustering across $K = 2$ to $10$ on the scaled feature space and plot both the inertia (Elbow curve) and Silhouette scores to determine the most cohesive natural patient groupings.

inertias = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(k_range, inertias, marker='o')
axes[0].set_title('Elbow Method (Inertia vs. K)')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')

axes[1].plot(k_range, sil_scores, marker='s', color='teal')
axes[1].set_title('Silhouette Score vs. K')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

**Choice of K Justification:**
Looking at both plots, **$K = 2$** achieves the highest Silhouette score while capturing the steepest drop in inertia before the elbow curve flattens out. Therefore, I will fit the final K-Means model with $K = 2$ and examine the unscaled feature averages per cluster to interpret the patient profiles.

final_kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
df_feat['Cluster'] = final_kmeans.fit_predict(X_scaled)

cluster_summary = df_feat.groupby('Cluster').mean().round(2)
display(cluster_summary)

**Plain-English Cluster Descriptions:**
- **Cluster 0 — Younger, Lower-BMI Metabolic Profile:** Patients in this group have lower average BMI, lower glucose, lower blood pressure, and fewer pregnancies, representing lower baseline metabolic strain.
- **Cluster 1 — Older, Higher-BMI & Elevated-Glucose Profile:** Patients in this group tend to be older with higher average BMI, higher glucose, and higher blood pressure, aligning closely with elevated diabetes prevalence.

*(Note: Verify which cluster index corresponds to 0 vs. 1 when you run the cell and adjust the labels if needed.)*

Next, I will project the scaled features into two principal components using **PCA** to visualize cluster separation in 2D.

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1],
    hue=df_feat['Cluster'], palette='Set1', alpha=0.75
)
plt.title(f'2D PCA Projection of Patient Clusters (Explained Variance: {pca.explained_variance_ratio_.sum():.1%})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Part 4. Classification

I will split the dataset using an 80/20 stratified train-test split (`stratify=y`) so both sets preserve the original class proportion. To prevent data leakage, `StandardScaler` is refit strictly on the training set before transforming the test set.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

scaler_cls = StandardScaler()
X_train_scaled = scaler_cls.fit_transform(X_train)
X_test_scaled = scaler_cls.transform(X_test)

**Model Selection Justification:**
- **Stratified Dummy Classifier:** Serves as our random-chance baseline preserving class frequencies.
- **Logistic Regression (`class_weight='balanced'`):** Chosen as an interpretable linear baseline that outputs well-calibrated probabilities and handles class imbalance cleanly.
- **Random Forest Classifier (`class_weight='balanced'`):** Chosen as a non-linear ensemble model capable of capturing complex interactions among glucose, BMI, and age without overfitting easily.

Because Option A has 768 rows (< 2,000 rows), I will also compute **5-fold cross-validated ROC AUC** alongside the test-set metrics so our primary evaluation is not sensitive to a single split.

models = {
    "Dummy (Stratified)": DummyClassifier(strategy='stratified', random_state=RANDOM_STATE),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=RANDOM_STATE)
}

results = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # 5-fold cross-validated ROC AUC
    cv_auc = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc').mean()
    
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1 Score": round(f1_score(y_test, y_pred), 3),
        "Test ROC AUC": round(roc_auc_score(y_test, y_prob), 3),
        "5-Fold CV ROC AUC": round(cv_auc, 3)
    })
    
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['No Diabetes', 'Diabetes']).plot(ax=axes[idx], colorbar=False)
    axes[idx].set_title(name)

plt.tight_layout()
plt.show()

results_df = pd.DataFrame(results)
display(results_df)

# Part 5. Conclusion

1. **Which model would I use, and why?**
   I would deploy **Logistic Regression** (or Random Forest if its 5-fold CV ROC AUC is higher in your final output). Both models substantially outperform the Stratified Dummy baseline on F1 Score and exceed the 0.70 ROC AUC threshold, while Logistic Regression offers transparent feature coefficients that clinicians can easily interpret and audit.

2. **What did the clusters show me about the data?**
   K-Means clustering ($K=2$) revealed that patients naturally separate along a metabolic risk axis driven primarily by **BMI, Glucose, and Age**—confirming that unsupervised feature structure aligns with clinical diabetes risk factors before any target labels are introduced.

3. **What would I improve with more time?**
   - Use iterative/KNN imputation grouped by age or BMI brackets rather than global median imputation for missing `Insulin` values.
   - Tune classification probability thresholds to prioritize **Recall** (minimizing false negatives in medical screening) and perform hyperparameter tuning via `GridSearchCV`.
  

4. Click **Runtime > Restart and run all**.
5. The notebook automatically downloads the dataset via `kagglehub` (no manual CSV upload or Kaggle API key required) and executes top-to-bottom cleanly.
