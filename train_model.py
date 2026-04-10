# -*- coding: utf-8 -*-
"""
Diabetes Disease Prediction System - Model Training
"""

# Basic libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML tools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay

# Model saving
import joblib
import json

# Load dataset
print("Loading dataset...")
db = pd.read_csv('diabetes_data_upload.csv')
print(f"Dataset shape: {db.shape}")
print("\nFirst few rows:")
print(db.head())

# Basic EDA
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)
print("\nDataset Info:")
print(db.info())
print("\nSummary Statistics:")
print(db.describe())
print("\nMissing Values:")
print(db.isnull().sum())
print("\nDuplicate Values:", db.duplicated().sum())

# Remove duplicates
db = db.drop_duplicates()
print(f"\nDataset shape after removing duplicates: {db.shape}")

# Class distribution
print("\nClass Distribution:")
print(db['class'].value_counts())

# Visualizations
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(x='class', data=db)
plt.title("Diabetes Class Distribution")

plt.subplot(1, 2, 2)
plt.hist(db['Age'], bins=20, edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig('eda_plots.png')
print("\nEDA plots saved as 'eda_plots.png'")

# Prepare for correlation heatmap
db_encoded = db.copy()
label_encoders = {}

for col in db_encoded.columns:
    le = LabelEncoder()
    db_encoded[col] = le.fit_transform(db_encoded[col])
    label_encoders[col] = le

# Correlation heatmap
plt.figure(figsize=(19, 10))
sns.heatmap(db_encoded.corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Correlation heatmap saved as 'correlation_heatmap.png'")

# Feature Engineering - Encode categorical variables
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

db_processed = db.copy()

# Store label encoders for later use
feature_encoders = {}

for col in db_processed.columns:
    if col != 'class':
        le = LabelEncoder()
        db_processed[col] = le.fit_transform(db_processed[col])
        feature_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
db_processed['class'] = target_encoder.fit_transform(db_processed['class'])

print("\nEncoded dataset:")
print(db_processed.head())

# Split features and target
X = db_processed.drop('class', axis=1)
y = db_processed['class']

# Train-test split
print("\n" + "="*50)
print("TRAIN-TEST SPLIT")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================
# MODEL 1: LOGISTIC REGRESSION
# ========================================
print("\n" + "="*50)
print("LOGISTIC REGRESSION MODEL")
print("="*50)

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_log):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

# ========================================
# MODEL 2: RANDOM FOREST
# ========================================
print("\n" + "="*50)
print("RANDOM FOREST MODEL")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# ========================================
# HYPERPARAMETER TUNING
# ========================================
print("\n" + "="*50)
print("HYPERPARAMETER TUNING")
print("="*50)

params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    params,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("\nPerforming GridSearchCV...")
grid.fit(X_train, y_train)

print(f"\nBest parameters: {grid.best_params_}")
print(f"Best cross-validation score: {grid.best_score_:.4f}")

best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)
y_prob_best = best_rf.predict_proba(X_test)[:, 1]

print("\nTuned Random Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_best):.4f}")

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test)
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_best)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_prob_best):.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png')
print("ROC curve saved as 'roc_curve.png'")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title("Top 10 Most Important Features")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")

# ========================================
# SAVE MODELS AND ENCODERS
# ========================================
print("\n" + "="*50)
print("SAVING MODELS")
print("="*50)

# Save models
joblib.dump(log_model, "logistic_model.pkl")
joblib.dump(best_rf, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save encoders
joblib.dump(feature_encoders, "feature_encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

# Save feature names
feature_names = list(X.columns)
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# Save model metadata
metadata = {
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_shape': db.shape,
    'train_size': X_train.shape[0],
    'test_size': X_test.shape[0],
    'logistic_accuracy': float(accuracy_score(y_test, y_pred_log)),
    'logistic_roc_auc': float(roc_auc_score(y_test, y_prob_log)),
    'rf_accuracy': float(accuracy_score(y_test, y_pred_best)),
    'rf_roc_auc': float(roc_auc_score(y_test, y_prob_best)),
    'best_params': grid.best_params_
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("\n✓ Models saved successfully:")
print("  - logistic_model.pkl")
print("  - random_forest_model.pkl")
print("  - scaler.pkl")
print("  - feature_encoders.pkl")
print("  - target_encoder.pkl")
print("  - feature_names.json")
print("  - model_metadata.json")

print("\n" + "="*50)
print("TRAINING COMPLETED!")
print("="*50)
