"""
Employee Performance Predictor - Model Training & Evaluation Pipeline
----------------------------------------------------------------------
Author: Data Science Team
Description:
    Trains a production-ready scikit-learn Pipeline (ColumnTransformer +
    RandomForestClassifier) to predict employee performance bands. Also fits
    a baseline Logistic Regression model for interview-ready benchmarking.

CRITICAL ETHICAL & SAFETY DISCLAIMER:
    1. SYNTHETIC DATA NOTICE: This model is trained on 100% synthetic data
       generated for educational and portfolio demonstration purposes. It does
       NOT establish real-world predictive validity for workplace personnel.
    2. ADVISORY USE ONLY: Machine learning predictions should NEVER be used as
       an autonomous decision-maker for hiring, promotion, compensation, or
       termination. Human-in-the-loop review and bias audits are mandatory.
    3. LEAKAGE PREVENTION: The continuous 'performance_score' and unique
       'employee_id' are strictly dropped from input features to prevent
       artificial target leakage.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_and_evaluate(
    data_path: str = "data/employee_performance.csv",
    model_save_path: str = "models/employee_performance_model.joblib",
    reports_dir: str = "reports"
) -> dict:
    """
    Train and evaluate classification models on the employee dataset.

    Parameters:
    -----------
    data_path : str
        Path to the synthetic employee dataset CSV.
    model_save_path : str
        Target file path for the exported joblib model pipeline.
    reports_dir : str
        Directory where evaluation reports and figures will be saved.

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and test artifacts.
    """
    print("=" * 70)
    print("EMPLOYEE PERFORMANCE PREDICTOR - MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Ensure required output directories exist
    model_file = Path(model_save_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    report_folder = Path(reports_dir)
    report_folder.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_file}. Please run src/generate_data.py first!"
        )

    df = pd.read_csv(data_file)
    print(f"[OK] Dataset loaded successfully: {df.shape[0]} rows x {df.shape[1]} columns")

    # 3. Target and Feature Separation (Preventing Target Leakage)
    # Target: performance_band ('Low', 'Medium', 'High')
    target_col = "performance_band"
    leakage_cols = ["employee_id", "performance_score", target_col]

    # Features (X) exclude identifiers and the score from which the band was derived
    X = df.drop(columns=leakage_cols)
    y = df[target_col]

    # Explicit class order for consistent reporting and matrix alignment
    class_labels = ["Low", "Medium", "High"]

    numeric_features = [
        "years_experience",
        "training_hours",
        "attendance_rate",
        "projects_completed",
        "manager_feedback_score",
    ]
    categorical_features = ["department"]

    print("\nFeature Specification:")
    print(f" - Categorical Features ({len(categorical_features)}): {categorical_features}")
    print(f" - Numerical Features ({len(numeric_features)}): {numeric_features}")
    print(f" - Excluded Features (Leakage Prevention): ['employee_id', 'performance_score']")
    print(f" - Target Variable: '{target_col}' -> Classes: {class_labels}")

    # 4. Train-Test Split (80% Train, 20% Test, Stratified by Target)
    # Stratification ensures each split maintains the ~19% Low, ~56% Medium, ~25% High ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nData Split (80/20 Stratified):")
    print(f" - Training Set: {X_train.shape[0]} samples")
    print(f" - Test Set:     {X_test.shape[0]} samples (Kept strictly hold-out)")

    # 5. Build Preprocessing Pipeline with ColumnTransformer
    # - Numeric: StandardScaler scales features to mean=0 and variance=1
    # - Categorical: OneHotEncoder creates dummy columns, ignores unseen categories
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop"
    )

    # 6. Primary Model: Random Forest Classifier Pipeline
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf_classifier),
        ]
    )

    # 7. Baseline Model: Logistic Regression Pipeline (for comparison)
    log_reg_classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )

    log_reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", log_reg_classifier),
        ]
    )

    # 8. Model Training (Fitting ONLY on the training set)
    print("\nFitting models on training data...")
    rf_pipeline.fit(X_train, y_train)
    log_reg_pipeline.fit(X_train, y_train)
    print("[OK] Models fitted successfully.")

    # 9. Model Evaluation on Hold-Out Test Set
    # Predictions
    rf_preds = rf_pipeline.predict(X_test)
    log_reg_preds = log_reg_pipeline.predict(X_test)

    # Calculate exact performance metrics
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_bal_acc = balanced_accuracy_score(y_test, rf_preds)

    lr_acc = accuracy_score(y_test, log_reg_preds)
    lr_bal_acc = balanced_accuracy_score(y_test, log_reg_preds)

    # 10. Display Results and Benchmark Table
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON (TEST SET EVALUATION)")
    print("=" * 70)
    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression (Baseline)", "Random Forest Classifier"],
        "Accuracy": [f"{lr_acc * 100:.2f}%", f"{rf_acc * 100:.2f}%"],
        "Balanced Accuracy": [f"{lr_bal_acc * 100:.2f}%", f"{rf_bal_acc * 100:.2f}%"],
    })
    print(comparison_df.to_string(index=False))

    # Detailed Classification Report for Random Forest
    rf_report_text = classification_report(
        y_test,
        rf_preds,
        labels=class_labels,
        digits=4
    )
    print("\n" + "-" * 50)
    print("RANDOM FOREST CLASSIFICATION REPORT:")
    print("-" * 50)
    print(rf_report_text)

    # Confusion Matrix for Random Forest
    rf_cm = confusion_matrix(y_test, rf_preds, labels=class_labels)
    cm_df = pd.DataFrame(rf_cm, index=class_labels, columns=class_labels)
    print("CONFUSION MATRIX (Actual vs. Predicted):")
    print(cm_df)

    # 11. Save Evaluation Artifacts in reports/
    # A. Save classification report as a text report
    report_file_path = report_folder / "classification_report.txt"
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write("EMPLOYEE PERFORMANCE PREDICTOR - EVALUATION REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write("MODEL COMPARISON:\n")
        f.write(comparison_df.to_string(index=False) + "\n\n")
        f.write("RANDOM FOREST DETAILED CLASSIFICATION REPORT:\n")
        f.write("-" * 55 + "\n")
        f.write(rf_report_text + "\n\n")
        f.write("CONFUSION MATRIX (Actual rows, Predicted cols):\n")
        f.write(cm_df.to_string() + "\n\n")
        f.write("ETHICAL NOTICE:\n")
        f.write(
            "This model was trained on synthetic data for demonstration.\n"
            "It must not be used for autonomous employee employment decisions.\n"
        )
    print(f"\n[OK] Classification report saved to: {report_file_path}")

    # B. Generate and save styled Confusion Matrix heatmap plot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=1.0,
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax
    )
    ax.set_title("Confusion Matrix - Random Forest Classifier", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Performance Band", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual Performance Band", fontsize=11, labelpad=8)
    plt.tight_layout()

    cm_image_path = report_folder / "confusion_matrix.png"
    plt.savefig(cm_image_path, dpi=200)
    plt.close()
    print(f"[OK] Confusion matrix plot saved to: {cm_image_path}")

    # 12. Save Trained Random Forest Pipeline with joblib
    joblib.dump(rf_pipeline, model_file)
    print(f"[OK] Champion model pipeline successfully serialized to: {model_file.resolve()}")

    print("\n" + "=" * 70)
    print("TRAINING & EVALUATION COMPLETE!")
    print("=" * 70)

    return {
        "rf_accuracy": rf_acc,
        "rf_balanced_accuracy": rf_bal_acc,
        "lr_accuracy": lr_acc,
        "lr_balanced_accuracy": lr_bal_acc,
        "pipeline": rf_pipeline,
    }


if __name__ == "__main__":
    train_and_evaluate(
        data_path="data/employee_performance.csv",
        model_save_path="models/employee_performance_model.joblib",
        reports_dir="reports"
    )
