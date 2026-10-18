"""
Employee Performance Predictor - Exploratory Data Analysis (EDA)
----------------------------------------------------------------
Author: Data Science Team
Description:
    Loads the synthetic employee dataset, computes descriptive statistics,
    and generates visual analytics charts stored in the reports/ folder.

Visualizations Generated:
    1. Performance-band count distribution (Bar Chart)
    2. Average performance score by department (Bar Chart)
    3. Training hours vs. Performance score (Scatter Plot)
    4. Numeric feature correlation matrix (Heatmap)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda(
    data_path: str = "data/employee_performance.csv",
    reports_dir: str = "reports"
) -> None:
    """
    Perform exploratory data analysis and export charts to reports/ folder.

    Parameters:
    -----------
    data_path : str
        Relative or absolute path to the employee performance CSV file.
    reports_dir : str
        Folder where output visualization images will be saved.
    """
    print("=" * 70)
    print("EMPLOYEE PERFORMANCE PREDICTOR - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Ensure target reports directory exists
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Reports directory ready: {output_dir.resolve()}\n")

    # 2. Load dataset
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_file}. Please run src/generate_data.py first!"
        )

    df = pd.read_csv(data_file)
    print(f"[OK] Dataset loaded successfully from: {data_file}\n")

    # 3. Basic Dataset Structure & Quality Inspection
    print("-" * 50)
    print("1. DATASET STRUCTURE & QUALITY CHECKS")
    print("-" * 50)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("Column Names:")
    for col in df.columns:
        print(f" - {col}")
    print()

    print("Data Types:")
    print(df.dtypes)
    print()

    print("Missing Values per Column:")
    missing = df.isnull().sum()
    print(missing)
    print()

    duplicates_count = df.duplicated().sum()
    print(f"Duplicate Rows Count: {duplicates_count}\n")

    # 4. Summary Statistics for Numeric Columns
    print("-" * 50)
    print("2. SUMMARY STATISTICS (NUMERIC COLUMNS)")
    print("-" * 50)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    print(df[numeric_cols].describe().round(2))
    print("\n" + "=" * 70 + "\n")

    # Configure global styling for modern, clean Seaborn plots
    sns.set_theme(style="whitegrid", font="sans-serif")

    # -------------------------------------------------------------
    # Chart 1: Performance-Band Count Distribution
    # -------------------------------------------------------------
    print("[*] Generating Chart 1: Performance-Band Count Distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"Low": "#e74c3c", "Medium": "#f39c12", "High": "#27ae60"}
    order = ["Low", "Medium", "High"]

    sns.countplot(
        data=df,
        x="performance_band",
        order=order,
        palette=palette,
        hue="performance_band",
        legend=False,
        ax=ax
    )

    # Annotate bar heights with exact counts and percentages
    total_rows = len(df)
    for p in ax.patches:
        height = int(p.get_height())
        pct = (height / total_rows) * 100
        ax.annotate(
            f"{height}\n({pct:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, height / 2.0),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold"
        )

    ax.set_title("Employee Performance Band Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Performance Band Tier", fontsize=12, labelpad=8)
    ax.set_ylabel("Employee Count", fontsize=12, labelpad=8)
    plt.tight_layout()

    chart1_path = output_dir / "performance_band_distribution.png"
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    print(f"    Saved: {chart1_path}")

    # -------------------------------------------------------------
    # Chart 2: Average Performance Score by Department
    # -------------------------------------------------------------
    print("[*] Generating Chart 2: Average Performance Score by Department...")
    dept_perf = (
        df.groupby("department")["performance_score"]
        .mean()
        .reset_index()
        .sort_values(by="performance_score", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=dept_perf,
        x="performance_score",
        y="department",
        palette="Blues_r",
        hue="department",
        legend=False,
        ax=ax
    )

    # Annotate average values at end of each bar
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{width:.1f}",
            (width - 4.0, p.get_y() + p.get_height() / 2.0),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold"
        )

    ax.set_title("Average Performance Score by Department", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Performance Score (Scale: 30 - 100)", fontsize=12, labelpad=8)
    ax.set_ylabel("Department", fontsize=12, labelpad=8)
    plt.tight_layout()

    chart2_path = output_dir / "avg_performance_by_department.png"
    plt.savefig(chart2_path, dpi=200)
    plt.close()
    print(f"    Saved: {chart2_path}")

    # -------------------------------------------------------------
    # Chart 3: Training Hours vs. Performance Score
    # -------------------------------------------------------------
    print("[*] Generating Chart 3: Training Hours vs. Performance Score...")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="training_hours",
        y="performance_score",
        hue="performance_band",
        palette=palette,
        hue_order=order,
        alpha=0.75,
        s=60,
        ax=ax
    )

    # Fit a smooth trend line to highlight upward trajectory
    sns.regplot(
        data=df,
        x="training_hours",
        y="performance_score",
        scatter=False,
        ax=ax,
        color="#2c3e50",
        line_kws={"linestyle": "--", "linewidth": 2, "label": "Linear Trendline"}
    )

    ax.set_title("Relationship: Training Hours vs. Performance Score", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Annual Training Hours Completed", fontsize=12, labelpad=8)
    ax.set_ylabel("Performance Score (30 - 100)", fontsize=12, labelpad=8)
    ax.legend(title="Performance Band", loc="upper left")
    plt.tight_layout()

    chart3_path = output_dir / "training_vs_performance.png"
    plt.savefig(chart3_path, dpi=200)
    plt.close()
    print(f"    Saved: {chart3_path}")

    # -------------------------------------------------------------
    # Chart 4: Correlation Heatmap for Numeric Columns
    # -------------------------------------------------------------
    print("[*] Generating Chart 4: Correlation Heatmap for Numeric Features...")
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        linewidths=0.8,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
        ax=ax
    )

    ax.set_title("Numeric Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    chart4_path = output_dir / "correlation_heatmap.png"
    plt.savefig(chart4_path, dpi=200)
    plt.close()
    print(f"    Saved: {chart4_path}")

    print("\n" + "=" * 70)
    print("EDA COMPLETE: All 4 analytical charts successfully generated in reports/")
    print("=" * 70)


if __name__ == "__main__":
    run_eda(
        data_path="data/employee_performance.csv",
        reports_dir="reports"
    )
