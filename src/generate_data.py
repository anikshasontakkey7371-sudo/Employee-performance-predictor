"""
Employee Performance Predictor - Synthetic Data Generator
---------------------------------------------------------
Author: Data Science Team
Description:
    Generates 1,000 rows of reproducible synthetic HR data to simulate
    employee productivity, attendance, and feedback metrics.

DISCLAIMER & ETHICAL SAFEGUARD:
    This dataset is 100% synthetically generated for educational, demonstration,
    and technical assessment purposes. It contains NO real or confidential
    employee personal identifiable information (PII). Results derived from
    synthetic data do NOT establish real-world predictive validity for actual
    workplace evaluations. Predictions should always be treated as advisory
    decision-support signals, requiring human review and fairness audits.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_employee_data(
    num_samples: int = 1000,
    random_seed: int = 42,
    output_path: str = "data/employee_performance.csv"
) -> pd.DataFrame:
    """
    Generate synthetic employee records and save to CSV.

    Parameters:
    -----------
    num_samples : int, default=1000
        Number of employee records to generate.
    random_seed : int, default=42
        Seed for NumPy random generator to ensure exact reproducibility.
    output_path : str, default="data/employee_performance.csv"
        File path where the generated CSV will be stored.

    Returns:
    --------
    pd.DataFrame
        The generated synthetic employee DataFrame.
    """
    # Set seed for deterministic reproducibility across runs
    rng = np.random.default_rng(random_seed)

    # 1. Unique Employee Identifiers
    employee_ids = [f"EMP-{i:04d}" for i in range(1001, 1001 + num_samples)]

    # 2. Departments with realistic distribution across corporate functions
    departments_pool = [
        "Engineering",
        "Sales",
        "Marketing",
        "Finance",
        "Operations",
        "Human Resources",
    ]
    department_probs = [0.28, 0.24, 0.16, 0.14, 0.12, 0.06]
    departments = rng.choice(departments_pool, size=num_samples, p=department_probs)

    # 3. Years of Work Experience (between 1 and 22 years)
    years_experience = rng.integers(low=1, high=23, size=num_samples)

    # 4. Annual Training Hours Completed (between 8 and 80 hours)
    training_hours = rng.integers(low=8, high=81, size=num_samples)

    # 5. Attendance Rate (percentage: 75.0% to 100.0%)
    # Normal distribution centered around 93% with a standard deviation of 4.5%
    raw_attendance = rng.normal(loc=93.0, scale=4.5, size=num_samples)
    attendance_rate = np.clip(raw_attendance, 75.0, 100.0).round(1)

    # 6. Projects Completed (1 to 15 projects, influenced slightly by experience)
    base_projects = rng.poisson(lam=4.5, size=num_samples)
    exp_boost = (years_experience // 4).astype(int)
    projects_completed = np.clip(base_projects + exp_boost, 1, 15)

    # 7. Manager Feedback Score (1.0 to 5.0 scale, rounded to 1 decimal place)
    raw_feedback = rng.normal(loc=3.4, scale=0.7, size=num_samples)
    manager_feedback_score = np.clip(raw_feedback, 1.0, 5.0).round(1)

    # 8. Performance Score (continuous score from ~30.0 to 100.0)
    # Normalized feature contributions to simulate realistic workplace evaluations:
    # - Manager Feedback: 30% weight
    # - Project Output: 25% weight
    # - Attendance Consistency: 20% weight
    # - Training & Upskilling: 15% weight
    # - Experience Maturity: 10% weight
    feedback_norm = (manager_feedback_score - 1.0) / 4.0   # 0.0 to 1.0
    attendance_norm = (attendance_rate - 75.0) / 25.0       # 0.0 to 1.0
    projects_norm = (projects_completed - 1.0) / 14.0       # 0.0 to 1.0
    training_norm = (training_hours - 8.0) / 72.0           # 0.0 to 1.0
    experience_norm = (years_experience - 1.0) / 21.0       # 0.0 to 1.0

    # Slight realistic noise/variation (unmeasured intangibles)
    noise = rng.normal(loc=0.0, scale=0.05, size=num_samples)

    composite_ratio = (
        0.30 * feedback_norm
        + 0.25 * projects_norm
        + 0.20 * attendance_norm
        + 0.15 * training_norm
        + 0.10 * experience_norm
        + noise
    )

    # Map composite ratio to standard corporate 35.0 - 98.0 score scale
    performance_score = np.clip(35.0 + composite_ratio * 60.0, 30.0, 100.0).round(1)

    # 9. Performance Band (Low, Medium, High categories)
    # Realistic Corporate Distribution:
    # - Low (< 62.0): ~20% (Needs Improvement / Underperforming)
    # - Medium (62.0 - 72.9): ~55% (Core / Solid Contributor)
    # - High (>= 73.0): ~25% (High Impact / Exceeds Expectations)
    bins = [0.0, 62.0, 73.0, 100.0]
    labels = ["Low", "Medium", "High"]
    performance_band = pd.cut(
        performance_score,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    )

    # Assemble into a clean Pandas DataFrame
    df = pd.DataFrame({
        "employee_id": employee_ids,
        "department": departments,
        "years_experience": years_experience,
        "training_hours": training_hours,
        "attendance_rate": attendance_rate,
        "projects_completed": projects_completed,
        "manager_feedback_score": manager_feedback_score,
        "performance_score": performance_score,
        "performance_band": performance_band,
    })

    # Ensure target directory exists before saving
    save_file = Path(output_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(save_file, index=False)
    print(f"Dataset successfully saved to: {save_file.resolve()}\n")

    return df


if __name__ == "__main__":
    print("=" * 65)
    print("Employee Performance Predictor - Synthetic Data Generation")
    print("=" * 65)

    # Generate the dataset
    data = generate_employee_data(
        num_samples=1000,
        random_seed=42,
        output_path="data/employee_performance.csv"
    )

    # Display dataset shape
    print(f"Dataset Shape: {data.shape[0]} rows x {data.shape[1]} columns\n")

    # Display the first 5 rows
    print("First 5 Rows:")
    print(data.head())
    print("\n" + "-" * 65 + "\n")

    # Display performance band counts and percentages
    print("Performance Band Distribution:")
    band_counts = data["performance_band"].value_counts(sort=False)
    band_percentages = data["performance_band"].value_counts(normalize=True, sort=False) * 100

    summary_df = pd.DataFrame({
        "Count": band_counts,
        "Percentage (%)": band_percentages.round(2)
    })
    print(summary_df)
    print("=" * 65)
