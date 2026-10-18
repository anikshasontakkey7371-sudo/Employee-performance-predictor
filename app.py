"""
Employee Performance Predictor - Streamlit Web Application
----------------------------------------------------------
Author: Data Science Team
Description:
    An interactive, beginner-friendly Streamlit dashboard that allows HR analysts
    and managers to test employee performance predictions using our trained
    scikit-learn Random Forest pipeline.

ETHICAL AI & SAFETY NOTICE:
    This application is an educational demonstration built on synthetic data.
    Predictions are purely advisory and must never be used as an autonomous basis
    for employment decisions (hiring, promotion, compensation, or termination).
    No Personally Identifiable Information (PII) such as names or employee IDs
    is collected or processed.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished typography, metric cards, and prediction badges
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .card-high {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46;
    }
    .card-medium {
        background-color: #eff6ff;
        border: 2px solid #3b82f6;
        color: #1e40af;
    }
    .card-low {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 2. Model Loading (Cached for fast performance with clear error reporting)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: str = "models/employee_performance_model.joblib"):
    """
    Load the serialized scikit-learn pipeline from disk.
    Cached via Streamlit so it is only loaded once in memory.
    Returns (model, error_message).
    """
    file_path = Path(model_path)
    if not file_path.exists():
        return None, f"Model file not found at '{file_path.resolve()}'. Please run `python src/train_model.py` first."
    try:
        model = joblib.load(file_path)
        return model, None
    except Exception as exc:
        return None, f"Failed to load model artifact: {str(exc)}"


# -----------------------------------------------------------------------------
# 3. Main Application Header & Advisory Notice
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">📈 Employee Performance Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A Data Analytics & Machine Learning Decision-Support System</div>',
    unsafe_allow_html=True
)

# Native Streamlit Warning Banner:
# Guaranteed readable text, contrast-safe across Light/Dark themes, and clear advisory information
st.warning(
    "**Advisory & Educational Notice:**\n\n"
    "This application is a portfolio and educational demonstration powered by "
    "**synthetic HR analytics data**. It is designed solely as an advisory decision-support "
    "exploration tool and **must never** be used as an autonomous decision-maker for real-world "
    "employment actions (such as promotions, performance reviews, compensation, or terminations). "
    "In compliance with data privacy standards, **no employee names, IDs, or confidential PII are collected**.",
    icon="⚠️"
)

# -----------------------------------------------------------------------------
# 4. Check Model Availability & Validate
# -----------------------------------------------------------------------------
model_path = "models/employee_performance_model.joblib"
model, load_error = load_trained_model(model_path)

if load_error is not None:
    st.error(f"🚨 Model Error: {load_error}")
    st.stop()


# -----------------------------------------------------------------------------
# 5. User Input Layout (Sidebar & Main Columns)
# -----------------------------------------------------------------------------
st.sidebar.header("📋 Employee Profile Attributes")
st.sidebar.caption("Adjust the sliders and dropdown to evaluate a prospective profile:")

# 1. Department (Categorical)
department = st.sidebar.selectbox(
    "Department",
    options=["Engineering", "Sales", "Marketing", "Finance", "Operations", "Human Resources"],
    index=0,
    help="Select the business unit or functional division."
)

# 2. Years of Experience (Numeric)
years_experience = st.sidebar.slider(
    "Years of Experience",
    min_value=1,
    max_value=25,
    value=5,
    step=1,
    help="Total professional experience in years."
)

# 3. Training Hours (Numeric)
training_hours = st.sidebar.slider(
    "Annual Training Hours",
    min_value=5,
    max_value=100,
    value=40,
    step=1,
    help="Total hours spent in upskilling, workshops, or training courses."
)

# 4. Attendance Rate (Numeric, Percentage)
attendance_rate = st.sidebar.slider(
    "Attendance Rate (%)",
    min_value=75.0,
    max_value=100.0,
    value=94.5,
    step=0.1,
    format="%.1f%%",
    help="Annual attendance percentage (accounting for approved leaves)."
)

# 5. Projects Completed (Numeric)
projects_completed = st.sidebar.slider(
    "Projects Completed",
    min_value=1,
    max_value=15,
    value=7,
    step=1,
    help="Number of projects successfully completed during the review period."
)

# 6. Manager Feedback Score (Numeric, 1.0 - 5.0)
manager_feedback_score = st.sidebar.slider(
    "Manager Feedback Score (1.0 - 5.0)",
    min_value=1.0,
    max_value=5.0,
    value=3.8,
    step=0.1,
    format="%.1f",
    help="Composite rating from direct manager reviews (scale: 1.0 to 5.0)."
)


# -----------------------------------------------------------------------------
# 6. Construct Input DataFrame (Matches Training Feature Names Exactly)
# -----------------------------------------------------------------------------
input_data = {
    "department": department,
    "years_experience": int(years_experience),
    "training_hours": int(training_hours),
    "attendance_rate": float(attendance_rate),
    "projects_completed": int(projects_completed),
    "manager_feedback_score": float(manager_feedback_score),
}

input_df = pd.DataFrame([input_data])

# -----------------------------------------------------------------------------
# 7. Real-Time Inference & UI Display
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.subheader("🔍 Selected Profile Summary")
    st.caption("Feature payload fed to the preprocessor and classifier pipeline:")

    # Display clean metric cards
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Department", department)
    m_col2.metric("Experience", f"{years_experience} yrs")
    m_col3.metric("Training", f"{training_hours} hrs")

    m_col4, m_col5, m_col6 = st.columns(3)
    m_col4.metric("Attendance", f"{attendance_rate:.1f}%")
    m_col5.metric("Projects", f"{projects_completed}")
    m_col6.metric("Feedback Score", f"{manager_feedback_score:.1f} / 5.0")

    # Table view of raw features
    with st.expander("View Raw Feature Input Table", expanded=False):
        st.dataframe(input_df, use_container_width=True)

with col_right:
    st.subheader("🎯 Prediction Output")

    try:
        # Trigger inference through the serialized pipeline
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        # Map predicted class to custom styling card
        if prediction == "High":
            card_class = "card-high"
            icon = "🌟"
            explanation = "Indicates strong potential for exceeding performance targets and leadership contributions."
        elif prediction == "Medium":
            card_class = "card-medium"
            icon = "✅"
            explanation = "Indicates steady, consistent delivery meeting corporate goals and expectations."
        else:  # "Low"
            card_class = "card-low"
            icon = "⚠️"
            explanation = "Indicates areas needing support, mentorship, or targeted training interventions."

        # Render styled prediction card
        st.markdown(
            f"""
            <div class="prediction-card {card_class}">
                <h4 style="margin:0; text-transform:uppercase; letter-spacing:1px;">Predicted Performance Band</h4>
                <h1 style="margin: 0.5rem 0; font-size:2.8rem;">{icon} {prediction}</h1>
                <p style="margin:0; font-size:0.95rem;">{explanation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Class Probability Distribution with progress bars
        st.markdown("##### Model Confidence Breakdown:")
        prob_dict = dict(zip(classes, probabilities))

        for band in ["High", "Medium", "Low"]:
            if band in prob_dict:
                prob_val = prob_dict[band]
                pct_val = prob_val * 100.0
                st.write(f"**{band} Band:** {pct_val:.1f}%")
                st.progress(float(prob_val))

    except Exception as infer_err:
        st.error(f"Prediction Error: Unable to compute prediction. Details: {str(infer_err)}")


# -----------------------------------------------------------------------------
# 8. Model Transparency & Analytical Reports Drawer
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("📊 View Model Transparency & Evaluation Artifacts", expanded=False):
    t_tab1, t_tab2 = st.tabs(["Confusion Matrix Plot", "Classification Report"])

    with t_tab1:
        cm_path = Path("reports/confusion_matrix.png")
        if cm_path.exists():
            st.image(str(cm_path), caption="Random Forest Test Set Confusion Matrix (200 Hold-out Samples)", width=550)
        else:
            st.info("Confusion matrix plot not found. Run `python src/train_model.py` to generate it.")

    with t_tab2:
        report_path = Path("reports/classification_report.txt")
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                st.text(f.read())
        else:
            st.info("Classification report not found. Run `python src/train_model.py` to generate it.")

st.caption("Employee Performance Predictor | Senior Data Science Architecture | Streamlit v1.66")
