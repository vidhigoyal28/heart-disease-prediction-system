import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ======================================
# CUSTOM CSS
# ======================================

st.markdown("""
<style>

.main-header {
    text-align:center;
    padding:20px;
    border-radius:15px;
    background: linear-gradient(90deg,#ff4b4b,#ff6b6b);
    color:white;
    margin-bottom:20px;
}

.result-card {
    padding:20px;
    border-radius:15px;
    background-color:#f8f9fa;
}

.metric-container {
    background-color:#f8f9fa;
    padding:15px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ======================================
# LOAD MODEL
# ======================================

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "models" / "rf_model.pkl"
)

# ======================================
# SIDEBAR
# ======================================

with st.sidebar:

    st.header("📊 Model Information")

    st.success("Best Model: Random Forest")

    st.metric(
        "Accuracy",
        "90.16%"
    )

    st.metric(
        "ROC-AUC",
        "95.45%"
    )

    st.markdown("---")

    st.subheader("🛠 Technologies")

    st.write("• Python")
    st.write("• Scikit-Learn")
    st.write("• Random Forest")
    st.write("• Streamlit")
    st.write("• Pandas")

# ======================================
# HEADER
# ======================================

st.markdown("""
<div class="main-header">
<h1>❤️ Heart Disease Prediction System</h1>
<p>
Predict cardiovascular disease risk using Machine Learning
</p>
</div>
""", unsafe_allow_html=True)

st.info("""
This application predicts the likelihood of heart disease
using a Random Forest Machine Learning model trained on the
UCI Heart Disease Dataset.
""")

# ======================================
# INPUT SECTION
# ======================================

st.header("🩺 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=20,
        max_value=100,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        [0,1],
        format_func=lambda x:
        "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [1,2,3,4]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=600,
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120",
        [0,1]
    )

with col2:

    restecg = st.selectbox(
        "Rest ECG",
        [0,1,2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=60,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0,1]
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0
    )

    slope = st.selectbox(
        "Slope",
        [1,2,3]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0,1,2,3]
    )

    thal = st.selectbox(
        "Thalassemia",
        [3,6,7]
    )

st.markdown("")

# ======================================
# PREDICTION BUTTON
# ======================================

if st.button(
    "🔍 Predict Heart Disease Risk",
    use_container_width=True
):

    features = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    st.markdown("---")

    st.header("📋 Prediction Result")

    st.metric(
        "Heart Disease Probability",
        f"{probability:.2%}"
    )

    st.progress(float(probability))

    # ======================================
    # RISK CATEGORY
    # ======================================

    if probability < 0.30:

        st.success("🟢 LOW RISK")

    elif probability < 0.70:

        st.warning("🟡 MODERATE RISK")

    else:

        st.error("🔴 HIGH RISK")

    # ======================================
    # PATIENT SUMMARY
    # ======================================

    st.subheader("👤 Patient Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Age", age)
    c2.metric("Blood Pressure", trestbps)
    c3.metric("Cholesterol", chol)

    # ======================================
    # RECOMMENDATIONS
    # ======================================

    st.subheader("💡 Health Recommendations")

    if probability > 0.70:

        st.error("""
        • Consult a cardiologist

        • Monitor blood pressure regularly

        • Reduce saturated fats

        • Exercise at least 30 minutes daily

        • Schedule regular health checkups
        """)

    elif probability > 0.30:

        st.warning("""
        • Improve diet quality

        • Exercise regularly

        • Maintain healthy body weight

        • Monitor cholesterol levels
        """)

    else:

        st.success("""
        • Maintain current healthy lifestyle

        • Continue regular exercise

        • Eat a balanced diet

        • Schedule routine checkups
        """)

    # ======================================
    # FEATURE IMPORTANCE
    # ======================================

    st.subheader("📈 Feature Importance")

    image_path = BASE_DIR / "results" / "feature_importance.png"

    if image_path.exists():

        st.image(
            str(image_path),
            use_container_width=True
        )

# ======================================
# FOOTER
# ======================================

st.markdown("---")

st.markdown(
"""
<center>

❤️ Heart Disease Prediction System

Built using Python, Scikit-Learn, Random Forest and Streamlit

</center>
""",
unsafe_allow_html=True
)