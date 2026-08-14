import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Student Risk Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Risk Predictor")
st.write(
    "This app predicts whether a student is **Safe**, **At-Risk**, or **High-Risk** "
    "based on academic and background information."
)

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# The exact order/names of features the model was trained on
FEATURE_NAMES = [
    "attendance",
    "study_hours",
    "past_failures",
    "assignments_completed_pct",
    "parental_education",
    "family_income",
    "extracurricular",
    "internet_access",
    "previous_grade",
    "final_score",
]

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
st.header("Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    study_hours = st.slider("Study Hours (per week)", 0.0, 20.0, 5.0, step=0.5)
    past_failures = st.number_input("Past Failures", min_value=0, max_value=10, value=1, step=1)
    assignments_completed_pct = st.slider("Assignments Completed (%)", 0, 100, 75)
    previous_grade = st.slider("Previous Grade (score)", 0, 100, 70)

with col2:
    final_score = st.slider("Final Score", 0, 100, 65)
    parental_education_label = st.selectbox(
        "Parental Education Level", ["Low", "Medium", "High"]
    )
    family_income_label = st.selectbox(
        "Family Income Level", ["Low", "Medium", "High"]
    )
    extracurricular_label = st.selectbox("Extracurricular Activities?", ["No", "Yes"])
    internet_access_label = st.selectbox("Internet Access at Home?", ["No", "Yes"])

# Map categorical labels to the numeric codes the model expects
level_map = {"Low": 0, "Medium": 1, "High": 2}
yes_no_map = {"No": 0, "Yes": 1}

parental_education = level_map[parental_education_label]
family_income = level_map[family_income_label]
extracurricular = yes_no_map[extracurricular_label]
internet_access = yes_no_map[internet_access_label]

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if st.button("Predict Risk Level", type="primary"):
    input_data = pd.DataFrame(
        [[
            attendance,
            study_hours,
            past_failures,
            assignments_completed_pct,
            parental_education,
            family_income,
            extracurricular,
            internet_access,
            previous_grade,
            final_score,
        ]],
        columns=FEATURE_NAMES,
    )

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == "Safe":
        st.success(f"✅ Predicted Status: **{prediction}**")
    elif prediction == "At-Risk":
        st.warning(f"⚠️ Predicted Status: **{prediction}**")
    else:
        st.error(f"🚨 Predicted Status: **{prediction}**")

    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame(
        {"Class": model.classes_, "Probability": probabilities}
    ).sort_values("Probability", ascending=False)
    st.bar_chart(prob_df.set_index("Class"))
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Note: 'Low/Medium/High' and 'Yes/No' selections are mapped to numeric codes "
    "(0, 1, 2) to match how the model was trained. If your original encoding was "
    "different, adjust the `level_map` / `yes_no_map` dictionaries in app.py."
)
