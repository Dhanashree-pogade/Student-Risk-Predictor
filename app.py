import streamlit as st
import pickle
import numpy as np
import os

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Prediction System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 100%);
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Cards */
    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }

    .prediction-title {
        font-size: 20px;
        margin-bottom: 10px;
    }

    .prediction-result {
        font-size: 36px;
        font-weight: 800;
    }

    /* Info boxes */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.06);
    }

    .info-number {
        font-size: 28px;
        font-weight: 700;
        color: #667eea;
    }

    .info-text {
        color: #6b7280;
        font-size: 14px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 17px;
        font-weight: 700;
        border: none;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #5a67d8, #6b46c1);
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        margin-top: 40px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():

    model_path = "model (1).pkl"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found.")
        st.stop()

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:

    st.markdown("## 🤖 AI Prediction")

    st.markdown("---")

    st.markdown("""
    ### 📌 About

    This application uses a **Machine Learning model**
    to generate predictions based on user-provided inputs.

    **Model:** Gaussian Naive Bayes

    **Framework:** Streamlit

    **Language:** Python
    """)

    st.markdown("---")

    st.markdown("### ⚙️ Model Information")

    st.write("🧠 Algorithm: GaussianNB")
    st.write("🐍 Python: Python 3")
    st.write("🚀 Deployment: Streamlit")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🤖 AI Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning powered prediction dashboard</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# TOP INFORMATION CARDS
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">🧠 AI</div>
        <div class="info-text">Machine Learning Model</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">⚡ Fast</div>
        <div class="info-text">Instant Prediction</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">🎯 Smart</div>
        <div class="info-text">Data Driven Results</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown("### 📊 Enter Input Data")

st.info(
    "⚠️ Replace the input fields below with the exact features "
    "used while training your model."
)

# ---------------------------------------------------------
# EXAMPLE INPUTS
# CHANGE THESE ACCORDING TO YOUR DATASET
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    feature_1 = st.number_input(
        "Feature 1",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

    feature_2 = st.number_input(
        "Feature 2",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

with col2:

    feature_3 = st.number_input(
        "Feature 3",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

    feature_4 = st.number_input(
        "Feature 4",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
st.markdown("### 🔮 Generate Prediction")

if st.button("🚀 Predict Now"):

    try:

        # IMPORTANT:
        # Keep the feature order exactly the same
        # as the order used during model training.

        input_data = np.array([[
            feature_1,
            feature_2,
            feature_3,
            feature_4
        ]])

        prediction = model.predict(input_data)

        result = prediction[0]

        # -------------------------------------------------
        # PREDICTION PROBABILITY
        # -------------------------------------------------
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)
            probability = np.max(probabilities) * 100

        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="prediction-box">
            <div class="prediction-title">
                🎯 Prediction Result
            </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div class="prediction-result">{result}</div>',
            unsafe_allow_html=True
        )

        if probability is not None:

            st.markdown(
                f"<p>Model Confidence: <b>{probability:.2f}%</b></p>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------------------------
        # DETAILS
        # -------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"✅ Prediction: **{result}**")

        with col2:
            if probability is not None:
                st.info(
                    f"📈 Confidence: **{probability:.2f}%**"
                )

    except Exception as e:

        st.error(
            "❌ Prediction failed. Please check that the "
            "number of inputs and their order match the "
            "features used during model training."
        )

        st.exception(e)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<div class="footer">
    <hr>
    <p>🤖 AI Prediction System | Built with Python & Streamlit</p>
    <p>Machine Learning • Data Science • Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)
