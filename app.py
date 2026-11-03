import streamlit as st
import tensorflow as tf
import joblib
import json
import numpy as np
from pathlib import Path


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="ReviewShield AI",
    page_icon="🛡️",
    layout="centered"
)


# ---------------- FILE PATHS ----------------

# ---------------- FILE PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "reviewshield_artifacts" / "ann_model.keras"
VECTORIZER_PATH = BASE_DIR / "reviewshield_artifacts" / "tfidf_vectorizer.pkl"
METRICS_PATH = BASE_DIR / "reviewshield_artifacts" / "metrics.json"


# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.title {
    text-align: center;
    color: #174477;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model_files():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    vectorizer = joblib.load(VECTORIZER_PATH)

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    return model, vectorizer, metrics


# ---------------- HEADER ----------------

st.markdown(
    '<div class="title">🛡️ ReviewShield AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Product Review Classification System</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Enter a product review to classify it as Original (OR) "
    "or Computer-Generated (CG)."
)


# ---------------- LOAD FILES ----------------

try:
    model, vectorizer, metrics = load_model_files()

except Exception as e:
    st.error("Model files could not be loaded.")
    st.write("Please check that all required files are available.")
    st.exception(e)
    st.stop()


# ---------------- EXAMPLE REVIEWS ----------------

st.subheader("Try Example Reviews")

col1, col2 = st.columns(2)

with col1:
    if st.button("Original Review Example"):
        st.session_state["review_input"] = (
            "I bought this product last week. "
            "It works well and the quality is good. "
            "I am happy with my purchase."
        )

with col2:
    if st.button("Generated Review Example"):
        st.session_state["review_input"] = (
            "This amazing product offers outstanding quality "
            "and excellent performance. Highly recommended "
            "for everyone."
        )


# ---------------- REVIEW INPUT ----------------

st.subheader("Enter Your Product Review")

review = st.text_area(
    "Write or paste your review:",
    key="review_input",
    height=160,
    placeholder="Type your product review here..."
)


# ---------------- PREDICTION ----------------

if st.button("Analyze Review", type="primary"):

    if not review.strip():

        st.warning("Please enter a review first.")

    else:

        try:
            # Yahan .toarray() add kar diya gaya hai
            review_vector = vectorizer.transform([review]).toarray()

            prediction = model.predict(
                review_vector,
                verbose=0
            )

            cg_score = float(
                np.asarray(prediction).reshape(-1)[0]
            )

            or_score = 1 - cg_score

            cg_percentage = cg_score * 100
            or_percentage = or_score * 100

            threshold = 0.55

            st.write("---")
            st.subheader("Analysis Result")

            if cg_score >= threshold:

                st.error("🤖 Computer-Generated Review (CG)")

            else:

                st.success("👤 Original Review (OR)")

            # Display scores

            st.subheader("Model Scores")

            score_col1, score_col2 = st.columns(2)

            with score_col1:
                st.metric(
                    "Original (OR)",
                    f"{or_percentage:.2f}%"
                )

            with score_col2:
                st.metric(
                    "Computer-Generated (CG)",
                    f"{cg_percentage:.2f}%"
                )

            st.caption(
                "These are model scores, not verified probabilities "
                "or proof of the review's true origin."
            )

        except Exception as e:

            st.error("Prediction failed.")
            st.exception(e)


# ---------------- MODEL PERFORMANCE ----------------

st.write("---")
st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Accuracy",
        f"{float(metrics.get('accuracy', 0)) * 100:.2f}%"
    )

with col2:
    st.metric(
        "F1 Score",
        f"{float(metrics.get('f1_score', 0)) * 100:.2f}%"
    )

col3, col4 = st.columns(2)

with col3:
    st.metric(
        "Precision",
        f"{float(metrics.get('precision', 0)) * 100:.2f}%"
    )

with col4:
    st.metric(
        "Recall",
        f"{float(metrics.get('recall', 0)) * 100:.2f}%"
    )


# ---------------- FOOTER ----------------

st.write("---")

st.markdown(
    "<center>Developed as an AI-Based Product Review "
    "Classification Project</center>",
    unsafe_allow_html=True
)




