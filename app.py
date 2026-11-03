

import json
import joblib
import streamlit as st
import tensorflow as tf
import numpy as np


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="ReviewShield AI",
    page_icon="🛡️",
    layout="centered"
)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.title {
    text-align: center;
    color: #173b67;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #555555;
    font-size: 16px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL FILES ----------------

@st.cache_resource
def load_model_files():

    model = tf.keras.models.load_model(
        "ann_model.keras",
        compile=False
    )

    vectorizer = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    with open("metrics.json", "r") as file:
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
    "Enter a product review below to check whether it is "
    "Original (OR) or Computer-Generated (CG)."
)


# ---------------- LOAD FILES SAFELY ----------------

try:
    model, vectorizer, metrics = load_model_files()

except Exception as e:
    st.error("Unable to load the model files.")
    st.write("Please check that all model files are present in the repository root.")
    st.exception(e)
    st.stop()


# ---------------- EXAMPLE REVIEWS ----------------

st.subheader("Try Example Reviews")

col1, col2 = st.columns(2)

with col1:
    if st.button("Original Review Example"):
        st.session_state["review_input"] = (
            "I bought this product last week and have been "
            "using it every day. The quality is good and "
            "it works as expected."
        )

with col2:
    if st.button("Generated Review Example"):
        st.session_state["review_input"] = (
            "This amazing product provides excellent quality "
            "and outstanding performance. It is highly "
            "recommended for everyone."
        )


# ---------------- USER INPUT ----------------

st.subheader("Enter Your Product Review")

review = st.text_area(
    "Write or paste your review here:",
    key="review_input",
    height=160,
    placeholder="Type your product review..."
)


# ---------------- PREDICTION ----------------

if st.button("Analyze Review", type="primary"):

    if not review.strip():
        st.warning("Please enter a review first.")

    else:
        try:
            # Convert review into TF-IDF features
            review_vector = vectorizer.transform([review])

            # ANN prediction
            prediction = model.predict(
                review_vector,
                verbose=0
            )

            # Get CG score
            cg_score = float(np.asarray(prediction).reshape(-1)[0])

            # Convert scores into percentages
            cg_percentage = cg_score * 100
            or_percentage = (1 - cg_score) * 100

            # Classification threshold
            threshold = 0.55

            st.write("---")
            st.subheader("Analysis Result")

            if cg_score >= threshold:

                st.error("🤖 Computer-Generated Review (CG)")

                st.markdown(
                    f"""
                    <div class="result-box"
                    style="background-color:#ffe5e5;">
                    <h3 style="color:#b30000;">
                    Computer-Generated (CG)
                    </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.success("👤 Original Review (OR)")

                st.markdown(
                    f"""
                    <div class="result-box"
                    style="background-color:#e5f5e9;">
                    <h3 style="color:#08752c;">
                    Original (OR)
                    </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Display model scores
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
                "These percentages are model scores, not verified "
                "probabilities or proof of a review's true origin."
            )

        except Exception as e:
            st.error("An error occurred during prediction.")
            st.exception(e)


# ---------------- MODEL PERFORMANCE ----------------

st.write("---")
st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Test Accuracy",
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




