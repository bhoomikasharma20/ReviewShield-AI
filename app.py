

import streamlit as st
import tensorflow as tf
import joblib
import json
import re
import os

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="ReviewShield AI",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------
# LOAD MODEL FILES
# -------------------------------

@st.cache_resource
def load_model_files():

    model_path = "reviewshield_artifacts/ann_model.keras"
    vectorizer_path = "reviewshield_artifacts/tfidf_vectorizer.pkl"
    metrics_path = "reviewshield_artifacts/metrics.json"

    # Check files
    for path in [model_path, vectorizer_path, metrics_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # Load ANN model
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    # Load TF-IDF vectorizer
    vectorizer = joblib.load(vectorizer_path)

    # Load performance metrics
    with open(metrics_path, "r") as file:
        metrics = json.load(file)

    return model, vectorizer, metrics


# -------------------------------
# TEXT PREPROCESSING
# -------------------------------

def clean_text(text):

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -------------------------------
# HEADER
# -------------------------------

st.title("🛡️ ReviewShield AI")

st.subheader(
    "AI-Powered Product Review Analyzer"
)

st.write(
    "Enter a product review to check whether its text "
    "resembles original or computer-generated reviews."
)

st.divider()


# -------------------------------
# LOAD MODEL SAFELY
# -------------------------------

try:
    model, vectorizer, metrics = load_model_files()

except Exception as e:

    st.error("Unable to load the AI model.")

    st.write("Please check the model files and dependencies.")

    st.exception(e)

    st.stop()


# -------------------------------
# EXAMPLE REVIEWS
# -------------------------------

examples = {

    "ChatGPT Review (CG)": (
        "This product is absolutely amazing! It offers excellent "
        "quality, outstanding performance, and great value for money. "
        "I highly recommend it to everyone."
    ),

    "Human Review (OR)": (
        "I bought this product last week. It works fine so far, "
        "although the packaging was slightly damaged. Overall, "
        "I am happy with my purchase."
    )
}


# -------------------------------
# SESSION STATE
# -------------------------------

if "review_text" not in st.session_state:
    st.session_state.review_text = ""


# -------------------------------
# EXAMPLE BUTTONS
# -------------------------------

st.subheader("📝 Try Example Reviews")

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🤖 ChatGPT Review (CG)",
        use_container_width=True
    ):

        st.session_state.review_text = examples[
            "ChatGPT Review (CG)"
        ]

with col2:

    if st.button(
        "👤 Human Review (OR)",
        use_container_width=True
    ):

        st.session_state.review_text = examples[
            "Human Review (OR)"
        ]


# -------------------------------
# REVIEW INPUT
# -------------------------------

review = st.text_area(
    "Enter your product review",
    placeholder="Type or paste your review here...",
    height=150,
    key="review_text"
)


# -------------------------------
# ANALYZE REVIEW
# -------------------------------

if st.button(
    "🔍 Analyze Review",
    type="primary",
    use_container_width=True
):

    if len(review.strip()) < 10:

        st.warning(
            "Please enter a review of at least 10 characters."
        )

    else:

        try:

            cleaned = clean_text(review)

            features = vectorizer.transform(
                [cleaned]
            ).toarray()

            score = float(
                model.predict(
                    features,
                    verbose=0
                )[0][0]
            )

            # Calculate class scores
            cg_score = score * 100
            or_score = (1 - score) * 100

            st.divider()

            st.subheader("🔍 Analysis Result")

            if score >= 0.55:

                st.warning(
                    "🤖 Potentially Computer-Generated (CG)"
                )

            else:

                st.success(
                    "👤 More Similar to Original (OR)"
                )

            # Display both scores
            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "🤖 CG Score",
                    f"{cg_score:.2f}%"
                )

            with col2:

                st.metric(
                    "👤 OR Score",
                    f"{or_score:.2f}%"
                )

            st.caption(
                "These are model scores, not verified probabilities. "
                "Predictions are not proof that a review is fake or genuine."
            )

        except Exception as e:

            st.error("Error while analyzing the review.")

            st.exception(e)


# -------------------------------
# MODEL PERFORMANCE
# -------------------------------

st.divider()

st.header("📊 Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Accuracy",
    f"{metrics['accuracy']:.2%}"
)

c2.metric(
    "Precision",
    f"{metrics['precision']:.2%}"
)

c3.metric(
    "Recall",
    f"{metrics['recall']:.2%}"
)

c4.metric(
    "F1-score",
    f"{metrics['f1_score']:.2%}"
)


# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "ReviewShield AI | ANN-Based Product Review Classification System"
)


