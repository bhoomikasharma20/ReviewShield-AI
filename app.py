
import streamlit as st
import tensorflow as tf
import joblib
import json
import re

st.set_page_config(
    page_title="ReviewShield AI",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_model_files():
    model = tf.keras.models.load_model(
        "reviewshield_artifacts/ann_model.keras"
    )

    vectorizer = joblib.load(
        "reviewshield_artifacts/tfidf_vectorizer.pkl"
    )

    with open("reviewshield_artifacts/metrics.json") as f:
        metrics = json.load(f)

    return model, vectorizer, metrics


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Website Header
st.title("🛡️ ReviewShield AI")
st.subheader("AI-Powered Product Review Analyzer")

st.write(
    "Enter a product review to check whether its text "
    "resembles original or computer-generated reviews."
)


# Load Model
model, vectorizer, metrics = load_model_files()


# Example Reviews
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


# Session State
if "review_text" not in st.session_state:
    st.session_state.review_text = ""


# Example Buttons
st.divider()
st.subheader("📝 Try Example Reviews")

col1, col2 = st.columns(2)

with col1:
    if st.button("🤖 ChatGPT Review (CG)", use_container_width=True):
        st.session_state.review_text = examples["ChatGPT Review (CG)"]
        st.rerun()

with col2:
    if st.button("👤 Human Review (OR)", use_container_width=True):
        st.session_state.review_text = examples["Human Review (OR)"]
        st.rerun()


# Review Input
review = st.text_area(
    "Enter your product review",
    placeholder="Type or paste your review here...",
    height=150,
    key="review_text"
)


# Analyze Review
if st.button("Analyze Review", type="primary"):

    if len(review.strip()) < 10:
        st.warning("Please enter a review of at least 10 characters.")

    else:
        cleaned = clean_text(review)

        features = vectorizer.transform([cleaned]).toarray()

        score = float(
            model.predict(features, verbose=0)[0][0]
        )

        # Both Class Scores
        cg_score = score * 100
        or_score = (1 - score) * 100

        st.divider()
        st.subheader("🔍 Analysis Result")

        if score >= 0.55:
            st.warning("Potentially Computer-Generated (CG)")
        else:
            st.success("More Similar to Original (OR)")

        # Display Both Scores
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
            "A high score does not guarantee a correct classification."
        )


# Model Performance
st.divider()
st.header("📊 Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Precision", f"{metrics['precision']:.1%}")
c3.metric("Recall", f"{metrics['recall']:.1%}")
c4.metric("F1-score", f"{metrics['f1_score']:.1%}")

