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

st.title("🛡️ ReviewShield AI")
st.subheader("AI-Powered Product Review Analyzer")
st.write(
    "Enter a product review to check whether its text "
    "resembles original or computer-generated reviews."
)

model, vectorizer, metrics = load_model_files()

review = st.text_area(
    "Enter your product review",
    placeholder="Type or paste your review here...",
    height=150
)

if st.button("Analyze Review", type="primary"):
    if len(review.strip()) < 10:
        st.warning("Please enter a review of at least 10 characters.")
    else:
        cleaned = clean_text(review)
        features = vectorizer.transform([cleaned]).toarray()
        score = float(model.predict(features, verbose=0)[0][0])

        if score >= 0.5:
            st.warning("Potentially Computer-Generated (CG)")
            st.metric("CG model score", f"{score:.1%}")
        else:
            st.success("More Similar to Original (OR)")
            st.metric("OR model score", f"{1-score:.1%}")

        st.caption(
            "This is a model prediction, not proof that a review "
            "is fake or genuine."
        )

st.divider()
st.header("📊 Model Performance")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Precision", f"{metrics['precision']:.1%}")
c3.metric("Recall", f"{metrics['recall']:.1%}")
c4.metric("F1-score", f"{metrics['f1_score']:.1%}")
