
import streamlit as st
import tensorflow as tf
import joblib
import json
import numpy as np
from pathlib import Path
import streamlit.components.v1 as components
import io
import wave
import math
import struct
import base64

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="ReviewShield AI",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- FILE PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ann_model.keras"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef5ff, #f8faff);
}

.title {
    text-align: center;
    color: #174477;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 17px;
    margin-bottom: 25px;
}

div.stButton > button {
    border-radius: 12px;
    font-weight: bold;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.03);
    border-color: #2563eb;
}

.result-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin: 15px 0;
    animation: appear 0.8s ease-in-out;
}

.cg-box {
    background: linear-gradient(135deg, #fff0f0, #ffe0e0);
    border: 2px solid #ef4444;
    color: #991b1b;
}

.or-box {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #10b981;
    color: #065f46;
}

.result-box h2 {
    font-size: 26px;
    margin: 5px;
}

.result-box p {
    font-size: 16px;
    margin: 5px;
}

@keyframes appear {
    from {
        opacity: 0;
        transform: translateY(20px) scale(0.95);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
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


# ---------------- BEEP SOUND ----------------

def play_beep():

    sample_rate = 44100
    duration = 0.20
    frequency = 750

    audio_buffer = io.BytesIO()

    with wave.open(audio_buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for i in range(int(sample_rate * duration)):
            value = int(
                10000 * math.sin(
                    2 * math.pi * frequency * i / sample_rate
                )
            )
            wav.writeframesraw(struct.pack("<h", value))

    audio_base64 = base64.b64encode(
        audio_buffer.getvalue()
    ).decode("utf-8")

    components.html(
        f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{audio_base64}"
                    type="audio/wav">
        </audio>
        """,
        height=0
    )


# ---------------- HEADER ----------------

st.markdown(
    '<div class="title">🛡️ ReviewShield AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Product Review Classification System</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter a product review to classify it as "
    "Original (OR) or Computer-Generated (CG)."
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

st.subheader("✨ Try Example Reviews")

col1, col2 = st.columns(2)

with col1:
    if st.button("👤 Original Review Example", use_container_width=True):
        st.session_state["review_input"] = (
            "I bought this product last week. "
            "It works well and the quality is good. "
            "I am happy with my purchase."
        )

with col2:
    if st.button("🤖 Generated Review Example", use_container_width=True):
        st.session_state["review_input"] = (
            "This amazing product offers outstanding quality "
            "and excellent performance. Highly recommended "
            "for everyone."
        )

# ---------------- REVIEW INPUT ----------------

st.subheader("📝 Enter Your Product Review")

review = st.text_area(
    "Write or paste your review:",
    key="review_input",
    height=160,
    placeholder="Type your product review here..."
)

# ---------------- PREDICTION ----------------

if st.button("🔍 Analyze Review", type="primary", use_container_width=True):

    if not review.strip():

        st.warning("Please enter a review first.")

    else:

        try:
            review_vector = vectorizer.transform(
                [review]
            ).toarray()

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
            st.subheader("📊 Analysis Result")

            # Play beep
            play_beep()

            if cg_score >= threshold:

                st.markdown("""
                <div class="result-box cg-box">
                    <h2>🤖 Computer-Generated Review</h2>
                    <p><b>Classification: CG</b></p>
                    <p>The model classified this review as computer-generated.</p>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-box or-box">
                    <h2>👤 Original Review</h2>
                    <p><b>Classification: OR</b></p>
                    <p>The model classified this review as original.</p>
                </div>
                """, unsafe_allow_html=True)

            # ---------------- SCORES ----------------

            st.subheader("📈 Model Scores")

            score_col1, score_col2 = st.columns(2)

            with score_col1:
                st.metric(
                    "👤 Original (OR)",
                    f"{or_percentage:.2f}%"
                )

            with score_col2:
                st.metric(
                    "🤖 Computer-Generated (CG)",
                    f"{cg_percentage:.2f}%"
                )

            st.write("**Original (OR) Score**")
            st.progress(float(or_score))

            st.write("**Computer-Generated (CG) Score**")
            st.progress(float(cg_score))

            st.caption(
                "These are model scores, not verified probabilities "
                "or proof of the review's true origin."
            )

        except Exception as e:

            st.error("Prediction failed.")
            st.exception(e)

# ---------------- MODEL PERFORMANCE ----------------

st.write("---")
st.subheader("🏆 Model Performance")

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
    '<div class="footer">'
    '🛡️ Developed as an AI-Based Product Review Classification Project'
    '</div>',
    unsafe_allow_html=True
)





