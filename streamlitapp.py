import streamlit as st
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Music Genre Classification")
st.write(
    """
    Upload an audio file (.wav) and the model will predict its music genre.
    This app demonstrates an end-to-end machine learning pipeline
    for audio feature extraction and inference.
    """
)

# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = Path("models/genre_classifier.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_features(audio, sr):
    """
    Extract audio features for ML inference.
    Returns a 1D feature vector.
    """
    features = []

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    features.extend(mfccs_mean)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    features.extend(chroma_mean)

    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(np.mean(spec_centroid))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))

    return np.array(features).reshape(1, -1)

# -----------------------------
# Audio Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Load audio
    audio, sr = librosa.load(uploaded_file, sr=None)

    st.subheader("Audio Visualization")

    # Plot waveform
    fig, ax = plt.subplots()
    ax.plot(audio)
    ax.set_title("Waveform")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # -----------------------------
    # Inference
    # -----------------------------
    if st.button("Predict Genre"):
        with st.spinner("Extracting features and predicting..."):
            features = extract_features(audio, sr)
            prediction = model.predict(features)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                confidence = np.max(probabilities) * 100
            else:
                confidence = None

        st.success(f"🎶 Predicted Genre: **{prediction}**")

        if confidence is not None:
            st.info(f"Model confidence: **{confidence:.2f}%**")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "Built with **Python, librosa, scikit-learn, Streamlit & Docker**"
)

