# music-analysis-ml
This is the M.Tech Final project worked on Music analysis.
🎵 Intelligent Music Analysis using Audio Signal Processing & Machine Learning

An end-to-end music intelligence system that analyzes raw audio signals to understand genre, emotion, and similarity using signal processing and machine learning techniques.

📌 Overview

This project focuses on content-based music analysis, leveraging audio signal processing and machine learning to extract meaningful insights directly from music files — without relying on user metadata.

The system performs:

🎼 Music genre classification

😊 Emotion / mood detection

🔍 Music similarity & recommendation

📊 Rich audio visualizations

The project is designed as an M.Tech-level academic project while following industry best practices for reproducibility, modularity, and experimentation.

🎯 Motivation

Traditional music recommendation systems heavily depend on user behavior. This project instead explores how machines can “understand” music itself by analyzing:

Frequency content

Rhythm and tempo

Harmonic and timbral characteristics

Such approaches are widely used in:

Music streaming platforms

Music information retrieval (MIR)

Audio-based recommendation systems

AI-driven media analytics

🧠 Key Concepts Covered

Digital Signal Processing (DSP)

Time–Frequency Analysis

Feature Engineering on Audio Signals

Classical Machine Learning & Deep Learning

Model Evaluation & Experimentation

End-to-End ML Pipelines

🧰 Tech Stack
🎵 Audio & Signal Processing

Librosa

NumPy

SciPy

🤖 Machine Learning

Scikit-learn

XGBoost

PyTorch (CNNs on spectrograms)

📊 Visualization

Matplotlib

Seaborn

📁 Datasets

GTZAN (Genre Classification)

DEAM (Emotion Analysis in Music)

Additional open-source audio datasets

🔬 Feature Extraction

The following features are extracted using Librosa:

Time-Domain Features

Zero Crossing Rate

Root Mean Square (RMS) Energy

Frequency-Domain Features

Spectral Centroid

Spectral Bandwidth

Spectral Roll-off

Spectral Contrast

Cepstral Features

MFCCs (13–40 coefficients)

Delta & Delta-Delta MFCCs

Rhythm & Harmony

Tempo (BPM)

Beat Tracking

Chroma Features

Tonnetz Representation

These features form the basis for both machine learning models and music similarity analysis.

🤖 Machine Learning Models
🎼 Genre Classification

Random Forest

XGBoost

CNN on Mel-Spectrograms

😊 Emotion / Mood Detection

Regression on Valence–Arousal space

Multi-class emotion classification (Happy, Sad, Calm, Energetic)

🔍 Music Similarity & Recommendation

Feature embeddings

Cosine similarity

k-Nearest Neighbors (k-NN)

📈 Evaluation Metrics

Accuracy, Precision, Recall, F1-score

Confusion Matrix

RMSE / MAE (for emotion regression)

Cross-validation

📊 Visualizations

The project includes rich audio visualizations such as:

Waveform plots

Spectrograms

Mel-Spectrograms

MFCC heatmaps

Feature correlation plots

PCA / t-SNE projections for song embeddings

🏗️ Project Structure

music-analysis-ml/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_audio_eda.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_genre_classification.ipynb
│   ├── 04_emotion_detection.ipynb
│
├── src/
│   ├── audio_features.py
│   ├── preprocessing.py
│   ├── train_genre_model.py
│   ├── train_emotion_model.py
│   ├── similarity_engine.py
│
├── models/
├── reports/
│   └── figures/
│
├── requirements.txt
└── README.md

