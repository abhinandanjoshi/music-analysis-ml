import librosa
import numpy as np




def extract_audio_features(file_path, sr=22050, n_mfcc=20):
"""
Extracts comprehensive audio features from a music file.
"""
y, sr = librosa.load(file_path, sr=sr)


features = {}


# Time-domain
features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
features['rms'] = np.mean(librosa.feature.rms(y=y))


# Frequency-domain
features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))


# MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
for i in range(n_mfcc):
features[f'mfcc_{i+1}'] = np.mean(mfccs[i])


# Rhythm
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
features['tempo'] = tempo


return features
