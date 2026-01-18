import pandas as pd




def normalize_features(df):
"""Standardize features for ML models."""
return (df - df.mean()) / df.std()
