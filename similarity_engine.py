from sklearn.metrics.pairwise import cosine_similarity




def compute_similarity(feature_matrix):
"""Compute cosine similarity between songs."""
return cosine_similarity(feature_matrix)
