"""
Data modeling utilities for classifier-based recommenders.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from logger import setup_logger

# Initialize logger
logger = setup_logger()

# Module-level cache for KMeans model
_kmeans_model = None

# Path for saving/loading models
MODELS_DIR = "pipeline/output/models"


def get_kmeans(X: np.ndarray = None, n_clusters: int = 5, seed: int = 42) -> KMeans:
    """
    Get or create a KMeans model. 
    """
    global _kmeans_model
    
    model_path = Path(MODELS_DIR) / "kmeans.pkl"
    
    # Return cached model if available
    if _kmeans_model is not None:
        return _kmeans_model
    
    # Try to load from disk
    if model_path.exists():
        logger.info(f"Loading KMeans model from {model_path}...")
        _kmeans_model = joblib.load(model_path)
        logger.info("KMeans model loaded successfully.")
        return _kmeans_model
    
    # Train new model if X is provided
    if X is None:
        raise ValueError("No saved KMeans model found and no training data (X) provided. "
                        "Please provide training data to train a new model.")
    
    logger.info(f"Training KMeans with n_clusters={n_clusters}...")
    _kmeans_model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    _kmeans_model.fit(X)
    logger.info(f"KMeans trained. Cluster distribution: {np.bincount(_kmeans_model.labels_)}")
    
    # Save model to disk
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(_kmeans_model, model_path)
    logger.info(f"KMeans model saved to {model_path}")
    
    return _kmeans_model


def create_classifier_features(user_vector: np.ndarray, train_matrix: np.ndarray,
                                            item_factors: np.ndarray, svd_model: TruncatedSVD,
                                            item_idx: int) -> np.ndarray:
    """
    Create feature vector for a user-item pair using user's interaction vector.
    """
    user_activity = user_vector.sum()
    item_popularity = train_matrix[:, item_idx].sum()
    
    # Transform user vector to latent space
    user_latent = svd_model.transform(user_vector.reshape(1, -1))[0]
    item_latent = item_factors[item_idx]
    latent_interaction = user_latent * item_latent

    features = np.concatenate([
        [user_activity, item_popularity],
        user_latent,
        item_latent,
        latent_interaction
    ])
    return features

def create_classifier_features_with_clustered(
    user_vector: np.ndarray, 
    train_matrix: np.ndarray,
    item_factors: np.ndarray, 
    svd_model: TruncatedSVD,
    item_idx: int,
    cluster_label: int,
    n_clusters: int = 5
) -> np.ndarray:
    """
    Create feature vector for a user-item pair with cluster label as additional feature.
    Expects pre-computed cluster_label for performance.
    """
    user_activity = user_vector.sum()
    item_popularity = train_matrix[:, item_idx].sum()
    
    user_latent = svd_model.transform(user_vector.reshape(1, -1))[0]
    item_latent = item_factors[item_idx]
    latent_interaction = user_latent * item_latent
    
    # One-hot encode the cluster label
    cluster_one_hot = np.zeros(n_clusters)
    cluster_one_hot[cluster_label] = 1
    
    features = np.concatenate([
        [user_activity, item_popularity],
        user_latent,
        item_latent,
        latent_interaction,
        cluster_one_hot
    ])
    return features


def prepare_classifier_training_data(train_matrix: np.ndarray, n_factors: int,
                                      n_neg_samples: int, seed: int = 42, 
                                      is_clustered: bool = False, n_clusters: int = 5) -> tuple:
    """
    Prepare SVD factors and training samples for classifier-based recommenders.
    """
    n_users, n_items = train_matrix.shape

    # Compute SVD factors
    n_components = min(n_factors, min(train_matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(train_matrix)
    item_factors = svd.components_.T

    # Generate training samples
    np.random.seed(seed)
    X_train, y_train = [], []
    
    mode_str = "clustered" if is_clustered else "standard"
    logger.info(f"Generating {mode_str} training samples for {n_users} users...")
    
    # Pre-compute cluster labels for all users (MUCH faster than computing per-sample)
    user_cluster_labels = None
    if is_clustered:
        logger.info("Pre-computing cluster labels from raw user interaction vectors...")
        kmeans = get_kmeans(train_matrix, n_clusters=n_clusters, seed=seed)
        user_cluster_labels = kmeans.labels_
        logger.info("Cluster labels computed.")

    for user_idx in range(n_users):
        if user_idx % 500 == 0:
            logger.info(f"  Processing user {user_idx}/{n_users}...")
        user_vector = train_matrix[user_idx]
        pos_items = np.where(user_vector > 0)[0]
        if len(pos_items) == 0:
            continue

        # Positive samples
        for item_idx in pos_items:
            if is_clustered:
                features = create_classifier_features_with_clustered(
                    user_vector, train_matrix, item_factors, svd, item_idx,
                    cluster_label=user_cluster_labels[user_idx], n_clusters=n_clusters
                )
            else:
                features = create_classifier_features(
                    user_vector, train_matrix, item_factors, svd, item_idx
                )
            X_train.append(features)
            y_train.append(1)

        # Negative samples
        neg_items = np.where(user_vector == 0)[0]
        n_neg = min(len(neg_items), len(pos_items) * n_neg_samples)

        if n_neg > 0:
            sampled_neg = np.random.choice(neg_items, size=n_neg, replace=False)
            for item_idx in sampled_neg:
                if is_clustered:
                    features = create_classifier_features_with_clustered(
                        user_vector, train_matrix, item_factors, svd, item_idx,
                        cluster_label=user_cluster_labels[user_idx], n_clusters=n_clusters
                    )
                else:
                    features = create_classifier_features(
                        user_vector, train_matrix, item_factors, svd, item_idx
                    )
                X_train.append(features)
                y_train.append(0)

    # Check class distribution
    class_dist = pd.Series(y_train).value_counts(normalize=True)
    logger.info("Class distribution:")
    logger.info(f"\n{class_dist}")

    return np.array(X_train), np.array(y_train), svd, item_factors

