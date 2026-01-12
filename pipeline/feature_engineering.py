"""
Data modeling utilities for classifier-based recommenders.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from logger import setup_logger

# Initialize logger
logger = setup_logger()

# Module-level cache for KMeans model
_kmeans_model = None

# Path for saving/loading models
MODELS_DIR = "pipeline/output/models"
EVALUATIONS_DIR = "pipeline/output/evaluations"


def find_optimal_k_elbow(X: np.ndarray, k_range: range = range(2, 11), seed: int = 42) -> int:
    """
    Find optimal number of clusters using the elbow method.
    Saves the elbow plot to the evaluations folder.
    
    Returns the optimal k based on the elbow point.
    """
    X_float64 = X.astype(np.float64)
    inertias = []
    
    logger.info(f"Running elbow method for k in range {k_range.start} to {k_range.stop - 1}...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        kmeans.fit(X_float64)
        inertias.append(kmeans.inertia_)
        logger.info(f"  k={k}: inertia={kmeans.inertia_:.2f}")
    
    # Find elbow point using the "knee" detection method
    # Calculate the rate of change (first derivative)
    k_values = list(k_range)
    
    # Use the elbow detection: find where the decrease in inertia slows down significantly
    # Calculate second derivative to find the point of maximum curvature
    first_derivative = np.diff(inertias)
    second_derivative = np.diff(first_derivative)
    
    # The elbow is where the second derivative is maximum (steepest change in slope)
    elbow_idx = np.argmax(second_derivative) + 1  # +1 because we lose one element in diff
    optimal_k = k_values[elbow_idx]
    
    logger.info(f"Optimal k detected: {optimal_k}")
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k = {optimal_k}')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
    plt.title('Elbow Method for Optimal k Selection', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # Save the plot
    Path(EVALUATIONS_DIR).mkdir(parents=True, exist_ok=True)
    plot_path = Path(EVALUATIONS_DIR) / "kmeans_elbow_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Elbow plot saved to {plot_path}")
    
    return optimal_k


def get_kmeans(X: np.ndarray = None, n_clusters: int = None, seed: int = 42) -> KMeans:
    """
    Get or create a KMeans model. 
    If not trained and n_clusters is not specified, uses elbow method to find optimal k.
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
    
    # Use elbow method to find optimal k if not specified
    if n_clusters is None:
        n_clusters = find_optimal_k_elbow(X, seed=seed)
    
    logger.info(f"Training KMeans with n_clusters={n_clusters}...")
    _kmeans_model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    # Ensure float64 dtype to prevent "const double" mismatch errors
    _kmeans_model.fit(X.astype(np.float64))
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
                                      is_clustered: bool = False, n_clusters: int = None) -> tuple:
    """
    Prepare SVD factors and training samples for classifier-based recommenders.
    If n_clusters is None, elbow method will be used to find optimal k.
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
    actual_n_clusters = None
    if is_clustered:
        logger.info("Pre-computing cluster labels from raw user interaction vectors...")
        kmeans = get_kmeans(train_matrix, n_clusters=n_clusters, seed=seed)
        user_cluster_labels = kmeans.labels_
        # Get actual n_clusters from trained model (important when elbow method was used)
        actual_n_clusters = kmeans.n_clusters
        logger.info(f"Cluster labels computed. Using {actual_n_clusters} clusters.")

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
                    cluster_label=user_cluster_labels[user_idx], n_clusters=actual_n_clusters
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
                        cluster_label=user_cluster_labels[user_idx], n_clusters=actual_n_clusters
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

