"""
Data modeling utilities for classifier-based recommenders.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def create_classifier_features(train_matrix: np.ndarray, user_factors: np.ndarray,
                                item_factors: np.ndarray, user_idx: int, item_idx: int) -> np.ndarray:
    """
    Create feature vector for a user-item pair.
    
    Parameters:
    -----------
    train_matrix : np.ndarray
        User-item interaction matrix
    user_factors : np.ndarray
        User latent factors from SVD
    item_factors : np.ndarray
        Item latent factors from SVD
    user_idx : int
        User index
    item_idx : int
        Item index
        
    Returns:
    --------
    np.ndarray
        Feature vector combining user activity, item popularity, and latent factors
    """
    user_activity = train_matrix[user_idx].sum()
    item_popularity = train_matrix[:, item_idx].sum()
    user_latent = user_factors[user_idx]
    item_latent = item_factors[item_idx]
    latent_interaction = user_latent * item_latent

    features = np.concatenate([
        [user_activity, item_popularity],
        user_latent,
        item_latent,
        latent_interaction
    ])
    return features


def prepare_classifier_training_data(train_matrix: np.ndarray, n_factors: int,
                                      n_neg_samples: int, seed: int = 42) -> tuple:
    """
    Prepare SVD factors and training samples for classifier-based recommenders.

    Parameters:
    -----------
    train_matrix : np.ndarray
        User-item interaction matrix
    n_factors : int
        Number of latent factors for SVD
    n_neg_samples : int
        Number of negative samples per positive sample
    seed : int
        Random seed
        
    Returns:
    --------
    tuple
        (X_train, y_train, user_factors, item_factors)
    """
    n_users, n_items = train_matrix.shape

    # Compute SVD factors
    n_components = min(n_factors, min(train_matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    user_factors = svd.fit_transform(train_matrix)
    item_factors = svd.components_.T

    # Generate training samples
    np.random.seed(seed)
    X_train, y_train = [], []

    for user_idx in range(n_users):
        pos_items = np.where(train_matrix[user_idx] > 0)[0]
        if len(pos_items) == 0:
            continue

        # Positive samples
        for item_idx in pos_items:
            features = create_classifier_features(train_matrix, user_factors, item_factors, user_idx, item_idx)
            X_train.append(features)
            y_train.append(1)

        # Negative samples
        neg_items = np.where(train_matrix[user_idx] == 0)[0]
        n_neg = min(len(neg_items), len(pos_items) * n_neg_samples)

        if n_neg > 0:
            sampled_neg = np.random.choice(neg_items, size=n_neg, replace=False)
            for item_idx in sampled_neg:
                features = create_classifier_features(train_matrix, user_factors, item_factors, user_idx, item_idx)
                X_train.append(features)
                y_train.append(0)

    # Check class distribution
    class_dist = pd.Series(y_train).value_counts(normalize=True)
    print("Class distribution:")
    print(class_dist)

    return np.array(X_train), np.array(y_train), user_factors, item_factors
