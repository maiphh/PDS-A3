"""
Evaluation functions for recommender systems.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, ndcg_score
)

from feature_engineering import create_classifier_features


def evaluate_classifiers(models: list, train_matrix: np.ndarray, test_data: dict, 
                         n_neg_samples: int = 5) -> pd.DataFrame:
    """
    Evaluate classifier performance on held-out test data.
    Uses test_data (held-out user-item pairs) as positive samples.
    
    Parameters:
    -----------
    models : list
        List of classifier-based recommender models
    train_matrix : np.ndarray
        Training user-item interaction matrix
    test_data : dict
        Dictionary mapping user_idx to list of held-out item indices
    n_neg_samples : int
        Number of negative samples per positive sample for testing
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with evaluation metrics for each model
    """
    results = []

    for model in models:
        print(f"Evaluating classifier: {model.name}...")

        np.random.seed(model.seed)
        X_test, y_test = [], []

        for user_idx, test_items in test_data.items():
            # Positive samples: held-out items from test_data
            for item_idx in test_items:
                X_test.append(create_classifier_features(
                    model.train_matrix, model.user_factors, model.item_factors, user_idx, item_idx
                ))
                y_test.append(1)

            # Negative samples: items the user never interacted with
            # (not in train_matrix AND not in test_data)
            all_items = set(range(train_matrix.shape[1]))
            known_items = set(np.where(train_matrix[user_idx] > 0)[0])
            test_items_set = set(test_items)
            neg_candidates = list(all_items - known_items - test_items_set)

            n_neg = min(len(neg_candidates), len(test_items) * n_neg_samples)
            if n_neg > 0:
                sampled_neg = np.random.choice(neg_candidates, size=n_neg, replace=False)
                for item_idx in sampled_neg:
                    X_test.append(create_classifier_features(
                        model.train_matrix, model.user_factors, model.item_factors, user_idx, item_idx
                    ))
                    y_test.append(0)

        X_test, y_test = np.array(X_test), np.array(y_test)

        # Predict using the trained model
        y_pred = model.model.predict(X_test)
        y_prob = model.model.predict_proba(X_test)[:, 1]

        # Compute metrics
        results.append({
            "Model": model.name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })

        # Print detailed report
        print(f"\n--- {model.name} Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=['No Interact', 'Interact']))

    return pd.DataFrame(results).set_index("Model").round(4)


def evaluate_recommenders(models: list, test_data: dict, 
                          k_values: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Evaluate a list of recommender models and return comparison DataFrame.
    
    Parameters:
    -----------
    models : list
        List of recommender models implementing recommend() method
    test_data : dict
        Dictionary mapping user_idx to list of relevant item indices
    k_values : list
        List of k values for precision@k, recall@k, etc.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with evaluation metrics for each model
    """
    results = []
    max_k = max(k_values)

    for model in models:
        print(f"Evaluating {model.name}...")
        metrics = {f"precision@{k}": [] for k in k_values}
        metrics.update({f"recall@{k}": [] for k in k_values})
        metrics.update({f"hit_rate@{k}": [] for k in k_values})
        metrics.update({f"ndcg@{k}": [] for k in k_values})

        for user_idx, relevant in test_data.items():
            recs = model.recommend(user_idx, n=max_k)
            relevant_set = set(relevant)

            for k in k_values:
                recs_k = recs[:k]
                hits = len(set(recs_k) & relevant_set)

                metrics[f"precision@{k}"].append(hits / k)
                metrics[f"recall@{k}"].append(hits / len(relevant) if relevant else 0)
                metrics[f"hit_rate@{k}"].append(1.0 if hits > 0 else 0.0)

                # NDCG using sklearn
                y_true = np.array([1 if i in relevant_set else 0 for i in recs_k]).reshape(1, -1)
                y_score = np.array([k - i for i in range(k)]).reshape(1, -1)  # rank scores
                if y_true.sum() > 0:
                    metrics[f"ndcg@{k}"].append(ndcg_score(y_true, y_score))
                else:
                    metrics[f"ndcg@{k}"].append(0.0)

        row = {"Model": model.name}
        row.update({m: np.mean(v) for m, v in metrics.items()})
        results.append(row)

    return pd.DataFrame(results).set_index("Model").round(4)
