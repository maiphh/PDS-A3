"""
Evaluation functions for recommender systems.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, ndcg_score
)

from feature_engineering import create_classifier_features, create_classifier_features_with_clustered, get_kmeans


def evaluate_classifiers(models: list, train_matrix: np.ndarray, test_data: dict, 
                         n_neg_samples: int = 5) -> pd.DataFrame:
    """
    Evaluate classifier performance on held-out test data.
    """
    results = []

    for model in models:
        print(f"Evaluating classifier: {model.name}...")
        
        # Check if this is a clustered model
        is_clustered = hasattr(model, 'n_clusters')
        
        # Pre-compute cluster labels if needed (using raw user vectors)
        user_cluster_labels = None
        if is_clustered:
            kmeans = get_kmeans(model.train_matrix, n_clusters=model.n_clusters, seed=model.seed)
            user_cluster_labels = kmeans.labels_

        np.random.seed(model.seed)
        X_test, y_test = [], []

        for user_idx, test_items in test_data.items():
            user_vector = train_matrix[user_idx]
            
            # Positive samples: held-out items from test_data
            for item_idx in test_items:
                if is_clustered:
                    X_test.append(create_classifier_features_with_clustered(
                        user_vector, model.train_matrix, model.item_factors, model.svd, item_idx,
                        cluster_label=user_cluster_labels[user_idx], n_clusters=model.n_clusters
                    ))
                else:
                    X_test.append(create_classifier_features(
                        user_vector, model.train_matrix, model.item_factors, model.svd, item_idx
                    ))
                y_test.append(1)

            # Negative samples: items the user never interacted with
            # (not in train_matrix AND not in test_data)
            all_items = set(range(train_matrix.shape[1]))
            known_items = set(np.where(user_vector > 0)[0])
            test_items_set = set(test_items)
            neg_candidates = list(all_items - known_items - test_items_set)

            n_neg = min(len(neg_candidates), len(test_items) * n_neg_samples)
            if n_neg > 0:
                sampled_neg = np.random.choice(neg_candidates, size=n_neg, replace=False)
                for item_idx in sampled_neg:
                    if is_clustered:
                        X_test.append(create_classifier_features_with_clustered(
                            user_vector, model.train_matrix, model.item_factors, model.svd, item_idx,
                            cluster_label=user_cluster_labels[user_idx], n_clusters=model.n_clusters
                        ))
                    else:
                        X_test.append(create_classifier_features(
                            user_vector, model.train_matrix, model.item_factors, model.svd, item_idx
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
                          k_values: list = [5, 10, 20],
                          max_users: int = None,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate a list of recommender models and return comparison DataFrame.
    """
    results = []
    max_k = max(k_values)
    
    # Sample users if max_users specified
    test_users = list(test_data.keys())
    if max_users and max_users < len(test_users):
        np.random.seed(42)
        test_users = list(np.random.choice(test_users, size=max_users, replace=False))
        if verbose:
            print(f"Sampling {max_users} users for evaluation (out of {len(test_data)})")

    n_users = len(test_users)

    for model in models:
        if verbose:
            print(f"Evaluating {model.name}...", end=" ", flush=True)
        
        metrics = {f"precision@{k}": [] for k in k_values}
        metrics.update({f"recall@{k}": [] for k in k_values})
        metrics.update({f"hit_rate@{k}": [] for k in k_values})
        metrics.update({f"ndcg@{k}": [] for k in k_values})

        for i, user_idx in enumerate(test_users):
            relevant = test_data[user_idx]
            user_vector = model.train_matrix[user_idx]
            recs = model.recommend(user_vector, n=max_k)
            relevant_set = set(relevant)

            for k in k_values:
                recs_k = recs[:k]
                hits = len(set(recs_k) & relevant_set)

                metrics[f"precision@{k}"].append(hits / k)
                metrics[f"recall@{k}"].append(hits / len(relevant) if relevant else 0)
                metrics[f"hit_rate@{k}"].append(1.0 if hits > 0 else 0.0)

                # NDCG using sklearn
                y_true = np.array([1 if item in relevant_set else 0 for item in recs_k]).reshape(1, -1)
                y_score = np.array([k - j for j in range(k)]).reshape(1, -1)  # rank scores
                if y_true.sum() > 0:
                    metrics[f"ndcg@{k}"].append(ndcg_score(y_true, y_score))
                else:
                    metrics[f"ndcg@{k}"].append(0.0)
            
            # Progress indicator
            if verbose and (i + 1) % 100 == 0:
                print(f"{i + 1}/{n_users}", end=" ", flush=True)
        
        if verbose:
            print("Done!")

        row = {"Model": model.name}
        row.update({m: np.mean(v) for m, v in metrics.items()})
        results.append(row)

    return pd.DataFrame(results).set_index("Model").round(4)

