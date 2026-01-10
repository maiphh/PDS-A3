"""
Recommender Systems for Microsoft Web Data

Two approaches implemented:
1. User-Based Collaborative Filtering (Memory-based)
2. SVD Matrix Factorization (Model-based)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_dst_to_dataframes(file_path):
    """
    Parses a DST file and returns two DataFrames:
    1. df_interactions: User-Item visits (C and V lines)
    2. df_attributes: Page metadata (A lines)
    """
    interactions = []
    attributes = []
    current_user_id = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            parts = line.strip().split(',')
            prefix = parts[0]
            
            if prefix == 'A':
                attributes.append({
                    'attr_id': int(parts[1]),
                    'title': parts[3].strip('"'),
                    'url': parts[4].strip('"')
                })
            elif prefix == 'C':
                current_user_id = int(parts[2])
            elif prefix == 'V' and current_user_id is not None:
                interactions.append({
                    'case_id': current_user_id,
                    'attr_id': int(parts[1]),
                })
    
    return pd.DataFrame(interactions), pd.DataFrame(attributes)


def convert_to_wide_dataframe(df, user_col='case_id', item_col='attr_id'):
    """Convert long-format interactions to wide user-item matrix."""
    return pd.crosstab(df[user_col], df[item_col])


def create_train_test_split(df, test_ratio=0.2, random_state=42):
    """
    For each user, hide some interactions for testing.
    Returns train_df and test_df (same shape, test contains held-out items).
    """
    np.random.seed(random_state)
    train = df.copy()
    test = pd.DataFrame(0, index=df.index, columns=df.columns)
    
    for user in df.index:
        interacted_items = df.columns[df.loc[user] == 1].tolist()
        
        if len(interacted_items) >= 2:
            n_test = max(1, int(len(interacted_items) * test_ratio))
            test_items = np.random.choice(interacted_items, size=n_test, replace=False)
            
            train.loc[user, test_items] = 0
            test.loc[user, test_items] = 1
    
    return train, test


# =============================================================================
# 1. User-Based Collaborative Filtering
# =============================================================================

class UserBasedCF:
    """
    User-Based Collaborative Filtering Recommender.
    
    Recommends items based on what similar users have liked.
    Uses cosine similarity to find k most similar users.
    """
    
    def __init__(self, k_neighbors=20):
        self.k = k_neighbors
        self.user_similarity = None
        self.train_matrix = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, train_df):
        """Compute user-user similarity matrix."""
        self.train_matrix = train_df.values
        self.user_ids = train_df.index
        self.item_ids = train_df.columns
        
        # Compute cosine similarity between users
        self.user_similarity = cosine_similarity(self.train_matrix)
        np.fill_diagonal(self.user_similarity, 0)  # Don't consider self-similarity
        
        print(f"[User-Based CF] Fitted on {len(self.user_ids)} users, {len(self.item_ids)} items")
        return self
    
    def predict(self, user_idx, top_n=10):
        """Predict top-N items for a user (by index)."""
        sim_scores = self.user_similarity[user_idx]
        top_k_users = np.argsort(sim_scores)[-self.k:]
        
        # Weighted sum of similar users' preferences
        weighted_scores = np.zeros(len(self.item_ids))
        for neighbor in top_k_users:
            weighted_scores += sim_scores[neighbor] * self.train_matrix[neighbor]
        
        # Mask already-interacted items
        weighted_scores[self.train_matrix[user_idx] == 1] = -np.inf
        
        return np.argsort(weighted_scores)[-top_n:][::-1]
    
    def recommend(self, user_id, top_n=10):
        """Get recommendations for a specific user ID."""
        user_idx = list(self.user_ids).index(user_id)
        item_indices = self.predict(user_idx, top_n)
        return [self.item_ids[i] for i in item_indices]


# =============================================================================
# 2. SVD Matrix Factorization
# =============================================================================

class SVDRecommender:
    """
    SVD-based Matrix Factorization Recommender.
    
    Learns latent factors for users and items to predict preferences.
    Uses TruncatedSVD for dimensionality reduction.
    """
    
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.train_matrix = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, train_df):
        """Learn latent factors via SVD."""
        self.train_matrix = train_df.values
        self.user_ids = train_df.index
        self.item_ids = train_df.columns
        
        # Decompose: User-Item = User-Factors @ Factors-Item
        self.user_factors = self.svd.fit_transform(self.train_matrix)
        self.item_factors = self.svd.components_.T
        
        explained_var = self.svd.explained_variance_ratio_.sum()
        print(f"[SVD] Fitted with {self.n_components} components, explained variance: {explained_var:.2%}")
        return self
    
    def predict(self, user_idx, top_n=10):
        """Predict top-N items for a user (by index)."""
        scores = self.user_factors[user_idx] @ self.item_factors.T
        
        # Mask already-interacted items
        scores[self.train_matrix[user_idx] == 1] = -np.inf
        
        return np.argsort(scores)[-top_n:][::-1]
    
    def recommend(self, user_id, top_n=10):
        """Get recommendations for a specific user ID."""
        user_idx = list(self.user_ids).index(user_id)
        item_indices = self.predict(user_idx, top_n)
        return [self.item_ids[i] for i in item_indices]


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_recommender(model, train_df, test_df, top_n=10):
    """
    Evaluate recommender using:
    - Hit Rate: % of users where at least one test item is recommended
    - Precision@K: % of recommended items that are in test set
    - Recall@K: % of test items that are recommended
    """
    hits = 0
    total_precision = 0
    total_recall = 0
    n_users_evaluated = 0
    
    for user_id in train_df.index:
        actual_items = set(test_df.columns[test_df.loc[user_id] == 1])
        
        if len(actual_items) == 0:
            continue
        
        recommended = set(model.recommend(user_id, top_n))
        hits_for_user = len(recommended & actual_items)
        
        if hits_for_user > 0:
            hits += 1
        
        total_precision += hits_for_user / top_n
        total_recall += hits_for_user / len(actual_items)
        n_users_evaluated += 1
    
    return {
        'Hit Rate': hits / n_users_evaluated,
        f'Precision@{top_n}': total_precision / n_users_evaluated,
        f'Recall@{top_n}': total_recall / n_users_evaluated,
        'Users Evaluated': n_users_evaluated
    }


# =============================================================================
# Main Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RECOMMENDER SYSTEMS DEMO")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    df_interactions, df_attributes = load_dst_to_dataframes('anonymous-msweb.data')
    df_wide = convert_to_wide_dataframe(df_interactions)
    
    # Filter users with at least 2 interactions
    user_interactions = df_wide.sum(axis=1)
    df_wide = df_wide[user_interactions >= 2]
    
    print(f"    Users: {df_wide.shape[0]}, Items: {df_wide.shape[1]}")
    sparsity = 1 - (df_wide.values.sum() / (df_wide.shape[0] * df_wide.shape[1]))
    print(f"    Sparsity: {sparsity:.2%}")
    
    # Train/test split
    print("\n[2] Creating train/test split...")
    train_df, test_df = create_train_test_split(df_wide)
    print(f"    Train interactions: {train_df.values.sum()}")
    print(f"    Test interactions: {test_df.values.sum()}")
    
    # Train models
    print("\n[3] Training models...")
    ubcf = UserBasedCF(k_neighbors=20)
    ubcf.fit(train_df)
    
    svd_rec = SVDRecommender(n_components=50)
    svd_rec.fit(train_df)
    
    # Evaluate
    print("\n[4] Evaluating models...")
    print("\n" + "-" * 40)
    print("User-Based Collaborative Filtering")
    print("-" * 40)
    ubcf_results = evaluate_recommender(ubcf, train_df, test_df, top_n=10)
    for metric, value in ubcf_results.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    
    print("\n" + "-" * 40)
    print("SVD Matrix Factorization")
    print("-" * 40)
    svd_results = evaluate_recommender(svd_rec, train_df, test_df, top_n=10)
    for metric, value in svd_results.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    
    # Demo recommendations
    sample_user = train_df.index[0]
    print(f"\n[5] Sample Recommendations for User {sample_user}")
    print("-" * 40)
    
    print("\nUser-Based CF Top 5:")
    for item_id in ubcf.recommend(sample_user, top_n=5):
        title = df_attributes[df_attributes['attr_id'] == item_id]['title'].values
        title = title[0] if len(title) > 0 else 'Unknown'
        print(f"  {item_id}: {title}")
    
    print("\nSVD Top 5:")
    for item_id in svd_rec.recommend(sample_user, top_n=5):
        title = df_attributes[df_attributes['attr_id'] == item_id]['title'].values
        title = title[0] if len(title) > 0 else 'Unknown'
        print(f"  {item_id}: {title}")
    
    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    comparison = pd.DataFrame({
        'User-Based CF': ubcf_results,
        'SVD': svd_results
    }).T
    print(comparison.to_string())
