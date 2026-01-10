"""
Data loading and preprocessing utilities for Anonymous Microsoft Web Data.
"""

import pandas as pd
import numpy as np
import os
import urllib.request


# Dataset URL from UCI Machine Learning Repository
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/anonymous/anonymous-msweb.data"
DATA_FILE = "anonymous-msweb.data"


def download_data(data_url: str = DATA_URL, data_file: str = DATA_FILE) -> str:
    """Download the dataset if it doesn't exist locally."""
    if not os.path.exists(data_file):
        print(f"Downloading {data_file} from UCI repository...")
        urllib.request.urlretrieve(data_url, data_file)
        print(f"Downloaded {data_file} successfully!")
    else:
        print(f"{data_file} already exists locally.")
    return data_file


def load_dst(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses a DST file and returns two DataFrames:
    1. df_interactions: User-Item visits (C and V lines)
    2. df_attributes: Page metadata (A lines)
    
    Parameters:
    -----------
    file_path : str
        Path to the DST file
        
    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (interactions_df, attributes_df)
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

            # 1. Process Attributes (Metadata)
            if prefix == 'A':
                # Format: A, attr_id, ignore, title, url
                attributes.append({
                    'attr_id': int(parts[1]),
                    'title': parts[3].strip('"'),
                    'url': parts[4].strip('"')
                })

            # 2. Process Cases (User IDs)
            elif prefix == 'C':
                # Format: C, "user_id_str", user_id_int
                current_user_id = int(parts[2])

            # 3. Process Votes (The Interaction)
            elif prefix == 'V' and current_user_id is not None:
                # Format: V, attr_id, ignore
                interactions.append({
                    'case_id': current_user_id,
                    'attr_id': int(parts[1]),
                })

    # Convert lists to DataFrames
    df_interactions = pd.DataFrame(interactions)
    df_attributes = pd.DataFrame(attributes)

    return df_interactions, df_attributes


def convert_to_wide_dataframe(df: pd.DataFrame, user_col: str = 'case_id', 
                               item_col: str = 'attr_id') -> pd.DataFrame:
    """
    Convert long-format interaction data to wide (pivot) format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Long-format DataFrame with user-item interactions
    user_col : str
        Column name for user IDs
    item_col : str
        Column name for item IDs
        
    Returns:
    --------
    pd.DataFrame
        Wide-format DataFrame with users as rows and items as columns
    """
    wide_df = pd.crosstab(df[user_col], df[item_col])
    return wide_df


def prepare_train_test_from_wide(wide_df: pd.DataFrame, test_ratio: float = 0.2,
                                  min_interactions: int = 2, seed: int = 42) -> tuple[np.ndarray, dict]:
    """
    Split wide_df into train matrix and test dict.
    
    Parameters:
    -----------
    wide_df : pd.DataFrame
        Wide-format user-item interaction matrix
    test_ratio : float
        Proportion of interactions to hold out for testing
    min_interactions : int
        Minimum interactions required for a user to be included in test
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple[np.ndarray, dict]
        Tuple of (train_matrix, test_data) where test_data maps user_idx to list of held-out item indices
    """
    np.random.seed(seed)

    matrix = wide_df.values.astype(np.float32)
    train_matrix = matrix.copy()
    test_data = {}

    for user_idx in range(matrix.shape[0]):
        item_indices = np.where(matrix[user_idx] > 0)[0].tolist()

        if len(item_indices) >= min_interactions:
            np.random.shuffle(item_indices)
            n_test = max(1, int(len(item_indices) * test_ratio))
            test_data[user_idx] = item_indices[:n_test]
            # Remove test items from train matrix
            for idx in item_indices[:n_test]:
                train_matrix[user_idx, idx] = 0.0

    print(f"Train: {int(train_matrix.sum())} interactions | Test: {len(test_data)} users, {sum(len(v) for v in test_data.values())} interactions")
    return train_matrix, test_data


def create_user_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create user activity features including activity level categorization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Interaction DataFrame with 'case_id' column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with user_id, num_pages_visited, and activity_level
    """
    # Compute the number of unique pages visited by each user
    user_activity = df.groupby("case_id").size()

    # Convert to DataFrame
    user_activity_df = user_activity.reset_index()
    user_activity_df.columns = ["case_id", "num_pages_visited"]

    # Categorise users based on number of pages visited
    user_activity_df["activity_level"] = pd.cut(
        user_activity_df["num_pages_visited"],
        bins=[0, 1, 3, 10, user_activity_df["num_pages_visited"].max()],
        labels=["Very Low", "Low", "Medium", "High"]
    )

    return user_activity_df


def create_page_popularity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create page popularity features including popularity level categorization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Interaction DataFrame with 'attr_id' and 'case_id' columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with attr_id, num_users_visited, and popularity_level
    """
    # Number of unique users visiting each page
    page_popularity = df.groupby("attr_id")["case_id"].nunique()

    # Convert to DataFrame
    page_popularity_df = page_popularity.reset_index()
    page_popularity_df.columns = ["attr_id", "num_users_visited"]

    # Categorise pages by popularity
    page_popularity_df["popularity_level"] = pd.qcut(
        page_popularity_df["num_users_visited"],
        q=4,
        labels=["Low", "Medium", "High", "Very High"]
    )

    return page_popularity_df
