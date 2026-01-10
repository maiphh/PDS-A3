"""
Demo CLI for testing recommender models on new users.
"""

import os
import sys
import random
import numpy as np

# Add pipeline directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dst, convert_to_wide_dataframe
from models import BaseRecommender

# Paths
TEST_DATA_FILE = "pipeline/data/anonymous-msweb.test"
MODELS_DIR = "pipeline/output/models"


def load_test_data(test_file: str = TEST_DATA_FILE) -> tuple:
    """
    Load test data and return interactions DataFrame and attributes DataFrame.
    """
    print(f"Loading test data from {test_file}...")
    test_df, attr_df = load_dst(test_file)
    wide_df = convert_to_wide_dataframe(test_df, user_col='case_id', item_col='attr_id')
    print(f"Loaded {len(wide_df)} test users with {len(wide_df.columns)} items")
    return test_df, attr_df, wide_df


def load_all_models(models_dir: str = MODELS_DIR) -> list:
    """
    Load all saved models from the output directory.
    """
    models = []
    
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' not found!")
        print("Please run main.py first to train and save models.")
        return models
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print(f"No model files found in {models_dir}")
        return models
    
    print(f"Loading {len(model_files)} models...")
    for model_file in model_files:
        filepath = os.path.join(models_dir, model_file)
        try:
            model = BaseRecommender.load(filepath)
            models.append(model)
        except Exception as e:
            print(f"  Failed to load {model_file}: {e}")
    
    print(f"Successfully loaded {len(models)} models")
    return models


def create_user_vector(user_items: list, n_items: int, item_to_idx: dict) -> np.ndarray:
    """
    Create a user interaction vector from a list of item IDs.
    """
    user_vector = np.zeros(n_items)
    for item_id in user_items:
        if item_id in item_to_idx:
            user_vector[item_to_idx[item_id]] = 1.0
    return user_vector


def get_item_names(item_indices: list, idx_to_item: dict, attr_df) -> list:
    """
    Convert item indices to item names (titles and URLs).
    """
    attr_dict = {row['attr_id']: (row['title'], row['url']) 
                 for _, row in attr_df.iterrows()}
    
    result = []
    for idx in item_indices:
        attr_id = idx_to_item.get(idx)
        if attr_id and attr_id in attr_dict:
            title, url = attr_dict[attr_id]
            result.append((attr_id, title, url))
        else:
            result.append((idx, "Unknown", ""))
    return result


def display_recommendations(models: list, user_vector: np.ndarray, 
                           idx_to_item: dict, attr_df, n: int = 5):
    """
    Display recommendations from all models for a given user vector.
    """
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    for model in models:
        try:
            recs = model.recommend(user_vector, n=n)
            items = get_item_names(recs, idx_to_item, attr_df)
            
            print(f"\n{model.name}:")
            for i, (attr_id, title, url) in enumerate(items, 1):
                print(f"   {i}. [{attr_id}] {title}")
                if url:
                    print(f"      URL: https://microsoft.com{url}")
        except Exception as e:
            print(f"\n{model.name}: Error - {e}")


def display_menu():
    """Display the main menu."""
    print("\n" + "-" * 50)
    print("RECOMMENDER DEMO")
    print("-" * 50)
    print("Commands:")
    print("  [number]  - Enter a test user index (0 to N-1)")
    print("  r         - Random test user")
    print("  q         - Quit")
    print("-" * 50)


def main():
    """Main demo CLI loop."""
    print("=" * 70)
    print("RECOMMENDER SYSTEM DEMO")
    print("=" * 70)
    
    # Load test data
    try:
        test_df, attr_df, test_wide_df = load_test_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure test data exists at the specified path.")
        return
    
    # Load models
    models = load_all_models()
    if not models:
        return
    sample_model = models[0]
    n_items = sample_model.train_matrix.shape[1]
    test_item_ids = test_wide_df.columns.tolist()
    item_to_idx = {item_id: idx for idx, item_id in enumerate(test_item_ids) if idx < n_items}
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    test_users = test_wide_df.values
    test_user_ids = test_wide_df.index.tolist()
    n_users = len(test_users)
    
    print(f"\nReady! {n_users} test users available.")
    
    # Main loop
    while True:
        display_menu()
        user_input = input("Enter command: ").strip().lower()
        
        if user_input == 'q':
            print("Goodbye!")
            break
        
        elif user_input == 'r':
            # Random user
            idx = random.randint(0, n_users - 1)
            print(f"\nSelected random user index: {idx} (User ID: {test_user_ids[idx]})")
        
        elif user_input.isdigit():
            idx = int(user_input)
            if 0 <= idx < n_users:
                print(f"\nSelected user index: {idx} (User ID: {test_user_ids[idx]})")
            else:
                print(f"Invalid index. Please enter 0 to {n_users - 1}")
                continue
        
        else:
            print("Invalid command. Try again.")
            continue
        
        # Get user's visited items
        user_row = test_wide_df.iloc[idx]
        visited_items = user_row[user_row > 0].index.tolist()
        
        # Display user's history
        print(f"\nUser History ({len(visited_items)} pages visited):")
        visited_info = get_item_names(
            [item_to_idx.get(item_id, -1) for item_id in visited_items[:5]], 
            idx_to_item, attr_df
        )
        for attr_id, title, url in visited_info:
            print(f"   • [{attr_id}] {title}")
        if len(visited_items) > 5:
            print(f"   ... and {len(visited_items) - 5} more")
        
        # Create user vector aligned with model's dimensions (float64 to match KMeans)
        user_vector = np.zeros(n_items, dtype=np.float64)
        for item_id in visited_items:
            if item_id in item_to_idx and item_to_idx[item_id] < n_items:
                user_vector[item_to_idx[item_id]] = 1.0
        
        # Get and display recommendations
        display_recommendations(models, user_vector, idx_to_item, attr_df, n=5)


if __name__ == "__main__":
    main()
