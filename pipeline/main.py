"""
Main pipeline script for Anonymous Microsoft Web Data recommender systems.
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader import (
    download_data, load_dst, convert_to_wide_dataframe, 
    prepare_train_test_from_wide, create_user_activity_features,
    create_page_popularity_features
)
from models import (
    PopularityRecommender, RandomRecommender, UserBasedCF, 
    ItemBasedCF, SVDRecommender, XGBoostRecommender, DecisionTreeRecommender,
    DecisionTreeClusteredRecommender
)
from evaluation import evaluate_classifiers, evaluate_recommenders


def main():
    """Main pipeline execution."""
    
    # ==========================================================================
    # Task 1: Retrieving and Preparing the Data
    # ==========================================================================
    print("=" * 60)
    print("Task 1: Retrieving and Preparing the Data")
    print("=" * 60)
    
    # Download and load data
    data_file = download_data()
    df, attr_df = load_dst(data_file)
    
    print(f"\nInteractions DataFrame shape: {df.shape}")
    print(f"Attributes DataFrame shape: {attr_df.shape}")
    print(f"\nNumber of unique users: {df['case_id'].nunique()}")
    print(f"Number of unique pages: {attr_df['attr_id'].nunique()}")
    
    # Check for missing values
    print("\nMissing values in interactions:")
    print(df.isnull().sum())
    
    print("\nMissing values in attributes:")
    print(attr_df.isnull().sum())
    
    # Check for duplicates
    duplicates = df.duplicated(subset=["case_id", "attr_id"]).sum()
    print(f"\nDuplicate user-page interactions: {duplicates}")
    
    # ==========================================================================
    # Task 2: Feature Engineering
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Task 2: Feature Engineering")
    print("=" * 60)
    
    # Create wide format matrix
    wide_df = convert_to_wide_dataframe(df, user_col='case_id', item_col='attr_id')
    print(f"\nWide DataFrame shape: {wide_df.shape}")
    
    
    # ==========================================================================
    # Task 3: Modeling
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Task 3: Modeling")
    print("=" * 60)
    
    # Prepare train/test split
    train_matrix, test_data = prepare_train_test_from_wide(wide_df)
    
    # Initialize models
    models = [
        PopularityRecommender(),
        RandomRecommender(),
        UserBasedCF(k=50),
        ItemBasedCF(k=50),
        SVDRecommender(n_factors=50),
        XGBoostRecommender(n_factors=20, n_neg_samples=1),
        DecisionTreeRecommender(n_factors=20, n_neg_samples=1),
        DecisionTreeClusteredRecommender(n_factors=20, n_neg_samples=1, n_clusters=5),
    ]
    
    # Train or load models
    print("\nLoading/Training models...")
    trained_models = []
    for model in models:
        if model.exists():
            # Load existing model
            loaded_model = type(model).load(model.get_model_path())
            trained_models.append(loaded_model)
        else:
            # Train and save new model
            print(f"  Training {model.name}...")
            model.fit(train_matrix)
            model.save()
            trained_models.append(model)
    models = trained_models
    print("Done!")
    
    # ==========================================================================
    # Task 4: Evaluation
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Task 4: Evaluation")
    print("=" * 60)
    
    # Evaluate classifier-based models
    classifier_models = [m for m in models if hasattr(m, 'model') and m.model is not None]
    if classifier_models:
        print("\n--- Classifier Evaluation ---")
        classifier_results = evaluate_classifiers(classifier_models, train_matrix, test_data)
        print("\nClassifier Results:")
        print(classifier_results)
    
    # Evaluate all recommender models
    print("\n--- Recommender Evaluation ---")
    recommender_results = evaluate_recommenders(models, test_data, k_values=[5], max_users=500)
    print("\nRecommender Results:")
    print(recommender_results)
    
    # ==========================================================================
    # Sample Recommendations
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Sample Recommendations")
    print("=" * 60)
    
    # Get a sample user with test data
    sample_user = list(test_data.keys())[0]
    sample_user_vector = train_matrix[sample_user]
    print(f"\nRecommendations for user index {sample_user}:")
    
    for model in models:
        recs = model.recommend(sample_user_vector, n=5)
        print(f"  {model.name}: {recs}")
    
    return models, recommender_results


if __name__ == "__main__":
    models, results = main()
