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
    ItemBasedCF, XGBoostRecommender, DecisionTreeRecommender,
    DecisionTreeClusteredRecommender
)
from evaluation import evaluate_classifiers, evaluate_recommenders

from utils import load_all_models

from logger import setup_logger

# Initialize logger
logger = setup_logger()

def main():
    """Main pipeline execution."""
    
    # ==========================================================================
    # Task 1: Retrieving and Preparing the Data
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Task 1: Retrieving and Preparing the Data")
    logger.info("=" * 60)
    
    # Download and load data
    data_file = download_data()
    df, attr_df = load_dst(data_file)
    
    logger.info(f"Interactions DataFrame shape: {df.shape}")
    logger.info(f"Attributes DataFrame shape: {attr_df.shape}")
    logger.info(f"Number of unique users: {df['case_id'].nunique()}")
    logger.info(f"Number of unique pages: {attr_df['attr_id'].nunique()}")
    
    # Check for missing values
    logger.info("Missing values in interactions:")
    logger.info(f"\n{df.isnull().sum()}")
    
    logger.info("Missing values in attributes:")
    logger.info(f"\n{attr_df.isnull().sum()}")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=["case_id", "attr_id"]).sum()
    logger.info(f"Duplicate user-page interactions: {duplicates}")
    
    # ==========================================================================
    # Task 2: Feature Engineering
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Task 2: Feature Engineering")
    logger.info("=" * 60)
    
    # Create wide format matrix
    wide_df = convert_to_wide_dataframe(df, user_col='case_id', item_col='attr_id')
    logger.info(f"Wide DataFrame shape: {wide_df.shape}")
    
    
    # ==========================================================================
    # Task 3: Modeling
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Task 3: Modeling")
    logger.info("=" * 60)
    
    # Prepare train/test split
    train_matrix, test_data = prepare_train_test_from_wide(wide_df)
    
    # Initialize models
    models = load_all_models(train_matrix=train_matrix)
    
    # ==========================================================================
    # Task 4: Evaluation
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Task 4: Evaluation")
    logger.info("=" * 60)
    
    # Define output directory for evaluations
    eval_output_dir = "pipeline/output/evaluations"

    # Evaluate classifier-based models
    classifier_models = [m for m in models if hasattr(m, 'model') and m.model is not None]
    if classifier_models:
        logger.info("--- Classifier Evaluation ---")
        classifier_results = evaluate_classifiers(classifier_models, train_matrix, test_data, output_dir=eval_output_dir)
        logger.info("Classifier Results:")
        logger.info(f"\n{classifier_results}")
    
    # Evaluate all recommender models
    logger.info("--- Recommender Evaluation ---")
    recommender_results = evaluate_recommenders(models, test_data, k_values=[5], max_users=500, output_dir=eval_output_dir)
    logger.info("Recommender Results:")
    logger.info(f"\n{recommender_results}")
    
    # ==========================================================================
    # Sample Recommendations
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Sample Recommendations")
    logger.info("=" * 60)
    
    # Get a sample user with test data
    sample_user = list(test_data.keys())[0]
    sample_user_vector = train_matrix[sample_user]
    logger.info(f"Recommendations for user index {sample_user}:")
    
    for model in models:
        recs = model.recommend(sample_user_vector, n=5)
        logger.info(f"{model.name}: {recs}")
    
    return models, recommender_results


if __name__ == "__main__":
    main()
