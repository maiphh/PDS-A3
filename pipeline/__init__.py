"""
Pipeline module for Anonymous Microsoft Web Data recommender systems.
"""

from .data_loader import load_dst, convert_to_wide_dataframe, prepare_train_test_from_wide
from .models import (
    BaseRecommender,
    PopularityRecommender,
    RandomRecommender,
    UserBasedCF,
    ItemBasedCF,
    SVDRecommender,
    XGBoostRecommender,
    DecisionTreeRecommender,
    DecisionTreeClusteredRecommender
)
from .feature_engineering import (
    get_kmeans,
    create_classifier_features,
    create_classifier_features_with_clustered,
    prepare_classifier_training_data
)
from .evaluation import evaluate_classifiers, evaluate_recommenders

__all__ = [
    # Data loader
    'load_dst',
    'convert_to_wide_dataframe',
    'prepare_train_test_from_wide',
    # Models
    'BaseRecommender',
    'PopularityRecommender',
    'RandomRecommender',
    'UserBasedCF',
    'ItemBasedCF',
    'SVDRecommender',
    'XGBoostRecommender',
    'DecisionTreeRecommender',
    # Feature engineering
    'get_kmeans',
    'create_classifier_features',
    'create_classifier_features_with_clustered',
    'prepare_classifier_training_data',
    # Evaluation
    'evaluate_classifiers',
    'evaluate_recommenders',
]

