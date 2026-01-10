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
    DecisionTreeRecommender
)
from feature_engineering import (
    create_classifier_features,
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
    # Data modeling
    'create_classifier_features',
    'prepare_classifier_training_data',
    # Evaluation
    'evaluate_classifiers',
    'evaluate_recommenders',
]
