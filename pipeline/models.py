"""
Recommender system model classes.
"""

from abc import ABC, abstractmethod
import os
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from feature_engineering import ( 
    create_classifier_features,
    create_classifier_features_with_clustered,
    prepare_classifier_training_data
)
from logger import setup_logger

# Initialize logger
logger = setup_logger()

# Default output directory for saved models
MODEL_OUTPUT_DIR = "pipeline/output/models"


class BaseRecommender(ABC):
    """Abstract base class for all recommender models."""

    def __init__(self, name: str):
        self.name = name
        self.train_matrix = None

    @abstractmethod
    def fit(self, train_matrix: np.ndarray):
        """Train the model on user-item interaction matrix."""
        pass

    @abstractmethod
    def recommend(self, user_vector: np.ndarray, n: int = 10) -> list:
        """
        Generate top-n recommendations for a user (excluding known items).
        
        Parameters:
        -----------
        user_vector : np.ndarray
            User's interaction vector (1D array of shape [n_items])
        n : int
            Number of recommendations to return
            
        Returns:
        --------
        list
            List of recommended item indices
        """
        pass

    def get_model_path(self, output_dir: str = None) -> str:
        """Get the default file path for this model."""
        output_dir = output_dir or MODEL_OUTPUT_DIR
        # Sanitize model name for filename
        safe_name = self.name.replace("(", "_").replace(")", "").replace("=", "")
        return os.path.join(output_dir, f"{safe_name}.pkl")

    def save(self, filepath: str = None):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the model. If None, uses default path.
        """
        filepath = filepath or self.get_model_path()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model file.
            
        Returns:
        --------
        BaseRecommender
            The loaded model instance.
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model

    def exists(self, output_dir: str = None) -> bool:
        """Check if a saved model file exists."""
        return os.path.exists(self.get_model_path(output_dir))


class PopularityRecommender(BaseRecommender):
    """Recommends most popular items to all users."""

    def __init__(self):
        super().__init__("Popularity")

    def fit(self, train_matrix: np.ndarray):
        self.train_matrix = train_matrix
        self.popular_items = np.argsort(train_matrix.sum(axis=0))[::-1]
        return self

    def recommend(self, user_vector: np.ndarray, n: int = 10) -> list:
        known = set(np.where(user_vector > 0)[0])
        return [int(i) for i in self.popular_items if i not in known][:n]


class RandomRecommender(BaseRecommender):
    """Random baseline recommender."""

    def __init__(self, seed: int = 42):
        super().__init__("Random")
        self.rng = np.random.RandomState(seed)

    def fit(self, train_matrix: np.ndarray):
        self.train_matrix = train_matrix
        self.n_items = train_matrix.shape[1]
        return self

    def recommend(self, user_vector: np.ndarray, n: int = 10) -> list:
        known = set(np.where(user_vector > 0)[0])
        candidates = [i for i in range(self.n_items) if i not in known]
        self.rng.shuffle(candidates)
        return candidates[:n]


class UserBasedCF(BaseRecommender):
    """User-based Collaborative Filtering."""

    def __init__(self, k: int = 50):
        super().__init__(f"UserCF(k={k})")
        self.k = k

    def fit(self, train_matrix: np.ndarray):
        self.train_matrix = train_matrix
        self.user_sim = cosine_similarity(train_matrix)
        np.fill_diagonal(self.user_sim, 0)
        return self

    def recommend(self, user_vector: np.ndarray, n: int = 10) -> list:
        # Compute similarity between input user and all training users
        sim = cosine_similarity(user_vector.reshape(1, -1), self.train_matrix)[0]
        top_users = np.argsort(sim)[::-1][:self.k]

        scores = np.zeros(self.train_matrix.shape[1])
        for u in top_users:
            scores += sim[u] * self.train_matrix[u]

        known = np.where(user_vector > 0)[0]
        scores[known] = -np.inf
        return np.argsort(scores)[::-1][:n].tolist()


class ItemBasedCF(BaseRecommender):
    """Item-based Collaborative Filtering."""

    def __init__(self, k: int = 50):
        super().__init__(f"ItemCF(k={k})")
        self.k = k

    def fit(self, train_matrix: np.ndarray):
        self.train_matrix = train_matrix
        self.item_sim = cosine_similarity(train_matrix.T)
        np.fill_diagonal(self.item_sim, 0)
        return self

    def recommend(self, user_vector: np.ndarray, n: int = 10) -> list:
        user_items = np.where(user_vector > 0)[0]

        if len(user_items) == 0:
            return np.argsort(self.train_matrix.sum(axis=0))[::-1][:n].tolist()

        scores = self.item_sim[user_items].sum(axis=0)
        scores[user_items] = -np.inf
        return np.argsort(scores)[::-1][:n].tolist()


class XGBoostRecommender(BaseRecommender):
    """XGBoost-based recommender using binary classification."""

    def __init__(self, n_factors: int = 20, n_neg_samples: int = 5, seed: int = 42):
        super().__init__(f"XGBoost(f={n_factors})")
        self.n_factors = n_factors
        self.n_neg_samples = n_neg_samples
        self.seed = seed
        self.model = None
        self.best_params_ = None

    def fit(self, train_matrix: np.ndarray, tune: bool = True, param_dist: dict = None, 
            cv: int = 3, n_iter: int = 10):
        """
        Train the model. Optionally tune hyperparameters.

        Parameters:
        -----------
        train_matrix : np.ndarray
        tune : bool - Whether to perform hyperparameter tuning
        param_dist : dict - Parameter distributions for RandomizedSearchCV (uses default if None)
        cv : int - Number of cross-validation folds
        n_iter : int - Number of parameter combinations to try
        """
        self.train_matrix = train_matrix
        
        # Use shared utility function
        X_train, y_train, self.svd, self.item_factors = prepare_classifier_training_data(
            train_matrix, self.n_factors, self.n_neg_samples, self.seed
        )

        if tune:
            if param_dist is None:
                param_dist = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.1, 0.2],
                }

            logger.info(f"Tuning {self.name} with RandomizedSearchCV (n_iter={n_iter})...")
            base_model = XGBClassifier(random_state=self.seed, eval_metric='logloss')
            random_search = RandomizedSearchCV(
                base_model, param_dist, n_iter=n_iter, cv=cv,
                scoring='roc_auc', n_jobs=-1, verbose=1, random_state=self.seed
            )
            random_search.fit(X_train, y_train)

            self.model = random_search.best_estimator_
            self.best_params_ = random_search.best_params_
            logger.info(f"Best params: {self.best_params_}")
            logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        else:
            self.model = XGBClassifier(random_state=self.seed)
            self.model.fit(X_train, y_train)

        return self

    def recommend(self, user_vector: np.ndarray, n: int = 10) -> list:
        known_items = set(np.where(user_vector > 0)[0])
        candidate_items = [i for i in range(self.train_matrix.shape[1]) if i not in known_items]

        if len(candidate_items) == 0:
            return []

        X_pred = np.array([
            create_classifier_features(
                user_vector, self.train_matrix, self.item_factors, self.svd, item_idx
            )
            for item_idx in candidate_items
        ])
        scores = self.model.predict_proba(X_pred)[:, 1]
        top_indices = np.argsort(scores)[::-1][:n]
        return [candidate_items[i] for i in top_indices]


class DecisionTreeRecommender(BaseRecommender):
    """Decision Tree-based recommender using binary classification."""

    def __init__(self, n_factors: int = 20, n_neg_samples: int = 5, seed: int = 42):
        super().__init__(f"DecisionTree(f={n_factors})")
        self.n_factors = n_factors
        self.n_neg_samples = n_neg_samples
        self.seed = seed
        self.model = None
        self.best_params_ = None

    def fit(self, train_matrix: np.ndarray, tune: bool = True, param_dist: dict = None, 
            cv: int = 3, n_iter: int = 10):
        """
        Train the model. Optionally tune hyperparameters.

        Parameters:
        -----------
        train_matrix : np.ndarray
        tune : bool - Whether to perform hyperparameter tuning
        param_dist : dict - Parameter distributions for RandomizedSearchCV (uses default if None)
        cv : int - Number of cross-validation folds
        n_iter : int - Number of parameter combinations to try
        """
        self.train_matrix = train_matrix
        
        # Use shared utility function
        X_train, y_train, self.svd, self.item_factors = prepare_classifier_training_data(
            train_matrix, self.n_factors, self.n_neg_samples, self.seed
        )

        if tune:
            if param_dist is None:
                param_dist = {
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                }

            logger.info(f"Tuning {self.name} with RandomizedSearchCV (n_iter={n_iter})...")
            base_model = DecisionTreeClassifier(random_state=self.seed)
            random_search = RandomizedSearchCV(
                base_model, param_dist, n_iter=n_iter, cv=cv,
                scoring='roc_auc', n_jobs=-1, verbose=1, random_state=self.seed
            )
            random_search.fit(X_train, y_train)
            self.model = random_search.best_estimator_
            self.best_params_ = random_search.best_params_
            logger.info(f"Best params: {self.best_params_}")
            logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        else:
            self.model = DecisionTreeClassifier(random_state=self.seed)
            self.model.fit(X_train, y_train)

        return self

    def recommend(self, user_vector: np.ndarray, n: int = 10) -> list:
        known_items = set(np.where(user_vector > 0)[0])
        candidate_items = [i for i in range(self.train_matrix.shape[1]) if i not in known_items]

        if len(candidate_items) == 0:
            return []

        X_pred = np.array([
            create_classifier_features(
                user_vector, self.train_matrix, self.item_factors, self.svd, item_idx
            )
            for item_idx in candidate_items
        ])
        scores = self.model.predict_proba(X_pred)[:, 1]
        top_indices = np.argsort(scores)[::-1][:n]
        return [candidate_items[i] for i in top_indices]


class DecisionTreeClusteredRecommender(BaseRecommender):
    """
    Decision Tree-based recommender with KMeans clustering.
    """

    def __init__(self, n_factors: int = 20, n_neg_samples: int = 5, 
                 n_clusters: int = 5, seed: int = 42):
        super().__init__(f"DecisionTreeClustered(f={n_factors},k={n_clusters})")
        self.n_factors = n_factors
        self.n_neg_samples = n_neg_samples
        self.n_clusters = n_clusters
        self.seed = seed
        self.model = None
        self.best_params_ = None

    def fit(self, train_matrix: np.ndarray, tune: bool = True, param_dist: dict = None, 
            cv: int = 3, n_iter: int = 10):
        """
        Train the model with clustered features.

        Parameters:
        -----------
        train_matrix : np.ndarray
        tune : bool - Whether to perform hyperparameter tuning
        param_dist : dict - Parameter distributions for RandomizedSearchCV
        cv : int - Number of cross-validation folds
        n_iter : int - Number of parameter combinations to try
        """
        self.train_matrix = train_matrix
        
        # Use shared utility function with clustering enabled
        X_train, y_train, self.svd, self.item_factors = prepare_classifier_training_data(
            train_matrix, self.n_factors, self.n_neg_samples, self.seed,
            is_clustered=True, n_clusters=self.n_clusters
        )

        if tune:
            if param_dist is None:
                param_dist = {
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                }

            logger.info(f"Tuning {self.name} with RandomizedSearchCV (n_iter={n_iter})...")
            base_model = DecisionTreeClassifier(random_state=self.seed)
            random_search = RandomizedSearchCV(
                base_model, param_dist, n_iter=n_iter, cv=cv,
                scoring='roc_auc', n_jobs=-1, verbose=1, random_state=self.seed
            )
            random_search.fit(X_train, y_train)
            self.model = random_search.best_estimator_
            self.best_params_ = random_search.best_params_
            logger.info(f"Best params: {self.best_params_}")
            logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        else:
            self.model = DecisionTreeClassifier(random_state=self.seed)
            self.model.fit(X_train, y_train)

        return self

    def recommend(self, user_vector: np.ndarray, n: int = 10) -> list:
        known_items = set(np.where(user_vector > 0)[0])
        candidate_items = [i for i in range(self.train_matrix.shape[1]) if i not in known_items]

        if len(candidate_items) == 0:
            return []

        # Compute cluster label for this user using raw interaction vector
        from feature_engineering import get_kmeans
        kmeans = get_kmeans(self.train_matrix, n_clusters=self.n_clusters, seed=self.seed)
        # Cast to float64 to match KMeans expected dtype (const double)
        user_vector_cast = user_vector.astype(np.float64)
        cluster_label = kmeans.predict(user_vector_cast.reshape(1, -1))[0]

        X_pred = np.array([
            create_classifier_features_with_clustered(
                user_vector, self.train_matrix, self.item_factors, self.svd, item_idx,
                cluster_label=cluster_label, n_clusters=self.n_clusters
            )
            for item_idx in candidate_items
        ])
        scores = self.model.predict_proba(X_pred)[:, 1]
        top_indices = np.argsort(scores)[::-1][:n]
        return [candidate_items[i] for i in top_indices]


class XGBoostClusteredRecommender(BaseRecommender):
    """
    XGBoost-based recommender with KMeans clustering.
    """

    def __init__(self, n_factors: int = 20, n_neg_samples: int = 5, 
                 n_clusters: int = 5, seed: int = 42):
        super().__init__(f"XGBoostClustered(f={n_factors},k={n_clusters})")
        self.n_factors = n_factors
        self.n_neg_samples = n_neg_samples
        self.n_clusters = n_clusters
        self.seed = seed
        self.model = None
        self.best_params_ = None

    def fit(self, train_matrix: np.ndarray, tune: bool = True, param_dist: dict = None, 
            cv: int = 3, n_iter: int = 10):
        """
        Train the model with clustered features.

        Parameters:
        -----------
        train_matrix : np.ndarray
        tune : bool - Whether to perform hyperparameter tuning
        param_dist : dict - Parameter distributions for RandomizedSearchCV
        cv : int - Number of cross-validation folds
        n_iter : int - Number of parameter combinations to try
        """
        self.train_matrix = train_matrix
        
        # Use shared utility function with clustering enabled
        X_train, y_train, self.svd, self.item_factors = prepare_classifier_training_data(
            train_matrix, self.n_factors, self.n_neg_samples, self.seed,
            is_clustered=True, n_clusters=self.n_clusters
        )

        if tune:
            if param_dist is None:
                param_dist = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.1, 0.2],
                }

            logger.info(f"Tuning {self.name} with RandomizedSearchCV (n_iter={n_iter})...")
            base_model = XGBClassifier(random_state=self.seed, eval_metric='logloss')
            random_search = RandomizedSearchCV(
                base_model, param_dist, n_iter=n_iter, cv=cv,
                scoring='roc_auc', n_jobs=-1, verbose=1, random_state=self.seed
            )
            random_search.fit(X_train, y_train)
            self.model = random_search.best_estimator_
            self.best_params_ = random_search.best_params_
            logger.info(f"Best params: {self.best_params_}")
            logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        else:
            self.model = XGBClassifier(random_state=self.seed, eval_metric='logloss')
            self.model.fit(X_train, y_train)

        return self

    def recommend(self, user_vector: np.ndarray, n: int = 10) -> list:
        known_items = set(np.where(user_vector > 0)[0])
        candidate_items = [i for i in range(self.train_matrix.shape[1]) if i not in known_items]

        if len(candidate_items) == 0:
            return []

        # Compute cluster label for this user using raw interaction vector
        from feature_engineering import get_kmeans
        kmeans = get_kmeans(self.train_matrix, n_clusters=self.n_clusters, seed=self.seed)
        # Cast to float64 to match KMeans expected dtype (const double)
        user_vector_cast = user_vector.astype(np.float64)
        cluster_label = kmeans.predict(user_vector_cast.reshape(1, -1))[0]

        X_pred = np.array([
            create_classifier_features_with_clustered(
                user_vector, self.train_matrix, self.item_factors, self.svd, item_idx,
                cluster_label=cluster_label, n_clusters=self.n_clusters
            )
            for item_idx in candidate_items
        ])
        scores = self.model.predict_proba(X_pred)[:, 1]
        top_indices = np.argsort(scores)[::-1][:n]
        return [candidate_items[i] for i in top_indices]


class VotingEnsembleRecommender(BaseRecommender):
    """
    Voting Ensemble Recommender using weighted reciprocal rank voting.
    """

    def __init__(self, base_models: list[BaseRecommender], popularity_fallback: BaseRecommender,
                 cold_start_threshold: int = 3, weights: list = None):
        super().__init__(f"VotingEnsemble")
        
        self.base_models = base_models
        self.popularity_fallback = popularity_fallback
        self.cold_start_threshold = cold_start_threshold
        
        # Set weights (default to equal weights if not provided)
        if weights is None:
            self.weights = [1.0] * len(base_models)
        else:
            if len(weights) != len(base_models):
                raise ValueError("Number of weights must match number of base models")
            self.weights = weights
        
        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]

    def fit(self, train_matrix: np.ndarray):
        """
        Store reference to training matrix.
        
        Note: This model expects pre-trained base models and popularity fallback.
        The base models should already be fitted before being passed to this ensemble.
        """
        self.train_matrix = train_matrix
        logger.info(f"VotingEnsembleRecommender initialized with {len(self.base_models)} base models")
        logger.info(f"Cold-start threshold: {self.cold_start_threshold}")
        logger.info(f"Base models: {[m.name for m in self.base_models]}")
        logger.info(f"Model weights: {self.weights}")
        return self

    def _is_cold_start_user(self, user_vector: np.ndarray) -> bool:
        """Check if user has fewer interactions than cold-start threshold."""
        n_interactions = np.sum(user_vector > 0)
        return n_interactions < self.cold_start_threshold

    def _reciprocal_rank_fusion(self, rankings: list, n: int) -> list:
        """
        Combine multiple rankings using weighted reciprocal rank fusion.
        """
        # k parameter for reciprocal rank (standard is 60)
        k = 60
        
        # Aggregate scores using weighted reciprocal rank fusion
        item_scores = {}
        
        for model_idx, ranking in enumerate(rankings):
            weight = self.weights[model_idx]
            for rank, item_idx in enumerate(ranking):
                if item_idx not in item_scores:
                    item_scores[item_idx] = 0.0
                # Reciprocal rank formula: weight * 1 / (k + rank)
                item_scores[item_idx] += weight * (1.0 / (k + rank + 1))
        
        # Sort items by aggregated score (descending)
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [item_idx for item_idx, _ in sorted_items[:n]]

    def recommend(self, user_vector: np.ndarray, n: int = 10) -> list:
        """
        Generate top-n recommendations.
        """
        # Check for cold-start user
        if self._is_cold_start_user(user_vector):
            logger.info(f"Cold-start user detected, using popularity fallback")
            return self.popularity_fallback.recommend(user_vector, n)
        
        # Get recommendations from each base model
        # Request more than n to allow for diversity in fusion
        n_candidates = min(n * 3, self.train_matrix.shape[1])
        
        rankings = []
        for model in self.base_models:
            try:
                model_recs = model.recommend(user_vector, n_candidates)
                rankings.append(model_recs)
            except Exception as e:
                logger.warning(f"Model {model.name} failed to generate recommendations: {e}")
                continue
        
        # If all models failed, fall back to popularity
        if len(rankings) == 0:
            logger.warning("All base models failed, using popularity fallback")
            return self.popularity_fallback.recommend(user_vector, n)
        
        # Combine rankings using reciprocal rank fusion
        return self._reciprocal_rank_fusion(rankings, n)

