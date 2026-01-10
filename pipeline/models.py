"""
Recommender system model classes.
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from feature_engineering import create_classifier_features, prepare_classifier_training_data


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
    def recommend(self, user_idx: int, n: int = 10) -> list:
        """Generate top-n recommendations for a user (excluding known items)."""
        pass


class PopularityRecommender(BaseRecommender):
    """Recommends most popular items to all users."""

    def __init__(self):
        super().__init__("Popularity")

    def fit(self, train_matrix: np.ndarray):
        self.train_matrix = train_matrix
        self.popular_items = np.argsort(train_matrix.sum(axis=0))[::-1]
        return self

    def recommend(self, user_idx: int, n: int = 10) -> list:
        known = set(np.where(self.train_matrix[user_idx] > 0)[0])
        return [i for i in self.popular_items if i not in known][:n]


class RandomRecommender(BaseRecommender):
    """Random baseline recommender."""

    def __init__(self, seed: int = 42):
        super().__init__("Random")
        self.rng = np.random.RandomState(seed)

    def fit(self, train_matrix: np.ndarray):
        self.train_matrix = train_matrix
        self.n_items = train_matrix.shape[1]
        return self

    def recommend(self, user_idx: int, n: int = 10) -> list:
        known = set(np.where(self.train_matrix[user_idx] > 0)[0])
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

    def recommend(self, user_idx: int, n: int = 10) -> list:
        sim = self.user_sim[user_idx]
        top_users = np.argsort(sim)[::-1][:self.k]

        scores = np.zeros(self.train_matrix.shape[1])
        for u in top_users:
            scores += sim[u] * self.train_matrix[u]

        known = np.where(self.train_matrix[user_idx] > 0)[0]
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

    def recommend(self, user_idx: int, n: int = 10) -> list:
        user_items = np.where(self.train_matrix[user_idx] > 0)[0]

        if len(user_items) == 0:
            return np.argsort(self.train_matrix.sum(axis=0))[::-1][:n].tolist()

        scores = self.item_sim[user_items].sum(axis=0)
        scores[user_items] = -np.inf
        return np.argsort(scores)[::-1][:n].tolist()


class SVDRecommender(BaseRecommender):
    """Matrix Factorization using SVD."""

    def __init__(self, n_factors: int = 50):
        super().__init__(f"SVD(f={n_factors})")
        self.n_factors = n_factors

    def fit(self, train_matrix: np.ndarray):
        self.train_matrix = train_matrix
        svd = TruncatedSVD(n_components=min(self.n_factors, min(train_matrix.shape) - 1))
        user_factors = svd.fit_transform(train_matrix)
        self.predictions = user_factors @ svd.components_
        return self

    def recommend(self, user_idx: int, n: int = 10) -> list:
        scores = self.predictions[user_idx].copy()
        known = np.where(self.train_matrix[user_idx] > 0)[0]
        scores[known] = -np.inf
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
        X_train, y_train, self.user_factors, self.item_factors = prepare_classifier_training_data(
            train_matrix, self.n_factors, self.n_neg_samples, self.seed
        )

        if tune:
            if param_dist is None:
                param_dist = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.1, 0.2],
                }

            print(f"Tuning {self.name} with RandomizedSearchCV (n_iter={n_iter})...")
            base_model = XGBClassifier(random_state=self.seed, eval_metric='logloss')
            random_search = RandomizedSearchCV(
                base_model, param_dist, n_iter=n_iter, cv=cv,
                scoring='roc_auc', n_jobs=-1, verbose=1, random_state=self.seed
            )
            random_search.fit(X_train, y_train)

            self.model = random_search.best_estimator_
            self.best_params_ = random_search.best_params_
            print(f"Best params: {self.best_params_}")
            print(f"Best CV score: {random_search.best_score_:.4f}")
        else:
            self.model = XGBClassifier(random_state=self.seed)
            self.model.fit(X_train, y_train)

        return self

    def recommend(self, user_idx: int, n: int = 10) -> list:
        known_items = set(np.where(self.train_matrix[user_idx] > 0)[0])
        candidate_items = [i for i in range(self.train_matrix.shape[1]) if i not in known_items]

        if len(candidate_items) == 0:
            return []

        X_pred = np.array([
            create_classifier_features(self.train_matrix, self.user_factors, self.item_factors, user_idx, item_idx)
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
        X_train, y_train, self.user_factors, self.item_factors = prepare_classifier_training_data(
            train_matrix, self.n_factors, self.n_neg_samples, self.seed
        )

        if tune:
            if param_dist is None:
                param_dist = {
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                }

            print(f"Tuning {self.name} with RandomizedSearchCV (n_iter={n_iter})...")
            base_model = DecisionTreeClassifier(random_state=self.seed)
            random_search = RandomizedSearchCV(
                base_model, param_dist, n_iter=n_iter, cv=cv,
                scoring='roc_auc', n_jobs=-1, verbose=1, random_state=self.seed
            )
            random_search.fit(X_train, y_train)
            self.model = random_search.best_estimator_
            self.best_params_ = random_search.best_params_
            print(f"Best params: {self.best_params_}")
            print(f"Best CV score: {random_search.best_score_:.4f}")
        else:
            self.model = DecisionTreeClassifier(random_state=self.seed)
            self.model.fit(X_train, y_train)

        return self

    def recommend(self, user_idx: int, n: int = 10) -> list:
        known_items = set(np.where(self.train_matrix[user_idx] > 0)[0])
        candidate_items = [i for i in range(self.train_matrix.shape[1]) if i not in known_items]

        if len(candidate_items) == 0:
            return []

        X_pred = np.array([
            create_classifier_features(self.train_matrix, self.user_factors, self.item_factors, user_idx, item_idx)
            for item_idx in candidate_items
        ])
        scores = self.model.predict_proba(X_pred)[:, 1]
        top_indices = np.argsort(scores)[::-1][:n]
        return [candidate_items[i] for i in top_indices]
