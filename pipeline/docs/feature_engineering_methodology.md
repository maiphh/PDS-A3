# Feature Engineering Methodology for Classifier-Based Recommender Systems

## Abstract

This document presents a comprehensive methodology for feature engineering in the context of classifier-based recommender systems. The approach transforms the recommendation problem into a binary classification task by constructing rich feature representations that capture user-item interaction patterns through dimensionality reduction, latent factor analysis, and optional clustering techniques.

---

## 1. Introduction

Recommender systems are fundamental to modern information retrieval and personalization applications. While traditional collaborative filtering approaches rely directly on user-item interaction matrices, classifier-based recommenders reframe the problem as predicting whether a user will interact with a given item.

This methodology describes the feature engineering pipeline that enables such classification-based recommendations, with emphasis on:

- **Latent factor extraction** via Truncated Singular Value Decomposition (SVD)
- **User activity and item popularity metrics**
- **Latent interaction features** capturing user-item compatibility
- **User clustering** for enhanced personalization

---

## 2. Memory-Based vs. Model-Based Recommender Systems

A fundamental distinction in recommender system design lies in how user-item interactions are processed and utilized. This section examines the contrasting data requirements between memory-based collaborative filtering and model-based approaches.

### 2.1 Memory-Based Collaborative Filtering

Memory-based methods, also known as neighborhood-based approaches, operate **directly on the raw user-item interaction matrix** without requiring explicit feature engineering.

#### 2.1.1 User-Based Collaborative Filtering (UserCF)

| Aspect | Description |
|--------|-------------|
| **Input Data** | Raw binary interaction matrix $\mathbf{R} \in \{0,1\}^{m \times n}$ |
| **Core Operation** | Compute pairwise user similarity via cosine similarity |
| **Similarity Formula** | $\text{sim}(u, v) = \frac{\mathbf{r}_u \cdot \mathbf{r}_v}{\|\mathbf{r}_u\| \|\mathbf{r}_v\|}$ |
| **Prediction** | Weighted aggregation of k-nearest neighbors' preferences |
| **Feature Engineering** | **None required** — operates on raw vectors |

The prediction for user $u$ on item $i$ is computed as:

$$\hat{r}_{u,i} = \sum_{v \in N_k(u)} \text{sim}(u, v) \cdot r_{v,i}$$

where $N_k(u)$ represents the $k$ most similar users to $u$.

#### 2.1.2 Item-Based Collaborative Filtering (ItemCF)

| Aspect | Description |
|--------|-------------|
| **Input Data** | Raw binary interaction matrix $\mathbf{R} \in \{0,1\}^{m \times n}$ |
| **Core Operation** | Compute pairwise item similarity via cosine similarity |
| **Similarity Formula** | $\text{sim}(i, j) = \frac{\mathbf{r}_i^T \cdot \mathbf{r}_j^T}{\|\mathbf{r}_i^T\| \|\mathbf{r}_j^T\|}$ |
| **Prediction** | Sum of similarities between user's known items and candidates |
| **Feature Engineering** | **None required** — operates on raw vectors |

### 2.2 Model-Based Approaches

Model-based recommenders learn an explicit model from the data, requiring **engineered feature representations** that capture latent structure.

#### 2.2.1 Matrix Factorization (SVD)

| Aspect | Description |
|--------|-------------|
| **Input Data** | Raw interaction matrix for training |
| **Core Operation** | Truncated SVD decomposition |
| **Learned Representations** | User factors $\mathbf{U}$, Item factors $\mathbf{V}$ |
| **Feature Engineering** | **Implicit** — latent factors are learned, not hand-crafted |
| **Prediction** | Dot product in latent space: $\hat{r}_{u,i} = \mathbf{u} \cdot \mathbf{v}_i^T$ |

#### 2.2.2 Classifier-Based Recommenders (Decision Tree, XGBoost)

| Aspect | Description |
|--------|-------------|
| **Input Data** | Requires **explicit feature engineering** |
| **Core Operation** | Binary classification (interact / not interact) |
| **Feature Requirements** | Engineered user-item pair features |
| **Prediction** | $P(\text{interact} | \mathbf{x}_{u,i})$ via classifier probability |

**Required Engineered Features:**
- User activity scores
- Item popularity scores
- User latent representation (via SVD transform)
- Item latent representation
- Latent interaction features (element-wise product)
- Optional: Cluster membership (one-hot encoded)

### 2.3 Comparative Summary

| Characteristic | Memory-Based | Model-Based (Classifier) |
|---------------|--------------|-------------------------|
| **Data Input** | Raw interaction matrix | Engineered feature vectors |
| **Preprocessing** | Minimal (optional normalization) | SVD factorization, feature construction |
| **Scalability** | Limited (stores full matrix) | Better (compact model) |
| **Cold Start** | Severe (requires overlap) | Partial mitigation via features |
| **Interpretability** | High (similarity-based) | Moderate (depends on classifier) |
| **Flexibility** | Fixed similarity metric | Can incorporate diverse features |

### 2.4 Rationale for Feature Engineering in Model-Based Systems

Memory-based systems leverage the **intrinsic similarity structure** in raw interaction vectors. However, classifier-based recommenders require explicit features because:

1. **Classifiers need fixed-dimension inputs** — Raw user-item pairs don't naturally form a feature vector suitable for standard ML classifiers.

2. **Capturing interaction semantics** — The relationship between a user and item must be encoded into features that reflect compatibility (e.g., latent factor products).

3. **Enabling generalization** — Engineered features (activity, popularity, latent representations) allow the model to generalize patterns beyond direct co-occurrence statistics.

4. **Supporting augmentation** — Additional context (cluster membership, temporal features) can be incorporated into feature vectors, which is not possible with pure similarity-based methods.

---

## 3. Data Representation

### 3.1 User-Item Interaction Matrix

The input data is represented as a binary user-item interaction matrix $\mathbf{R} \in \{0, 1\}^{m \times n}$, where:
- $m$ = number of users
- $n$ = number of items
- $R_{u,i} = 1$ if user $u$ interacted with item $i$, otherwise $0$

### 3.2 User Interaction Vectors

Each user is represented by a row vector $\mathbf{r}_u \in \{0, 1\}^{n}$ from the interaction matrix, capturing their complete interaction history across all items.

---

## 4. Dimensionality Reduction via Truncated SVD

### 4.1 Motivation

High-dimensional sparse interaction vectors are computationally expensive and prone to noise. Truncated Singular Value Decomposition (SVD) is employed to extract dense, low-dimensional latent representations that capture the underlying structure of user-item interactions.

### 4.2 Implementation

The SVD decomposition is computed on the full training matrix:

$$\mathbf{R} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

Where:
- $\mathbf{U} \in \mathbb{R}^{m \times k}$ contains user latent factors
- $\mathbf{\Sigma} \in \mathbb{R}^{k \times k}$ is the diagonal matrix of singular values
- $\mathbf{V} \in \mathbb{R}^{n \times k}$ contains item latent factors
- $k$ = number of latent factors (hyperparameter)

The number of components is bounded by:

$$k = \min(n\_factors, \min(m, n) - 1)$$

### 4.3 Item Factor Extraction

Item latent factors are extracted as the transposed components of the fitted SVD model:

$$\mathbf{F}_{item} = \mathbf{V}^T \in \mathbb{R}^{n \times k}$$

Each item $i$ is then represented by a $k$-dimensional vector $\mathbf{f}_i$.

---

## 5. Feature Construction

### 5.1 Base Feature Set

For each user-item pair $(u, i)$, the following features are constructed:

#### 5.1.1 User Activity Score
Measures the overall engagement level of a user:

$$\text{user\_activity} = \sum_{j=1}^{n} r_{u,j}$$

This scalar captures how active a user is across all items.

#### 5.1.2 Item Popularity Score
Measures how frequently an item is interacted with across all users:

$$\text{item\_popularity} = \sum_{v=1}^{m} R_{v,i}$$

This feature helps distinguish popular items from niche ones.

#### 5.1.3 User Latent Representation
The user's interaction vector is projected into the latent space:

$$\mathbf{z}_u = \text{SVD.transform}(\mathbf{r}_u) \in \mathbb{R}^{k}$$

This transformation maps the sparse binary vector to a dense representation capturing user preferences.

#### 5.1.4 Item Latent Representation
The pre-computed item latent factor:

$$\mathbf{f}_i \in \mathbb{R}^{k}$$

#### 5.1.5 Latent Interaction Features
Element-wise product of user and item latent vectors:

$$\mathbf{z}_{int} = \mathbf{z}_u \odot \mathbf{f}_i \in \mathbb{R}^{k}$$

This captures the compatibility between user preferences and item characteristics in latent space.

### 5.2 Feature Vector Assembly

The final feature vector $\mathbf{x}_{u,i}$ is constructed by concatenation:

$$\mathbf{x}_{u,i} = [\text{user\_activity}, \text{item\_popularity}, \mathbf{z}_u, \mathbf{f}_i, \mathbf{z}_{int}]$$

**Total feature dimensionality:** $2 + 3k$ features

---

## 6. User Clustering Extension

### 6.1 Motivation

Users with similar behavioral patterns may respond similarly to recommendations. K-Means clustering on raw interaction vectors segments users into behavioral groups, providing additional contextual information for the classifier.

### 6.2 K-Means Clustering

Users are clustered in the original interaction space:

$$\mathbf{C} = \text{KMeans}(\mathbf{R}, k=n\_clusters)$$

Each user is assigned a cluster label $c_u \in \{0, 1, ..., n\_clusters-1\}$.

### 6.3 Model Persistence

The K-Means model is persisted to disk for consistency between training and inference:

1. **Check cache** – Return cached model if available in memory
2. **Load from disk** – Attempt to load previously saved model
3. **Train new model** – Fit on training data and save to disk

### 6.4 Cluster-Augmented Features

The cluster label is one-hot encoded and appended to the base feature vector:

$$\mathbf{c}_{one\_hot} = [0, ..., 1, ..., 0] \in \{0, 1\}^{n\_clusters}$$

**Extended feature vector:**

$$\mathbf{x}_{u,i}^{clustered} = [\mathbf{x}_{u,i}, \mathbf{c}_{one\_hot}]$$

**Total feature dimensionality:** $2 + 3k + n\_clusters$ features

---

## 7. Training Sample Generation

### 7.1 Positive and Negative Sampling Strategy

The training data is constructed with explicit positive and negative examples:

#### Positive Samples
For each user $u$, all items where $R_{u,i} = 1$ constitute positive samples with label $y = 1$.

#### Negative Sampling
For each user, negative samples are drawn from items where $R_{u,i} = 0$:

$$n\_neg = \min(|\text{neg\_items}|, |\text{pos\_items}| \times n\_neg\_samples)$$

This controlled sampling prevents class imbalance and ensures representative negative examples.

### 7.2 Efficiency Considerations

For clustered features, cluster labels are **pre-computed** for all users before sample generation, avoiding redundant K-Means predictions during the feature creation loop.

---

## 8. Feature Summary

| Feature Category | Dimensionality | Description |
|-----------------|----------------|-------------|
| User Activity | 1 | Total interactions for the user |
| Item Popularity | 1 | Total interactions for the item |
| User Latent | $k$ | SVD-projected user representation |
| Item Latent | $k$ | Pre-computed item latent factors |
| Latent Interaction | $k$ | Element-wise user-item compatibility |
| Cluster (optional) | $n\_clusters$ | One-hot encoded user cluster |

**Standard features:** $2 + 3k$  
**Clustered features:** $2 + 3k + n\_clusters$

---

## 9. References

1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30–37.

2. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. *Proceedings of the 26th International Conference on World Wide Web*, 173–182.

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

4. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1(14), 281–297.

---

*Document generated based on implementation in `feature_engineering.py`*
