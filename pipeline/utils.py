from models import *

def load_all_models(models_dir: str = "pipeline/output/models") -> list:
    """
    Load all saved models from the output directory.
    """
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
    logger.info("Loading/Training models...")
    trained_models = []
    for model in models:
        if model.exists():
            # Load existing model
            loaded_model = type(model).load(model.get_model_path())
            trained_models.append(loaded_model)
        else:
            # Train and save new model
            logger.info(f"Training {model.name}...")
            model.fit(train_matrix)
            model.save()
            trained_models.append(model)
    models = trained_models
    logger.info("Done!")
    return models