"""
Exploratory Data Analysis pipeline for Anonymous Microsoft Web Data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import convert_to_wide_dataframe
from logger import setup_logger

logger = setup_logger()

EDA_OUTPUT_PATH = "pipeline/output/eda"


def ensure_output_dir():
    os.makedirs(EDA_OUTPUT_PATH, exist_ok=True)


def print_dataset_summary(df: pd.DataFrame, attr_df: pd.DataFrame):
    """Print basic dataset statistics."""
    logger.info(f"Interactions DataFrame shape: {df.shape}")
    logger.info(f"Attributes DataFrame shape: {attr_df.shape}")
    logger.info(f"Number of unique users: {df['case_id'].nunique()}")
    logger.info(f"Number of unique pages: {attr_df['attr_id'].nunique()}")


def calculate_sparsity(df: pd.DataFrame) -> float:
    """Calculate and return sparsity of user-item interaction matrix."""
    wide_df = convert_to_wide_dataframe(df, user_col='case_id', item_col='attr_id')
    total_elements = wide_df.shape[0] * wide_df.shape[1]
    non_zero_elements = (wide_df > 0).sum().sum()
    sparsity = (1 - (non_zero_elements / total_elements)) * 100
    logger.info(f"The interaction matrix is {sparsity:.2f}% sparse.")
    return sparsity


def plot_user_activity_distribution(df: pd.DataFrame):
    """Hypothesis 1: User activity is uneven."""
    user_activity = df.groupby("case_id").size()
    
    plt.figure(figsize=(8, 4))
    user_activity.hist(bins=50)
    plt.xlabel("Number of page visits per user")
    plt.ylabel("Number of users")
    plt.title("Distribution of User Activity Levels")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_PATH, "user_activity_distribution.png"), dpi=150)
    plt.close()
    logger.info("Saved user_activity_distribution.png")


def analyze_user_activity_page_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """Hypothesis 2: Highly active users tend to visit more popular pages."""
    page_popularity = df.groupby("attr_id")["case_id"].nunique()
    user_activity = df.groupby("case_id").size()
    
    df_analysis = df.copy()
    df_analysis["user_activity"] = df_analysis["case_id"].map(user_activity)
    df_analysis["page_popularity"] = df_analysis["attr_id"].map(page_popularity)
    
    correlation = df_analysis[["user_activity", "page_popularity"]].corr()
    logger.info(f"User activity vs page popularity correlation:\n{correlation}")
    return correlation


def plot_user_diversity(df: pd.DataFrame):
    """Hypothesis 3: Users tend to visit new pages rather than revisiting."""
    user_activity = df.groupby("case_id").size()
    user_diversity = df.groupby("case_id")["attr_id"].nunique()
    
    eda_user_df = pd.DataFrame({
        "activity": user_activity,
        "diversity": user_diversity
    })
    
    plt.figure(figsize=(7, 5))
    plt.scatter(eda_user_df["activity"], eda_user_df["diversity"], alpha=0.4)
    plt.xlabel("User Activity (Total Page Visits)")
    plt.ylabel("Browsing Diversity (Unique Pages Visited)")
    plt.title("User Activity vs Browsing Diversity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_PATH, "user_activity_vs_diversity.png"), dpi=150)
    plt.close()
    logger.info("Saved user_activity_vs_diversity.png")


def plot_page_popularity_distribution(df: pd.DataFrame):
    """Hypothesis 4: Page popularity follows a long tail distribution."""
    page_popularity = df.groupby("attr_id")["case_id"].nunique()
    page_popularity_df = page_popularity.reset_index()
    page_popularity_df.columns = ["attr_id", "num_users_visited"]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(page_popularity_df['num_users_visited'], bins=50, kde=True)
    plt.title('Distribution of Number of Users Visiting Each Page')
    plt.xlabel('Number of Users Visited')
    plt.ylabel('Number of Pages')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_PATH, "page_popularity_distribution.png"), dpi=150)
    plt.close()
    logger.info("Saved page_popularity_distribution.png")


def eda(df: pd.DataFrame, attr_df: pd.DataFrame):
    """Main EDA pipeline."""
    logger.info("Starting EDA pipeline...")
    ensure_output_dir()
    
    # Dataset summary
    print_dataset_summary(df, attr_df)
    
    # Sparsity analysis
    calculate_sparsity(df)
    
    # Hypothesis 1: User activity distribution
    plot_user_activity_distribution(df)
    
    # Hypothesis 2: User activity vs page popularity correlation
    analyze_user_activity_page_popularity(df)
    
    # Hypothesis 3: User activity vs browsing diversity
    plot_user_diversity(df)
    
    # Hypothesis 4: Page popularity distribution (long tail)
    plot_page_popularity_distribution(df)
    
    logger.info("EDA pipeline completed. Plots saved to: " + EDA_OUTPUT_PATH)

if __name__ == "__main__":
    eda()