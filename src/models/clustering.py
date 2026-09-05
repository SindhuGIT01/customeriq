"""Customer segmentation via K-Means clustering.

Selects behavioral/value features, picks K via the elbow method, fits
K-Means, profiles each cluster in business terms, visualizes the segments,
and exports the labeled dataset.
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import save_metrics_json
from src.features.engineering import engineer_features

DATA_PATH = "data/processed/customers_clean.csv"
OUTPUT_PATH = "data/processed/customers_segmented.csv"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "clustering_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "clustering_profile.json")
REPORTS_DIR = "reports"
RANDOM_STATE = 42

CLUSTER_FEATURES = [
    "monthly_spend", "tenure_months", "usage_per_month",
    "support_ticket_rate", "customer_lifetime_value",
]
K_CANDIDATES = range(1, 11)


def load_cluster_data(path: str = DATA_PATH):
    """Load cleaned data, engineer features, and return the full df, the
    scaled matrix used for clustering, and the fitted scaler (so a caller
    can transform a new customer's features the same way later)."""
    df = pd.read_csv(path)
    df = engineer_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])
    return df, X_scaled, scaler


def compute_inertias(X_scaled, k_values=K_CANDIDATES) -> list:
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return inertias


def choose_elbow_k(k_values, inertias) -> int:
    """Pick K as the point of maximum distance from the line joining the
    first and last points of the inertia curve (the standard geometric
    elbow-detection heuristic)."""
    k_values = np.array(list(k_values), dtype=float)
    inertias = np.array(inertias, dtype=float)

    # Normalize both axes to [0, 1] so distance isn't dominated by scale.
    k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    inertia_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min())

    p1 = np.array([k_norm[0], inertia_norm[0]])
    p2 = np.array([k_norm[-1], inertia_norm[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    distances = []
    for x, y in zip(k_norm, inertia_norm, strict=True):
        point_vec = np.array([x, y]) - p1
        proj_len = np.dot(point_vec, line_vec_norm)
        proj_point = p1 + proj_len * line_vec_norm
        distances.append(np.linalg.norm(np.array([x, y]) - proj_point))

    best_idx = int(np.argmax(distances))
    return int(k_values[best_idx])


def plot_elbow(k_values, inertias, chosen_k: int, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(k_values), inertias, marker="o", color="#4C72B0")
    ax.axvline(chosen_k, color="red", linestyle="--", label=f"chosen K = {chosen_k}")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow method for K-Means")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def label_cluster(row: pd.Series, overall_churn_rate: float) -> str:
    """Turn a cluster's z-scored feature averages (plus its raw churn rate)
    into a business label.

    Thresholds are deliberately loose (0.3, not the usual 0.5+) since with
    only a handful of clusters, z-scores are computed over very few points
    and a strict cutoff can miss a cluster that's clearly the most/least
    valuable relative to the others just because the gap isn't extreme.
    """
    if row["customer_lifetime_value"] > 0.3:
        if row["churn_rate"] <= overall_churn_rate:
            return "High-Value Loyal"
        return "High-Value At-Risk"
    if row["tenure_months"] < -0.5:
        return "New Customer"
    if row["support_ticket_rate"] > 0.3 and row["usage_per_month"] < 0:
        return "High-Risk"
    if row["customer_lifetime_value"] < -0.3:
        return "Low-Value"
    return "Steady Mid-Tier"


def profile_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cluster feature averages, churn rate, size, and a
    business label based on how each cluster compares to the population."""
    cluster_means = df.groupby("cluster")[CLUSTER_FEATURES].mean()
    z_scores = (cluster_means - cluster_means.mean()) / cluster_means.std()

    profile = cluster_means.copy()
    profile["churn_rate"] = df.groupby("cluster")["churn"].mean()
    profile["size"] = df.groupby("cluster").size()

    overall_churn_rate = df["churn"].mean()
    labeling_input = z_scores.copy()
    labeling_input["churn_rate"] = profile["churn_rate"]
    profile["segment_label"] = labeling_input.apply(
        lambda row: label_cluster(row, overall_churn_rate), axis=1
    )
    return profile


def plot_clusters_pca(X_scaled, cluster_labels, segment_labels, output_path: str):
    """2D PCA projection of the clustering features, colored by cluster."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_clusters = sorted(pd.unique(cluster_labels))
    cmap = plt.get_cmap("tab10")
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        label = f"Cluster {cluster_id}: {segment_labels[cluster_id]}"
        ax.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.5,
                   color=cmap(i % 10), label=label)

    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({explained[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} variance)")
    ax.set_title("Customer segments (PCA projection)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    df, X_scaled, scaler = load_cluster_data()

    k_values = list(K_CANDIDATES)
    inertias = compute_inertias(X_scaled, k_values)
    chosen_k = choose_elbow_k(k_values, inertias)
    print(f"Elbow method chose K = {chosen_k}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    elbow_plot_path = os.path.join(REPORTS_DIR, "clustering_elbow.png")
    plot_elbow(k_values, inertias, chosen_k, elbow_plot_path)
    print(f"Saved elbow plot to {elbow_plot_path}")

    kmeans = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    profile = profile_clusters(df)
    df["segment_label"] = df["cluster"].map(profile["segment_label"])

    print("\nCluster profile (feature averages in original units):")
    print(profile.to_string(float_format=lambda x: f"{x:,.2f}"))

    scatter_path = os.path.join(REPORTS_DIR, "clustering_pca_scatter.png")
    plot_clusters_pca(X_scaled, df["cluster"].to_numpy(), profile["segment_label"], scatter_path)
    print(f"\nSaved cluster scatter plot to {scatter_path}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved segmented dataset to {OUTPUT_PATH}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    # Bundle the fitted KMeans with the scaler and the cluster -> business
    # label mapping, so a caller (e.g. the API) can segment a brand-new
    # customer without needing to re-run the whole clustering pipeline.
    joblib.dump(
        {"kmeans": kmeans, "scaler": scaler, "segment_labels": profile["segment_label"].to_dict()},
        MODEL_PATH,
    )
    print(f"Saved clustering model to {MODEL_PATH}")

    save_metrics_json(
        {"chosen_k": chosen_k, "profile": profile.reset_index().to_dict(orient="records")},
        METRICS_PATH,
    )
    print(f"Saved cluster profile to {METRICS_PATH}")

    return profile


if __name__ == "__main__":
    main()
