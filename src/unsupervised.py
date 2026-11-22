import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def run_kmeans_pipeline(df, features, k):
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(X_scaled)
    var_explained = sum(pca.explained_variance_ratio_) * 100

    df = df.copy()
    df['cluster'] = labels
    df['PC1'] = pca_transformed[:, 0]
    df['PC2'] = pca_transformed[:, 1]

    return df, X_scaled, kmeans, pca, var_explained


def build_cluster_name_map(kmeans, features):
    """Name K=3 clusters from their centroids.

    K-Means cluster indices are arbitrary, so fixed index-to-name mappings
    become wrong when the data or survey year changes. For K=3, the cluster
    with the strongest income/expenditure profile is labelled high-income,
    the poorest remaining cluster is labelled rural/high-poverty, and the
    third is labelled middle-income.
    """
    n_clusters = len(kmeans.cluster_centers_)
    generic = {i: f"Cluster {i}" for i in range(n_clusters)}

    if n_clusters != 3:
        return generic

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    income_cols = [
        c for c in ("income_median", "income_mean", "expenditure_mean")
        if c in centers.columns
    ]
    if not income_cols or "poverty" not in centers.columns:
        return generic

    income_score = centers[income_cols].mean(axis=1)
    poverty_score = centers["poverty"]

    ordered = income_score.sort_values(ascending=False).index.tolist()
    high_income = int(ordered[0])
    remaining = ordered[1:]
    high_poverty = int(poverty_score.loc[remaining].idxmax())
    middle = int(next(i for i in remaining if i != high_poverty))

    return {
        high_income: "High-Income Metro Hubs",
        middle: "Middle-Income Towns & Suburbs",
        high_poverty: "Rural & High-Poverty Districts",
    }


def compute_optimal_k_metrics(X_scaled, k_min=2, k_max=10):
    k_range = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    return k_range, inertias, silhouettes


def compute_cluster_distances(X_scaled, kmeans):
    distances = []
    for i in range(len(X_scaled)):
        centroid = kmeans.cluster_centers_[kmeans.labels_[i]]
        distances.append(np.linalg.norm(X_scaled[i] - centroid))
    return distances
