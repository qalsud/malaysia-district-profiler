import numpy as np
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
