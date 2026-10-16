"""
Clustering — KMeans Customer Segmentation
==========================================
Segmentation non-supervisée des clients en clusters comportementaux.

Algorithme :
  1. Sélection des features RFM + comportementales (sans la cible Churn)
  2. Normalisation StandardScaler
  3. Sélection automatique du nombre optimal de clusters K
     via méthode Elbow (inertie) + Silhouette Score
  4. Entraînement du KMeans final
  5. Visualisation PCA 2D des clusters
  6. Sauvegarde du modèle + des labels dans SQLite

Pourquoi le clustering est utile ici ?
  - Permet de segmenter les clients SANS utiliser la cible Churn
  - Les segments peuvent enrichir le modèle supervisé (feature "Cluster")
  - Donne une vision business (personas clients) actionnable
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif pour la génération de fichiers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sqlite3
import os
import joblib
import logging
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Features utilisées pour le clustering
# (comportementales, pas de cible)
# ─────────────────────────────────────────────
CLUSTERING_FEATURES = [
    "Frequency",
    "MonetaryTotal",
    "CustomerTenureDays",
    "SatisfactionScore",
    "SupportTicketsCount",
    "ReturnRatio",
    "AvgBasketValue",
    "EngagementScore",
    "TenureFrequencyRatio",
]


# ─────────────────────────────────────────────
# Sélection du K optimal
# ─────────────────────────────────────────────
def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 9),
    random_state: int = 42,
) -> tuple[int, dict]:
    """
    Détermine le nombre optimal de clusters K via :
      - Méthode Elbow (minimisation de l'inertie)
      - Silhouette Score (maximisation de la cohérence intra-cluster)

    Retourne (k_optimal, métriques_dict).
    """
    inertias   = []
    silhouettes = []

    logger.info(f"[CLUSTERING] Recherche du K optimal dans {list(k_range)}...")

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        logger.info(f"  K={k} → Inertie={km.inertia_:.0f} | Silhouette={sil:.4f}")

    # K optimal = celui avec le meilleur silhouette score
    best_idx = int(np.argmax(silhouettes))
    k_optimal = list(k_range)[best_idx]

    metrics = {
        "k_range":       list(k_range),
        "inertias":      [round(i, 2) for i in inertias],
        "silhouettes":   [round(s, 4) for s in silhouettes],
        "k_optimal":     k_optimal,
        "best_silhouette": round(silhouettes[best_idx], 4),
    }

    logger.info(f"[CLUSTERING] K optimal sélectionné : K={k_optimal} (Silhouette={silhouettes[best_idx]:.4f})")
    return k_optimal, metrics


# ─────────────────────────────────────────────
# Visualisation Elbow + Silhouette
# ─────────────────────────────────────────────
def plot_elbow_silhouette(metrics: dict, output_dir: str = "reports") -> str:
    """
    Génère un graphique double :
      - Gauche : courbe Elbow (inertie vs K)
      - Droite : courbe Silhouette (score vs K)
    """
    os.makedirs(output_dir, exist_ok=True)
    k_range    = metrics["k_range"]
    inertias   = metrics["inertias"]
    silhouettes = metrics["silhouettes"]
    k_opt      = metrics["k_optimal"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sélection du K optimal — KMeans Clustering", fontsize=14, fontweight="bold")

    # Elbow
    axes[0].plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
    axes[0].axvline(x=k_opt, color="red", linestyle="--", label=f"K optimal = {k_opt}")
    axes[0].set_xlabel("Nombre de clusters (K)")
    axes[0].set_ylabel("Inertie (Within-Cluster Sum of Squares)")
    axes[0].set_title("Méthode Elbow")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Silhouette
    axes[1].plot(k_range, silhouettes, "gs-", linewidth=2, markersize=8)
    axes[1].axvline(x=k_opt, color="red", linestyle="--", label=f"K optimal = {k_opt}")
    axes[1].set_xlabel("Nombre de clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "clustering_elbow_silhouette.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[CLUSTERING] Graphique Elbow/Silhouette → {path}")
    return path


# ─────────────────────────────────────────────
# Visualisation PCA 2D des clusters
# ─────────────────────────────────────────────
def plot_clusters_pca(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    k: int,
    output_dir: str = "reports",
) -> str:
    """
    Réduit X_scaled à 2 dimensions via PCA et affiche les clusters colorés.
    Permet une visualisation intuitive de la séparation des groupes.
    """
    os.makedirs(output_dir, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    colors = cm.tab10(np.linspace(0, 0.9, k))

    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster_id in range(k):
        mask = labels == cluster_id
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[colors[cluster_id]], label=f"Cluster {cluster_id}",
            alpha=0.6, s=40, edgecolors="white", linewidths=0.3
        )

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    ax.set_title(f"Segmentation KMeans — K={k} clusters (vue PCA 2D)", fontsize=13, fontweight="bold")
    ax.legend(loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "clustering_pca_2d.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[CLUSTERING] Graphique PCA 2D → {path}")
    return path


# ─────────────────────────────────────────────
# Heatmap des profils de clusters
# ─────────────────────────────────────────────
def plot_cluster_heatmap(cluster_profiles: pd.DataFrame, output_dir: str = "reports") -> str:
    """
    Génère une heatmap des moyennes normalisées de chaque feature par cluster.
    Permet de comprendre visuellement le profil de chaque segment.
    """
    import seaborn as sns
    os.makedirs(output_dir, exist_ok=True)

    # Normalisation min-max pour la heatmap
    profiles_norm = cluster_profiles.copy()
    for col in profiles_norm.columns:
        col_range = profiles_norm[col].max() - profiles_norm[col].min()
        if col_range > 0:
            profiles_norm[col] = (profiles_norm[col] - profiles_norm[col].min()) / col_range

    fig, ax = plt.subplots(figsize=(12, max(4, len(cluster_profiles) * 0.8 + 2)))
    sns.heatmap(
        profiles_norm,
        annot=cluster_profiles.round(2),
        fmt=".2f",
        cmap="RdYlGn",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Valeur normalisée (0–1)"},
    )
    ax.set_title("Profils des clusters — Moyennes par feature", fontsize=13, fontweight="bold")
    ax.set_xlabel("Features comportementales")
    ax.set_ylabel("Cluster ID")

    plt.tight_layout()
    path = os.path.join(output_dir, "clustering_profiles_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[CLUSTERING] Heatmap des profils → {path}")
    return path


# ─────────────────────────────────────────────
# Sauvegarde dans SQLite
# ─────────────────────────────────────────────
def save_clusters_to_db(
    df_with_labels: pd.DataFrame,
    db_path: str = "data/warehouse/ecommerce.db",
) -> int:
    """
    Sauvegarde le DataFrame enrichi avec les labels de cluster dans SQLite.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df_with_labels.to_sql("customers_clustered", conn, if_exists="replace", index=False)
        count = pd.read_sql("SELECT COUNT(*) as n FROM customers_clustered", conn)["n"][0]
    logger.info(f"[CLUSTERING] {count} lignes sauvegardées dans 'customers_clustered'")
    return int(count)


# ─────────────────────────────────────────────
# Pipeline principal de clustering
# ─────────────────────────────────────────────
def run_clustering(
    df: pd.DataFrame,
    k_range: range = range(2, 9),
    output_dir: str = "reports",
    model_dir: str = "models",
    db_path: str = "data/warehouse/ecommerce.db",
    random_state: int = 42,
) -> dict:
    """
    Pipeline complet de segmentation KMeans.

    Étapes :
      1. Sélection et validation des features de clustering
      2. Imputation des valeurs manquantes (médiane)
      3. Normalisation StandardScaler
      4. Sélection automatique du K optimal (Elbow + Silhouette)
      5. Entraînement KMeans final
      6. Calcul des profils de clusters (moyennes par groupe)
      7. Visualisations (Elbow, PCA 2D, Heatmap)
      8. Sauvegarde du modèle + labels SQLite

    Args:
        df           : DataFrame transformé (sortie du Transform ETL)
        k_range      : plage de K à tester
        output_dir   : dossier de sortie des graphiques
        model_dir    : dossier de sauvegarde du modèle KMeans
        db_path      : chemin SQLite
        random_state : graine aléatoire

    Returns:
        Dictionnaire de métriques et chemins de sortie
    """
    start = datetime.now()
    logger.info(f"[CLUSTERING] Démarrage du pipeline de segmentation")

    # 1. Sélection des features disponibles
    available_features = [f for f in CLUSTERING_FEATURES if f in df.columns]
    if len(available_features) < 3:
        raise ValueError(
            f"[CLUSTERING] Pas assez de features disponibles ({available_features}). "
            f"Lancez d'abord le pipeline ETL."
        )
    logger.info(f"[CLUSTERING] Features utilisées ({len(available_features)}) : {available_features}")

    X = df[available_features].copy()

    # 2. Imputation des nulls par médiane
    X = X.fillna(X.median())

    # 3. Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Sélection du K optimal
    k_optimal, k_metrics = find_optimal_k(X_scaled, k_range, random_state)

    # 5. Entraînement KMeans final
    logger.info(f"[CLUSTERING] Entraînement KMeans avec K={k_optimal}...")
    kmeans = KMeans(n_clusters=k_optimal, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    final_silhouette = silhouette_score(X_scaled, labels)
    logger.info(f"[CLUSTERING] KMeans entraîné — Silhouette final: {final_silhouette:.4f}")

    # 6. Profils des clusters (moyennes par feature)
    df_result = df.copy()
    df_result["Cluster"] = labels
    cluster_profiles = df_result.groupby("Cluster")[available_features].mean().round(3)
    logger.info(f"[CLUSTERING] Profils calculés pour {k_optimal} clusters")

    # 7. Visualisations
    plot_paths = {
        "elbow_silhouette": plot_elbow_silhouette(k_metrics, output_dir),
        "pca_2d":           plot_clusters_pca(X_scaled, labels, k_optimal, output_dir),
        "heatmap":          plot_cluster_heatmap(cluster_profiles, output_dir),
    }

    # 8. Sauvegarde du modèle
    os.makedirs(model_dir, exist_ok=True)
    kmeans_path = os.path.join(model_dir, "kmeans_model.pkl")
    scaler_path = os.path.join(model_dir, "kmeans_scaler.pkl")
    joblib.dump(kmeans, kmeans_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(available_features, os.path.join(model_dir, "clustering_features.pkl"))
    logger.info(f"[CLUSTERING] Modèle sauvegardé → {kmeans_path}")

    # 9. Sauvegarde SQLite
    rows_saved = save_clusters_to_db(df_result, db_path)

    elapsed = (datetime.now() - start).total_seconds()

    # Distribution des clusters
    cluster_dist = df_result["Cluster"].value_counts().sort_index().to_dict()

    report = {
        "k_optimal":         k_optimal,
        "k_metrics":         k_metrics,
        "final_silhouette":  round(final_silhouette, 4),
        "features_used":     available_features,
        "cluster_distribution": {int(k): int(v) for k, v in cluster_dist.items()},
        "cluster_profiles":  cluster_profiles.to_dict(),
        "plot_paths":        plot_paths,
        "model_path":        kmeans_path,
        "rows_saved_to_db":  rows_saved,
        "clustering_time_s": round(elapsed, 4),
    }

    logger.info(
        f"[CLUSTERING] Terminé en {elapsed:.2f}s | "
        f"K={k_optimal} | Silhouette={final_silhouette:.4f} | "
        f"Distribution: {cluster_dist}"
    )
    return report
