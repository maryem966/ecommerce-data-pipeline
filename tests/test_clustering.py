"""
Tests unitaires — Clustering KMeans
=====================================
Teste le pipeline de segmentation non-supervisée.

Exécution : pytest tests/test_clustering.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clustering.kmeans_segmentation import (
    find_optimal_k,
    CLUSTERING_FEATURES,
    run_clustering,
)
from clustering.cluster_profiler import assign_cluster_personas


# ─────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────
@pytest.fixture
def clusterable_df():
    """DataFrame avec les features de clustering disponibles."""
    np.random.seed(42)
    n = 300
    # Créer 3 groupes distincts pour que KMeans les détecte
    group1 = pd.DataFrame({
        "Frequency":            np.random.uniform(1, 5, n // 3),
        "MonetaryTotal":        np.random.uniform(100, 500, n // 3),
        "CustomerTenureDays":   np.random.uniform(10, 100, n // 3),
        "SatisfactionScore":    np.random.uniform(1, 2.5, n // 3),
        "SupportTicketsCount":  np.random.uniform(5, 10, n // 3),
        "ReturnRatio":          np.random.uniform(0.3, 0.5, n // 3),
        "AvgBasketValue":       np.random.uniform(20, 100, n // 3),
        "EngagementScore":      np.random.uniform(0, 2, n // 3),
        "TenureFrequencyRatio": np.random.uniform(0, 0.05, n // 3),
        "Churn":                np.ones(n // 3, dtype=int),
    })
    group2 = pd.DataFrame({
        "Frequency":            np.random.uniform(20, 40, n // 3),
        "MonetaryTotal":        np.random.uniform(3000, 6000, n // 3),
        "CustomerTenureDays":   np.random.uniform(500, 1000, n // 3),
        "SatisfactionScore":    np.random.uniform(4, 5, n // 3),
        "SupportTicketsCount":  np.random.uniform(0, 2, n // 3),
        "ReturnRatio":          np.random.uniform(0, 0.05, n // 3),
        "AvgBasketValue":       np.random.uniform(150, 300, n // 3),
        "EngagementScore":      np.random.uniform(10, 20, n // 3),
        "TenureFrequencyRatio": np.random.uniform(0.05, 0.1, n // 3),
        "Churn":                np.zeros(n // 3, dtype=int),
    })
    group3 = pd.DataFrame({
        "Frequency":            np.random.uniform(8, 15, n // 3),
        "MonetaryTotal":        np.random.uniform(1000, 2000, n // 3),
        "CustomerTenureDays":   np.random.uniform(200, 400, n // 3),
        "SatisfactionScore":    np.random.uniform(3, 4, n // 3),
        "SupportTicketsCount":  np.random.uniform(2, 5, n // 3),
        "ReturnRatio":          np.random.uniform(0.1, 0.2, n // 3),
        "AvgBasketValue":       np.random.uniform(80, 150, n // 3),
        "EngagementScore":      np.random.uniform(5, 10, n // 3),
        "TenureFrequencyRatio": np.random.uniform(0.02, 0.05, n // 3),
        "Churn":                np.random.randint(0, 2, n // 3),
    })
    return pd.concat([group1, group2, group3], ignore_index=True)


@pytest.fixture
def scaled_data(clusterable_df):
    """Données normalisées pour le clustering."""
    from sklearn.preprocessing import StandardScaler
    available = [f for f in CLUSTERING_FEATURES if f in clusterable_df.columns]
    X = clusterable_df[available].fillna(clusterable_df[available].median())
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# ─────────────────────────────────────────────
# Tests : find_optimal_k
# ─────────────────────────────────────────────
class TestFindOptimalK:

    def test_returns_k_in_range(self, scaled_data):
        """Le K optimal doit être dans la plage testée."""
        k_range = range(2, 6)
        k_opt, metrics = find_optimal_k(scaled_data, k_range)
        assert k_opt in k_range

    def test_metrics_keys_present(self, scaled_data):
        """Les métriques doivent contenir les clés attendues."""
        k_range = range(2, 5)
        _, metrics = find_optimal_k(scaled_data, k_range)
        for key in ["k_range", "inertias", "silhouettes", "k_optimal", "best_silhouette"]:
            assert key in metrics

    def test_inertias_decreasing(self, scaled_data):
        """L'inertie doit être décroissante (plus K augmente, moins d'inertie)."""
        k_range = range(2, 7)
        _, metrics = find_optimal_k(scaled_data, k_range)
        inertias = metrics["inertias"]
        # Inertie globalement décroissante
        assert inertias[0] >= inertias[-1]

    def test_silhouette_values_in_range(self, scaled_data):
        """Les silhouette scores doivent être entre -1 et 1."""
        k_range = range(2, 5)
        _, metrics = find_optimal_k(scaled_data, k_range)
        for s in metrics["silhouettes"]:
            assert -1 <= s <= 1

    def test_metrics_lengths_match(self, scaled_data):
        """Les longueurs de inertias et silhouettes doivent correspondre à k_range."""
        k_range = range(2, 6)
        _, metrics = find_optimal_k(scaled_data, k_range)
        assert len(metrics["inertias"]) == len(k_range)
        assert len(metrics["silhouettes"]) == len(k_range)


# ─────────────────────────────────────────────
# Tests : run_clustering
# ─────────────────────────────────────────────
class TestRunClustering:

    def test_cluster_labels_assigned_to_all_rows(self, clusterable_df, tmp_path):
        """Chaque ligne doit recevoir un label de cluster."""
        report = run_clustering(
            clusterable_df,
            k_range=range(2, 5),
            output_dir=str(tmp_path),
            model_dir=str(tmp_path),
            db_path=str(tmp_path / "test.db"),
        )
        total = sum(report["cluster_distribution"].values())
        assert total == len(clusterable_df)

    def test_k_optimal_in_range(self, clusterable_df, tmp_path):
        """Le K optimal sélectionné doit être dans la plage testée."""
        k_range = range(2, 5)
        report = run_clustering(
            clusterable_df,
            k_range=k_range,
            output_dir=str(tmp_path),
            model_dir=str(tmp_path),
            db_path=str(tmp_path / "test.db"),
        )
        assert report["k_optimal"] in list(k_range)

    def test_silhouette_positive(self, clusterable_df, tmp_path):
        """Le silhouette score final doit être positif (clusters bien séparés)."""
        report = run_clustering(
            clusterable_df,
            k_range=range(2, 5),
            output_dir=str(tmp_path),
            model_dir=str(tmp_path),
            db_path=str(tmp_path / "test.db"),
        )
        # Avec 3 groupes distincts, le silhouette doit être > 0
        assert report["final_silhouette"] > 0

    def test_model_saved(self, clusterable_df, tmp_path):
        """Le modèle KMeans doit être sauvegardé sur le disque."""
        run_clustering(
            clusterable_df,
            k_range=range(2, 4),
            output_dir=str(tmp_path),
            model_dir=str(tmp_path),
            db_path=str(tmp_path / "test.db"),
        )
        assert (tmp_path / "kmeans_model.pkl").exists()
        assert (tmp_path / "kmeans_scaler.pkl").exists()

    def test_plots_generated(self, clusterable_df, tmp_path):
        """Les graphiques doivent être générés."""
        report = run_clustering(
            clusterable_df,
            k_range=range(2, 4),
            output_dir=str(tmp_path),
            model_dir=str(tmp_path),
            db_path=str(tmp_path / "test.db"),
        )
        for plot_key, plot_path in report["plot_paths"].items():
            assert os.path.exists(plot_path), f"Graphique manquant: {plot_key} → {plot_path}"

    def test_insufficient_features_raises(self):
        """Une erreur doit être levée si moins de 3 features sont disponibles."""
        df = pd.DataFrame({"Frequency": [1, 2, 3], "Churn": [0, 1, 0]})
        with pytest.raises(ValueError, match="Pas assez de features"):
            run_clustering(df)


# ─────────────────────────────────────────────
# Tests : assign_cluster_personas
# ─────────────────────────────────────────────
class TestAssignClusterPersonas:

    def test_persona_assigned_to_each_cluster(self, clusterable_df, tmp_path):
        """Chaque cluster doit recevoir un persona."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        available = [f for f in CLUSTERING_FEATURES if f in clusterable_df.columns]
        X = clusterable_df[available].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        clusterable_df = clusterable_df.copy()
        clusterable_df["Cluster"] = labels
        profiles = clusterable_df.groupby("Cluster")[available].mean()

        personas = assign_cluster_personas(profiles)
        assert len(personas) == 3
        for cid, info in personas.items():
            assert "persona" in info
            assert "description" in info
            assert "action" in info
            assert "stats" in info

    def test_persona_is_string(self, clusterable_df, tmp_path):
        """Le persona doit être une chaîne non vide."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        available = [f for f in CLUSTERING_FEATURES if f in clusterable_df.columns]
        X = clusterable_df[available].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_scaled)
        clusterable_df = clusterable_df.copy()
        clusterable_df["Cluster"] = labels
        profiles = clusterable_df.groupby("Cluster")[available].mean()
        personas = assign_cluster_personas(profiles)
        for cid, info in personas.items():
            assert isinstance(info["persona"], str) and len(info["persona"]) > 0
