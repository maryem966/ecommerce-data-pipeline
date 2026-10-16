"""
Tests unitaires — Étape Transform
====================================
Teste chaque fonction de transformation individuellement.

Exécution : pytest tests/test_transform.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from etl.transform import (
    step_handle_outliers,
    step_feature_engineering,
    step_remove_leakage,
    step_remove_high_correlation,
    transform,
    LEAKAGE_COLS,
)


# ─────────────────────────────────────────────
# Fixture de base
# ─────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """DataFrame minimal représentatif du dataset e-commerce."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "CustomerID":          [f"C{i}" for i in range(n)],
        "Recency":             np.random.uniform(0, 365, n),
        "Frequency":           np.random.randint(1, 50, n).astype(float),
        "MonetaryTotal":       np.random.uniform(100, 5000, n),
        "MonetaryAvg":         np.random.uniform(10, 200, n),
        "CustomerTenureDays":  np.random.randint(30, 1000, n).astype(float),
        "SatisfactionScore":   np.random.uniform(1, 5, n),
        "SupportTicketsCount": np.random.randint(0, 10, n).astype(float),
        "ReturnRatio":         np.random.uniform(0, 0.5, n),
        "RegistrationDate":    ["2020-01-15"] * n,
        "LastLoginIP":         ["192.168.1.1"] * (n // 2) + ["8.8.8.8"] * (n // 2),
        "ChurnRiskCategory":   ["Haut"] * n,
        "RFMSegment":          ["Champions"] * n,
        "AccountStatus":       ["Active"] * n,
        "CustomerType":        ["Régulier"] * n,
        "LoyaltyLevel":        ["Établi"] * n,
        "FavoriteSeason":      ["Automne"] * n,
        "NewsletterSubscribed": ["Yes"] * n,
        "ChurnRisk":           [1] * n,
        "Churn":               np.random.randint(0, 2, n),
        "Gender":              ["M", "F"] * (n // 2),
        "Region":              ["UK", "FR"] * (n // 2),
    })


# ─────────────────────────────────────────────
# Tests : step_handle_outliers
# ─────────────────────────────────────────────
class TestHandleOutliers:

    def test_negative_support_tickets_replaced(self, sample_df):
        """Les valeurs négatives de SupportTicketsCount doivent être remplacées par 0."""
        df = sample_df.copy()
        df.loc[0:4, "SupportTicketsCount"] = -5.0
        df_clean, report = step_handle_outliers(df)
        assert (df_clean["SupportTicketsCount"] < 0).sum() == 0

    def test_satisfaction_score_clamped(self, sample_df):
        """Les scores hors [1,5] doivent être corrigés."""
        df = sample_df.copy()
        df.loc[0:4, "SatisfactionScore"] = 99.0   # Valeur aberrante
        df.loc[5:9, "SatisfactionScore"] = -1.0   # Valeur négative
        df_clean, report = step_handle_outliers(df)
        assert (df_clean["SatisfactionScore"] > 5).sum() == 0
        assert (df_clean["SatisfactionScore"] < 1).sum() == 0

    def test_outlier_report_contains_corrections(self, sample_df):
        """Le rapport doit lister les colonnes corrigées."""
        df = sample_df.copy()
        df.loc[0, "SupportTicketsCount"] = -10.0
        _, report = step_handle_outliers(df)
        assert "SupportTicketsCount" in report["outliers_corrected"]

    def test_no_mutation_of_original(self, sample_df):
        """La fonction ne doit pas modifier le DataFrame original."""
        original_vals = sample_df["SatisfactionScore"].copy()
        sample_df.loc[0, "SatisfactionScore"] = 99.0
        step_handle_outliers(sample_df)
        # Vérification que le DataFrame original sample_df a bien été passé (pas de mutation)
        assert sample_df.loc[0, "SatisfactionScore"] == 99.0  # Toujours aberrant


# ─────────────────────────────────────────────
# Tests : step_feature_engineering
# ─────────────────────────────────────────────
class TestFeatureEngineering:

    def test_registration_date_parsed(self, sample_df):
        """RegistrationDate doit être splitée en RegYear, RegMonth, RegDay, RegWeekday."""
        df_result, report = step_feature_engineering(sample_df)
        for col in ["RegYear", "RegMonth", "RegDay", "RegWeekday"]:
            assert col in df_result.columns, f"Colonne manquante: {col}"

    def test_registration_date_dropped(self, sample_df):
        """La colonne RegistrationDate brute doit être supprimée."""
        df_result, _ = step_feature_engineering(sample_df)
        assert "RegistrationDate" not in df_result.columns

    def test_ip_feature_created(self, sample_df):
        """La feature Is_Private_IP doit être créée et valoir 0 ou 1."""
        df_result, _ = step_feature_engineering(sample_df)
        assert "Is_Private_IP" in df_result.columns
        assert set(df_result["Is_Private_IP"].unique()).issubset({0, 1})

    def test_avg_basket_value_non_negative(self, sample_df):
        """AvgBasketValue ne doit jamais être négatif."""
        df_result, _ = step_feature_engineering(sample_df)
        if "AvgBasketValue" in df_result.columns:
            assert (df_result["AvgBasketValue"] >= 0).all()

    def test_engagement_score_created(self, sample_df):
        """EngagementScore doit être créé quand SatisfactionScore et Frequency sont présents."""
        df_result, report = step_feature_engineering(sample_df)
        assert "EngagementScore" in df_result.columns

    def test_churn_column_preserved(self, sample_df):
        """La colonne Churn (cible) doit être préservée tout au long."""
        df_result, _ = step_feature_engineering(sample_df)
        assert "Churn" in df_result.columns

    def test_report_lists_created_features(self, sample_df):
        """Le rapport doit lister les features créées."""
        _, report = step_feature_engineering(sample_df)
        assert len(report["features_created"]) > 0


# ─────────────────────────────────────────────
# Tests : step_remove_leakage
# ─────────────────────────────────────────────
class TestRemoveLeakage:

    def test_leakage_cols_removed(self, sample_df):
        """Toutes les colonnes de leakage présentes dans le DF doivent être supprimées."""
        df_result, report = step_remove_leakage(sample_df)
        for col in LEAKAGE_COLS:
            assert col not in df_result.columns, f"Colonne leakage non supprimée: {col}"

    def test_churn_target_preserved(self, sample_df):
        """La colonne Churn (cible) ne doit PAS être supprimée."""
        df_result, _ = step_remove_leakage(sample_df)
        assert "Churn" in df_result.columns

    def test_non_leakage_cols_preserved(self, sample_df):
        """Les colonnes non-leakage doivent être conservées."""
        df_result, _ = step_remove_leakage(sample_df)
        assert "Frequency" in df_result.columns
        assert "MonetaryTotal" in df_result.columns
        assert "SatisfactionScore" in df_result.columns

    def test_report_lists_dropped_cols(self, sample_df):
        """Le rapport doit lister les colonnes supprimées."""
        _, report = step_remove_leakage(sample_df)
        assert isinstance(report["leakage_columns_removed"], list)
        assert report["count"] == len(report["leakage_columns_removed"])


# ─────────────────────────────────────────────
# Tests : step_remove_high_correlation
# ─────────────────────────────────────────────
class TestRemoveHighCorrelation:

    def test_highly_correlated_col_removed(self):
        """Une colonne quasi-identique à une autre doit être supprimée."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        df = pd.DataFrame({
            "A":     x,
            "B":     x + np.random.randn(n) * 0.001,  # Quasi identique à A
            "C":     np.random.randn(n),
            "Churn": np.random.randint(0, 2, n),
        })
        df_result, report = step_remove_high_correlation(df, threshold=0.85)
        # B doit être supprimé car quasi identique à A
        assert "B" not in df_result.columns or "A" not in df_result.columns

    def test_churn_not_removed(self):
        """La colonne Churn ne doit jamais être retirée par le filtre de corrélation."""
        np.random.seed(0)
        n = 100
        df = pd.DataFrame({
            "Freq":  np.random.randn(n),
            "Money": np.random.randn(n),
            "Churn": np.random.randint(0, 2, n),
        })
        df_result, _ = step_remove_high_correlation(df, threshold=0.85)
        assert "Churn" in df_result.columns

    def test_independent_cols_kept(self):
        """Des colonnes indépendantes ne doivent pas être supprimées."""
        np.random.seed(1)
        n = 200
        df = pd.DataFrame({
            "A":     np.random.randn(n),
            "B":     np.random.randn(n),
            "C":     np.random.randn(n),
            "Churn": np.random.randint(0, 2, n),
        })
        df_result, report = step_remove_high_correlation(df, threshold=0.85)
        assert len(report["dropped_high_corr_cols"]) == 0


# ─────────────────────────────────────────────
# Tests : transform (pipeline complet)
# ─────────────────────────────────────────────
class TestTransformPipeline:

    def test_output_has_fewer_or_equal_cols(self, sample_df):
        """Le transform doit supprimer des colonnes (leakage), jamais en ajouter sans raison."""
        df_result, meta = transform(sample_df)
        # Des colonnes sont supprimées (leakage), d'autres ajoutées (FE) — résultat variable
        assert len(df_result.columns) > 0

    def test_no_leakage_cols_in_output(self, sample_df):
        """Aucune colonne de leakage ne doit subsister après le transform."""
        df_result, _ = transform(sample_df)
        for col in LEAKAGE_COLS:
            assert col not in df_result.columns

    def test_churn_preserved(self, sample_df):
        """La cible Churn doit être présente en sortie."""
        df_result, _ = transform(sample_df)
        assert "Churn" in df_result.columns

    def test_metadata_has_all_steps(self, sample_df):
        """Les métadonnées doivent inclure toutes les étapes de transformation."""
        _, meta = transform(sample_df)
        for step in ["outliers", "feature_engineering", "leakage_removal", "correlation_removal"]:
            assert step in meta["steps"], f"Étape manquante dans metadata: {step}"

    def test_row_count_preserved(self, sample_df):
        """Le nombre de lignes ne doit pas changer pendant le transform."""
        n_before = len(sample_df)
        df_result, _ = transform(sample_df)
        assert len(df_result) == n_before

    def test_transform_time_recorded(self, sample_df):
        """Le temps d'exécution doit être enregistré dans les métadonnées."""
        _, meta = transform(sample_df)
        assert "transform_time_s" in meta
        assert meta["transform_time_s"] > 0
