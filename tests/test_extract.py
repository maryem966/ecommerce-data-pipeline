"""
Tests unitaires — Étape Extract
=================================
Teste les fonctions de validation du schéma et de qualité des données
de l'étape Extract du pipeline ETL.

Exécution : pytest tests/test_extract.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajout du répertoire racine au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from etl.extract import validate_schema, compute_data_quality_report, REQUIRED_COLUMNS


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def valid_df():
    """DataFrame valide avec toutes les colonnes requises."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "CustomerID":        [f"C{i}" for i in range(n)],
        "Recency":           np.random.uniform(0, 365, n),
        "Frequency":         np.random.randint(1, 50, n).astype(float),
        "MonetaryTotal":     np.random.uniform(100, 5000, n),
        "MonetaryAvg":       np.random.uniform(10, 200, n),
        "CustomerTenureDays": np.random.randint(30, 1000, n).astype(float),
        "SatisfactionScore": np.random.uniform(1, 5, n),
        "SupportTicketsCount": np.random.randint(0, 10, n).astype(float),
        "ReturnRatio":       np.random.uniform(0, 0.5, n),
        "Churn":             np.random.randint(0, 2, n),
    })


@pytest.fixture
def df_with_nulls(valid_df):
    """DataFrame avec des valeurs manquantes."""
    df = valid_df.copy()
    df.loc[0:9, "SatisfactionScore"] = np.nan   # 10% de nulls
    df.loc[5:9, "MonetaryTotal"] = np.nan
    return df


@pytest.fixture
def df_missing_required_col(valid_df):
    """DataFrame sans la colonne 'Churn' (obligatoire)."""
    return valid_df.drop(columns=["Churn"])


@pytest.fixture
def df_with_duplicates(valid_df):
    """DataFrame contenant des lignes dupliquées."""
    return pd.concat([valid_df, valid_df.iloc[:10]], ignore_index=True)


# ─────────────────────────────────────────────
# Tests : validate_schema
# ─────────────────────────────────────────────
class TestValidateSchema:

    def test_valid_df_passes_schema(self, valid_df):
        """Un DataFrame avec toutes les colonnes requises doit être valide."""
        report = validate_schema(valid_df)
        assert report["schema_valid"] is True
        assert report["missing_required_columns"] == []

    def test_missing_column_detected(self, df_missing_required_col):
        """Une colonne manquante doit être signalée."""
        report = validate_schema(df_missing_required_col)
        assert report["schema_valid"] is False
        assert "Churn" in report["missing_required_columns"]

    def test_all_required_columns_counted(self, valid_df):
        """Le rapport doit comptabiliser le bon nombre de colonnes."""
        report = validate_schema(valid_df)
        assert report["total_columns"] == len(valid_df.columns)

    def test_extra_columns_counted(self, valid_df):
        """Les colonnes supplémentaires doivent être comptées (non bloquantes)."""
        df = valid_df.copy()
        df["ExtraCol1"] = 0
        df["ExtraCol2"] = "test"
        report = validate_schema(df)
        assert report["extra_columns_count"] >= 2
        # Le schéma reste valide même avec des colonnes supplémentaires
        assert report["schema_valid"] is True

    def test_empty_df_schema(self):
        """Un DataFrame vide doit signaler toutes les colonnes manquantes."""
        report = validate_schema(pd.DataFrame())
        assert report["schema_valid"] is False
        assert len(report["missing_required_columns"]) == len(REQUIRED_COLUMNS)


# ─────────────────────────────────────────────
# Tests : compute_data_quality_report
# ─────────────────────────────────────────────
class TestDataQualityReport:

    def test_clean_df_high_quality_score(self, valid_df):
        """Un DataFrame propre doit avoir un score de qualité élevé (>= 90)."""
        report = compute_data_quality_report(valid_df)
        assert report["quality_score"] >= 90

    def test_null_summary_detected(self, df_with_nulls):
        """Les colonnes avec des nulls doivent apparaître dans null_summary."""
        report = compute_data_quality_report(df_with_nulls)
        assert "SatisfactionScore" in report["null_summary"]
        assert report["null_summary"]["SatisfactionScore"]["count"] > 0

    def test_duplicate_detection(self, df_with_duplicates):
        """Les lignes dupliquées doivent être comptées correctement."""
        report = compute_data_quality_report(df_with_duplicates)
        assert report["duplicate_rows"] == 10

    def test_churn_distribution_computed(self, valid_df):
        """La distribution de Churn doit être calculée."""
        report = compute_data_quality_report(valid_df)
        churn_dist = report["churn_distribution"]
        assert len(churn_dist) > 0
        total_pct = sum(churn_dist.values())
        assert abs(total_pct - 1.0) < 0.01  # Sum ≈ 100%

    def test_total_rows_correct(self, valid_df):
        """Le nombre de lignes doit correspondre."""
        report = compute_data_quality_report(valid_df)
        assert report["total_rows"] == len(valid_df)

    def test_quality_score_range(self, valid_df):
        """Le score de qualité doit être entre 0 et 100."""
        report = compute_data_quality_report(valid_df)
        assert 0 <= report["quality_score"] <= 100

    def test_numeric_stats_present(self, valid_df):
        """Les statistiques des colonnes numériques clés doivent être présentes."""
        report = compute_data_quality_report(valid_df)
        stats = report["numeric_stats"]
        # Au moins Frequency et MonetaryTotal doivent être présents
        assert "Frequency" in stats
        assert "MonetaryTotal" in stats
        for col, s in stats.items():
            assert "min" in s
            assert "max" in s
            assert s["min"] <= s["max"]
