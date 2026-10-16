"""
ETL Step 1 — EXTRACT
=====================
Responsabilités :
  - Chargement du fichier source (CSV/XLSX)
  - Validation du schéma : colonnes attendues, types, cardinalités
  - Production d'un rapport de qualité des données brutes
  - Aucune transformation n'a lieu ici (séparation des responsabilités)
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Schéma attendu du fichier source
# ─────────────────────────────────────────────
EXPECTED_COLUMNS = {
    # Identifiant
    "CustomerID": "object",
    # Métriques RFM
    "Recency": "float64",
    "Frequency": "float64",
    "MonetaryTotal": "float64",
    "MonetaryAvg": "float64",
    # Comportement
    "CustomerTenureDays": "float64",
    "SatisfactionScore": "float64",
    "SupportTicketsCount": "float64",
    "ReturnRatio": "float64",
    # Cible
    "Churn": "int64",
}

REQUIRED_COLUMNS = list(EXPECTED_COLUMNS.keys())


# ─────────────────────────────────────────────
# Validation du schéma
# ─────────────────────────────────────────────
def validate_schema(df: pd.DataFrame) -> dict:
    """
    Valide la présence des colonnes critiques et retourne un rapport de schéma.

    Retourne un dict avec :
      - missing_required   : colonnes obligatoires absentes
      - extra_columns      : colonnes non attendues (info)
      - total_columns      : nombre total de colonnes
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra   = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    report = {
        "missing_required_columns": missing,
        "extra_columns_count": len(extra),
        "total_columns": len(df.columns),
        "schema_valid": len(missing) == 0,
    }

    if missing:
        logger.warning(f"[EXTRACT] Colonnes manquantes : {missing}")
    else:
        logger.info("[EXTRACT] Schéma validé — toutes les colonnes requises sont présentes.")

    return report


# ─────────────────────────────────────────────
# Rapport de qualité des données
# ─────────────────────────────────────────────
def compute_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Calcule un rapport de qualité complet sur le DataFrame brut.

    Métriques calculées :
      - Taux de valeurs nulles par colonne
      - Nombre de doublons
      - Distribution de la variable cible (Churn)
      - Statistiques des colonnes numériques clés
    """
    total_rows = len(df)

    # Nulls
    null_counts  = df.isnull().sum()
    null_pct     = (null_counts / total_rows * 100).round(2)
    null_summary = {col: {"count": int(null_counts[col]), "pct": float(null_pct[col])}
                    for col in df.columns if null_counts[col] > 0}

    # Doublons
    duplicate_count = int(df.duplicated().sum())

    # Distribution cible
    churn_dist = {}
    if "Churn" in df.columns:
        vc = df["Churn"].value_counts(normalize=True).round(4)
        churn_dist = {str(k): float(v) for k, v in vc.items()}

    # Statistiques numériques clés
    numeric_stats = {}
    for col in ["Recency", "Frequency", "MonetaryTotal", "SatisfactionScore", "SupportTicketsCount"]:
        if col in df.columns:
            s = df[col].describe()
            numeric_stats[col] = {
                "min": round(float(s["min"]), 2),
                "max": round(float(s["max"]), 2),
                "mean": round(float(s["mean"]), 2),
                "std": round(float(s["std"]), 2),
                "null_count": int(df[col].isnull().sum()),
            }

    # Score global de qualité (0–100)
    avg_null_pct   = float(null_pct.mean())
    duplicate_pct  = duplicate_count / total_rows * 100
    quality_score  = max(0.0, round(100 - avg_null_pct - duplicate_pct, 1))

    report = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "duplicate_rows": duplicate_count,
        "duplicate_pct": round(duplicate_pct, 2),
        "null_summary": null_summary,
        "avg_null_pct": round(avg_null_pct, 2),
        "churn_distribution": churn_dist,
        "numeric_stats": numeric_stats,
        "quality_score": quality_score,
    }

    logger.info(
        f"[EXTRACT] Qualité des données — Score: {quality_score}/100 | "
        f"Nulls moy.: {avg_null_pct:.1f}% | Doublons: {duplicate_count}"
    )
    return report


# ─────────────────────────────────────────────
# Étape principale : Extract
# ─────────────────────────────────────────────
def extract(data_path: str = None) -> tuple[pd.DataFrame, dict]:
    """
    Étape EXTRACT du pipeline ETL.

    1. Charge le fichier source (CSV ou XLSX)
    2. Valide le schéma
    3. Calcule le rapport de qualité
    4. Retourne (DataFrame brut, metadata_dict)

    Args:
        data_path: chemin vers le fichier source. Si None, utilise le chemin par défaut.

    Returns:
        df       : DataFrame brut non transformé
        metadata : dictionnaire de métriques et de rapport qualité
    """
    if data_path is None:
        data_path = os.path.join(
            "data", "raw",
            "retail_customers_COMPLETE_CATEGORICAL - retail_customers_COMPLETE_CATEGORICAL.csv"
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[EXTRACT] Fichier source introuvable : {data_path}")

    logger.info(f"[EXTRACT] Chargement du fichier : {data_path}")
    start = datetime.now()

    # Chargement selon l'extension
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(data_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"[EXTRACT] Format non supporté : {ext}")

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"[EXTRACT] Fichier chargé en {elapsed:.2f}s — shape: {df.shape}")

    # Validation du schéma
    schema_report = validate_schema(df)

    # Rapport qualité
    quality_report = compute_data_quality_report(df)

    metadata = {
        "source_path":    data_path,
        "extract_time_s": round(elapsed, 4),
        "schema":         schema_report,
        "quality":        quality_report,
        "extracted_at":   datetime.now().isoformat(),
    }

    return df, metadata


# ─────────────────────────────────────────────
# Utilitaire : sauvegarde du rapport
# ─────────────────────────────────────────────
def save_extract_report(metadata: dict, output_dir: str = "reports") -> str:
    """Sauvegarde le rapport d'extraction en JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"extract_report_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"[EXTRACT] Rapport sauvegardé → {path}")
    return path
