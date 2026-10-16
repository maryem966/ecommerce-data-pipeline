"""
ETL Step 2 — TRANSFORM
========================
Responsabilités :
  - Nettoyage des outliers
  - Feature engineering (nouvelles variables comportementales)
  - Suppression des colonnes à data leakage
  - Réduction des corrélations
  - Retourne un DataFrame propre et enrichi prêt pour le Load ou l'entraînement

Chaque transformation est une fonction pure et testable indépendamment.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Colonnes à supprimer (data leakage / inutiles)
# ─────────────────────────────────────────────
LEAKAGE_COLS = [
    "CustomerID",
    "NewsletterSubscribed",
    "ChurnRiskCategory",
    "ChurnRisk",
    "AccountStatus",
    "CustomerType",
    "RFMSegment",
    "LoyaltyLevel",
    "FavoriteSeason",
    "Recency",          # Churn = 1 if Recency > 90 else 0
]


# ─────────────────────────────────────────────
# Transformation 1 : Correction des outliers
# ─────────────────────────────────────────────
def step_handle_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Détecte et corrige les valeurs aberrantes dans les colonnes clés.

    - SupportTicketsCount : plafonnement au 99e percentile, correction des négatifs
    - SatisfactionScore   : remplacement des valeurs hors [1,5] par la médiane valide

    Retourne (df_clean, rapport_dict).
    """
    report = {"outliers_corrected": {}}
    df = df.copy()

    if "SupportTicketsCount" in df.columns:
        upper = df["SupportTicketsCount"].quantile(0.99)
        n_above = int((df["SupportTicketsCount"] > upper).sum())
        n_neg   = int((df["SupportTicketsCount"] < 0).sum())
        df.loc[df["SupportTicketsCount"] > upper, "SupportTicketsCount"] = upper
        df.loc[df["SupportTicketsCount"] < 0,     "SupportTicketsCount"] = 0
        report["outliers_corrected"]["SupportTicketsCount"] = {"above_99p": n_above, "negatives": n_neg}

    if "SatisfactionScore" in df.columns:
        valid = df.loc[(df["SatisfactionScore"] >= 1) & (df["SatisfactionScore"] <= 5), "SatisfactionScore"]
        median_score = valid.median() if len(valid) > 0 else 3.0
        n_invalid = int(((df["SatisfactionScore"] > 5) | (df["SatisfactionScore"] < 1)).sum())
        df.loc[(df["SatisfactionScore"] > 5) | (df["SatisfactionScore"] < 1), "SatisfactionScore"] = median_score
        report["outliers_corrected"]["SatisfactionScore"] = {"invalid_replaced": n_invalid, "median_used": median_score}

    logger.info(f"[TRANSFORM] Outliers corrigés : {report['outliers_corrected']}")
    return df, report


# ─────────────────────────────────────────────
# Transformation 2 : Feature Engineering
# ─────────────────────────────────────────────
def step_feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Crée de nouvelles features comportementales et parse les dates.

    Nouvelles features créées :
      - RegYear, RegMonth, RegDay, RegWeekday  (depuis RegistrationDate)
      - Is_Private_IP                          (depuis LastLoginIP)
      - AvgBasketValue                         (MonetaryTotal / Frequency)
      - EngagementScore                        (combinaison SatisfactionScore + Frequency)
      - TenureFrequencyRatio                   (Frequency / CustomerTenureDays)
    """
    report = {"features_created": [], "columns_dropped": []}
    df = df.copy()

    # Parsing date d'inscription
    if "RegistrationDate" in df.columns:
        reg_date = pd.to_datetime(df["RegistrationDate"], dayfirst=True, errors="coerce")
        df["RegYear"]    = reg_date.dt.year.fillna(-1).astype(int)
        df["RegMonth"]   = reg_date.dt.month.fillna(-1).astype(int)
        df["RegDay"]     = reg_date.dt.day.fillna(-1).astype(int)
        df["RegWeekday"] = reg_date.dt.weekday.fillna(-1).astype(int)
        df = df.drop(columns=["RegistrationDate"])
        report["features_created"] += ["RegYear", "RegMonth", "RegDay", "RegWeekday"]
        report["columns_dropped"].append("RegistrationDate")

    # IP privée ou publique
    if "LastLoginIP" in df.columns:
        def is_private_ip(ip):
            try:
                return 1 if str(ip).startswith(("192.168.", "10.", "172.")) else 0
            except Exception:
                return 0
        df["Is_Private_IP"] = df["LastLoginIP"].apply(is_private_ip)
        df = df.drop(columns=["LastLoginIP"])
        report["features_created"].append("Is_Private_IP")
        report["columns_dropped"].append("LastLoginIP")

    # Valeur moyenne du panier
    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasketValue"] = np.where(
            df["Frequency"] > 0,
            df["MonetaryTotal"] / df["Frequency"],
            0
        )
        report["features_created"].append("AvgBasketValue")

    # Score d'engagement composite
    if "SatisfactionScore" in df.columns and "Frequency" in df.columns:
        df["EngagementScore"] = (
            df["SatisfactionScore"].fillna(3) * np.log1p(df["Frequency"].fillna(0))
        ).round(4)
        report["features_created"].append("EngagementScore")

    # Ratio fréquence / ancienneté
    if "Frequency" in df.columns and "CustomerTenureDays" in df.columns:
        df["TenureFrequencyRatio"] = np.where(
            df["CustomerTenureDays"] > 0,
            df["Frequency"] / df["CustomerTenureDays"],
            0
        ).round(6)
        report["features_created"].append("TenureFrequencyRatio")

    logger.info(
        f"[TRANSFORM] Features créées: {report['features_created']} | "
        f"Colonnes supprimées: {report['columns_dropped']}"
    )
    return df, report


# ─────────────────────────────────────────────
# Transformation 3 : Suppression du data leakage
# ─────────────────────────────────────────────
def step_remove_leakage(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Supprime les colonnes à data leakage (variables dérivées directement de la cible).
    Préserve la colonne 'Churn' (cible).
    """
    dropped = []
    df = df.copy()
    for col in LEAKAGE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped.append(col)

    report = {"leakage_columns_removed": dropped, "count": len(dropped)}
    logger.info(f"[TRANSFORM] Leakage supprimé — {len(dropped)} colonnes retirées : {dropped}")
    return df, report


# ─────────────────────────────────────────────
# Transformation 4 : Suppression des hautes corrélations
# ─────────────────────────────────────────────
def step_remove_high_correlation(
    df: pd.DataFrame, threshold: float = 0.85
) -> tuple[pd.DataFrame, dict]:
    """
    Supprime les features numériques dont la corrélation dépasse le seuil.
    Réduit la multicolinéarité et le sur-apprentissage.

    Algorithme :
      1. Calcul de la matrice de corrélation absolue
      2. Pour chaque paire (i,j) avec corr > threshold, on retire j
    """
    df = df.copy()

    # Exclure 'Churn' du calcul de corrélation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Churn" in numeric_cols:
        numeric_cols.remove("Churn")

    if len(numeric_cols) < 2:
        return df, {"dropped": [], "threshold": threshold}

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    df = df.drop(columns=to_drop, errors="ignore")
    report = {"dropped_high_corr_cols": to_drop, "count": len(to_drop), "threshold": threshold}
    logger.info(f"[TRANSFORM] Corrélations supprimées : {len(to_drop)} colonnes — {to_drop}")
    return df, report


# ─────────────────────────────────────────────
# Orchestrateur des transformations
# ─────────────────────────────────────────────
def transform(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Étape TRANSFORM du pipeline ETL.

    Enchaîne dans l'ordre :
      1. Correction des outliers
      2. Feature engineering
      3. Suppression du data leakage
      4. Suppression des hautes corrélations

    Args:
        df_raw : DataFrame brut issu de l'étape Extract

    Returns:
        df_clean : DataFrame transformé, prêt pour le Load ou l'entraînement
        metadata : dictionnaire de métriques de transformation
    """
    start = datetime.now()
    logger.info(f"[TRANSFORM] Début — shape entrée: {df_raw.shape}")

    metadata = {
        "input_shape": list(df_raw.shape),
        "steps": {},
    }

    # Étape 1 — Outliers
    df, report_outliers = step_handle_outliers(df_raw)
    metadata["steps"]["outliers"] = report_outliers

    # Étape 2 — Feature Engineering
    df, report_fe = step_feature_engineering(df)
    metadata["steps"]["feature_engineering"] = report_fe

    # Étape 3 — Suppression leakage
    df, report_leakage = step_remove_leakage(df)
    metadata["steps"]["leakage_removal"] = report_leakage

    # Étape 4 — Corrélations
    df, report_corr = step_remove_high_correlation(df)
    metadata["steps"]["correlation_removal"] = report_corr

    elapsed = (datetime.now() - start).total_seconds()
    metadata["output_shape"]    = list(df.shape)
    metadata["transform_time_s"] = round(elapsed, 4)
    metadata["new_features_count"] = len(report_fe["features_created"])

    logger.info(
        f"[TRANSFORM] Terminé en {elapsed:.2f}s — "
        f"shape: {df_raw.shape} → {df.shape} | "
        f"Features créées: {len(report_fe['features_created'])}"
    )
    return df, metadata
