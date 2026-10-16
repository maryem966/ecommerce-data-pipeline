"""
ETL Step 3 — LOAD
==================
Responsabilités :
  - Persistance du DataFrame transformé dans SQLite (data warehouse)
  - Export CSV du dataset nettoyé
  - Sauvegarde du rapport de pipeline complet (JSON)
  - Requêtes SQL d'exemple pour valider le chargement

Le Data Warehouse SQLite permet de requêter les données avec SQL,
simulant un entrepôt de données léger sans infrastructure externe.
"""

import pandas as pd
import sqlite3
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Chemins par défaut
# ─────────────────────────────────────────────
DEFAULT_DB_PATH  = os.path.join("data", "warehouse", "ecommerce.db")
DEFAULT_CSV_PATH = os.path.join("data", "processed", "processed_data.csv")


# ─────────────────────────────────────────────
# Chargement dans SQLite (Data Warehouse)
# ─────────────────────────────────────────────
def load_to_sqlite(
    df: pd.DataFrame,
    db_path: str = DEFAULT_DB_PATH,
    table_name: str = "customers_transformed",
    if_exists: str = "replace",
) -> dict:
    """
    Charge le DataFrame dans une base SQLite locale (Data Warehouse).

    Crée deux tables :
      - customers_transformed  : données clients nettoyées et enrichies
      - pipeline_runs          : journal des exécutions de la pipeline (audit trail)

    Args:
        df         : DataFrame transformé
        db_path    : chemin vers le fichier SQLite
        table_name : nom de la table principale
        if_exists  : comportement si la table existe ('replace' ou 'append')

    Returns:
        Rapport dict avec nb de lignes insérées et chemin DB
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    logger.info(f"[LOAD] Connexion à SQLite : {db_path}")

    with sqlite3.connect(db_path) as conn:
        # Table principale : données clients
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        logger.info(f"[LOAD] Table '{table_name}' chargée — {len(df)} lignes × {len(df.columns)} colonnes")

        # Table d'audit : journal des runs
        run_log = pd.DataFrame([{
            "run_id":      datetime.now().strftime("%Y%m%d_%H%M%S"),
            "table_name":  table_name,
            "rows_loaded": len(df),
            "cols_loaded": len(df.columns),
            "loaded_at":   datetime.now().isoformat(),
        }])
        run_log.to_sql("pipeline_runs", conn, if_exists="append", index=False)

        # Vérification rapide par SQL
        count = pd.read_sql(f"SELECT COUNT(*) as n FROM {table_name}", conn)["n"][0]

    report = {
        "db_path":      db_path,
        "table_name":   table_name,
        "rows_in_db":   int(count),
        "cols_in_db":   len(df.columns),
        "loaded_at":    datetime.now().isoformat(),
    }
    logger.info(f"[LOAD] Vérification SQL — {count} lignes présentes dans '{table_name}'")
    return report


# ─────────────────────────────────────────────
# Export CSV
# ─────────────────────────────────────────────
def load_to_csv(df: pd.DataFrame, csv_path: str = DEFAULT_CSV_PATH) -> dict:
    """
    Exporte le DataFrame transformé en CSV.

    Args:
        df       : DataFrame transformé
        csv_path : chemin de sortie

    Returns:
        Rapport dict avec taille du fichier
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    size_kb = os.path.getsize(csv_path) / 1024
    logger.info(f"[LOAD] CSV exporté → {csv_path} ({size_kb:.1f} KB)")
    return {"csv_path": csv_path, "size_kb": round(size_kb, 2), "rows": len(df)}


# ─────────────────────────────────────────────
# Sauvegarde du rapport de pipeline
# ─────────────────────────────────────────────
def save_pipeline_report(report: dict, output_dir: str = "reports") -> str:
    """
    Sauvegarde le rapport complet du run ETL en JSON.

    Ce rapport contient :
      - Les métriques de chaque étape (Extract, Transform, Load)
      - Les temps d'exécution
      - Les volumes de données
      - Les scores de qualité
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"pipeline_run_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"[LOAD] Rapport pipeline sauvegardé → {path}")
    return path


# ─────────────────────────────────────────────
# Requêtes SQL de validation
# ─────────────────────────────────────────────
def run_validation_queries(db_path: str = DEFAULT_DB_PATH) -> dict:
    """
    Exécute des requêtes SQL de validation sur le Data Warehouse.

    Retourne un dict de résultats — démontre la capacité à interroger
    les données transformées directement en SQL.
    """
    results = {}

    with sqlite3.connect(db_path) as conn:
        # 1. Nombre total de clients
        results["total_customers"] = int(
            pd.read_sql("SELECT COUNT(*) as n FROM customers_transformed", conn)["n"][0]
        )

        # 2. Distribution du churn
        churn_df = pd.read_sql(
            "SELECT Churn, COUNT(*) as count FROM customers_transformed GROUP BY Churn", conn
        )
        results["churn_distribution"] = churn_df.set_index("Churn")["count"].to_dict()

        # 3. Moyenne des métriques comportementales
        avg_df = pd.read_sql(
            """
            SELECT
                ROUND(AVG(Frequency), 2)          AS avg_frequency,
                ROUND(AVG(MonetaryTotal), 2)       AS avg_monetary,
                ROUND(AVG(SatisfactionScore), 2)   AS avg_satisfaction,
                ROUND(AVG(CustomerTenureDays), 2)  AS avg_tenure_days
            FROM customers_transformed
            """,
            conn
        )
        results["avg_metrics"] = avg_df.iloc[0].to_dict()

        # 4. Top 5 runs d'audit
        try:
            runs_df = pd.read_sql(
                "SELECT * FROM pipeline_runs ORDER BY loaded_at DESC LIMIT 5", conn
            )
            results["recent_pipeline_runs"] = runs_df.to_dict(orient="records")
        except Exception:
            results["recent_pipeline_runs"] = []

    logger.info(f"[LOAD] Requêtes SQL de validation exécutées : {list(results.keys())}")
    return results


# ─────────────────────────────────────────────
# Orchestrateur Load
# ─────────────────────────────────────────────
def load(
    df_clean: pd.DataFrame,
    pipeline_metadata: dict,
    db_path: str = DEFAULT_DB_PATH,
    csv_path: str = DEFAULT_CSV_PATH,
) -> dict:
    """
    Étape LOAD du pipeline ETL.

    1. Charge dans SQLite
    2. Exporte en CSV
    3. Valide via requêtes SQL
    4. Sauvegarde le rapport complet

    Args:
        df_clean          : DataFrame transformé (sortie du Transform)
        pipeline_metadata : métriques des étapes précédentes (Extract + Transform)
        db_path           : chemin SQLite
        csv_path          : chemin CSV de sortie

    Returns:
        Rapport complet du run ETL
    """
    start = datetime.now()
    logger.info(f"[LOAD] Début — {len(df_clean)} lignes à charger")

    # SQLite
    db_report  = load_to_sqlite(df_clean, db_path)

    # CSV
    csv_report = load_to_csv(df_clean, csv_path)

    # Requêtes de validation
    validation = run_validation_queries(db_path)

    elapsed = (datetime.now() - start).total_seconds()

    # Rapport final complet
    full_report = {
        **pipeline_metadata,
        "load": {
            "sqlite":     db_report,
            "csv":        csv_report,
            "validation": validation,
            "load_time_s": round(elapsed, 4),
        },
        "pipeline_completed_at": datetime.now().isoformat(),
    }

    # Sauvegarde JSON
    report_path = save_pipeline_report(full_report)
    full_report["report_path"] = report_path

    logger.info(f"[LOAD] Terminé en {elapsed:.2f}s")
    return full_report
