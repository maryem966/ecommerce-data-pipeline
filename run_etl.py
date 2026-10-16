"""
Point d'entrée principal — Pipeline ETL + Clustering
======================================================
Orchestre l'exécution complète :
  1. Pipeline ETL (Extract → Transform → Load)
  2. Clustering KMeans (segmentation non supervisée)
  3. Monitoring & rapport final

Usage :
    python run_etl.py                      # Pipeline complet
    python run_etl.py --etl-only           # ETL sans clustering
    python run_etl.py --cluster-only       # Clustering sur données existantes
    python run_etl.py --data path/to/file  # Fichier source personnalisé
"""

import argparse
import logging
import os
import sys
import time

# Support UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ─────────────────────────────────────────────
# Configuration du logging global
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join("reports", "etl_run.log") if os.path.exists("reports") else "etl_run.log",
            mode="a",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("run_etl")


def run_full_pipeline(data_path=None, skip_clustering=False, skip_etl=False):
    """
    Exécute le pipeline ETL complet avec clustering et monitoring.

    Args:
        data_path       : chemin vers le fichier source (None = défaut)
        skip_clustering : si True, ne lance pas le clustering
        skip_etl        : si True, utilise les données déjà transformées
    """
    from monitoring.pipeline_logger import PipelineMetricsCollector

    collector = PipelineMetricsCollector()
    t_global = time.perf_counter()

    df_clean = None
    etl_report = None

    # ── ÉTAPE 1 : PIPELINE ETL ─────────────────
    if not skip_etl:
        logger.info("=" * 60)
        logger.info("  PHASE 1 : PIPELINE ETL")
        logger.info("=" * 60)

        from etl.pipeline import ETLPipeline

        pipeline = ETLPipeline(data_path=data_path)
        etl_run = pipeline.run()

        if etl_run.status != "success":
            logger.error(f"Pipeline ETL échoué : {etl_run.error}")
            sys.exit(1)

        # Enregistrement monitoring
        for step_name, step_info in etl_run.steps.items():
            rows_in  = None
            rows_out = None
            if step_name == "Extract":
                rows_in = rows_out = etl_run.extract_meta.get("quality", {}).get("total_rows")
                collector.record_data_quality(etl_run.extract_meta.get("quality", {}))
            elif step_name == "Transform":
                in_shape  = etl_run.transform_meta.get("input_shape",  [None])
                out_shape = etl_run.transform_meta.get("output_shape", [None])
                rows_in, rows_out = in_shape[0], out_shape[0]
            elif step_name == "Load":
                rows_in = rows_out = etl_run.load_meta.get("load", {}).get("sqlite", {}).get("rows_in_db")

            collector.record_step(
                step_name=step_name,
                duration_s=step_info.get("duration_s", 0),
                rows_in=rows_in,
                rows_out=rows_out,
            )

        # Charger les données transformées pour le clustering
        import pandas as pd
        processed_path = "data/processed/processed_data.csv"
        if os.path.exists(processed_path):
            df_clean = pd.read_csv(processed_path)
            logger.info(f"Données transformées chargées : {df_clean.shape}")
        else:
            logger.warning("Fichier processed_data.csv non trouvé, clustering ignoré.")
            skip_clustering = True

    else:
        # Charger les données existantes directement
        import pandas as pd
        processed_path = "data/processed/processed_data.csv"
        if not os.path.exists(processed_path):
            logger.error(f"Fichier {processed_path} introuvable. Lancez d'abord le pipeline ETL.")
            sys.exit(1)
        df_clean = pd.read_csv(processed_path)
        logger.info(f"Données chargées depuis {processed_path} : {df_clean.shape}")

    # ── ÉTAPE 2 : CLUSTERING ───────────────────
    if not skip_clustering and df_clean is not None:
        logger.info("=" * 60)
        logger.info("  PHASE 2 : CLUSTERING KMeans")
        logger.info("=" * 60)

        from clustering.kmeans_segmentation import run_clustering
        from clustering.cluster_profiler    import generate_cluster_report

        t_cluster = time.perf_counter()
        cluster_report = run_clustering(df_clean)
        cluster_duration = round(time.perf_counter() - t_cluster, 4)

        # Profils et personas
        import pandas as pd
        profiles = pd.DataFrame(cluster_report["cluster_profiles"])
        report_path = generate_cluster_report(
            profiles,
            cluster_report["cluster_distribution"],
        )

        collector.record_clustering(cluster_report)
        collector.record_step(
            step_name="Clustering",
            duration_s=cluster_duration,
            rows_in=len(df_clean),
            rows_out=len(df_clean),
            extra={
                "k_optimal":  cluster_report["k_optimal"],
                "silhouette": cluster_report["final_silhouette"],
            },
        )

        logger.info(f"Rapport personas → {report_path}")
        for path_key, path_val in cluster_report["plot_paths"].items():
            logger.info(f"  Graphique {path_key} → {path_val}")

    # ── MONITORING FINAL ───────────────────────
    logger.info("=" * 60)
    logger.info("  PHASE 3 : MONITORING & RAPPORT")
    logger.info("=" * 60)

    collector.print_dashboard()
    metrics_path = collector.export_metrics()
    logger.info(f"Métriques de monitoring → {metrics_path}")

    total = round(time.perf_counter() - t_global, 2)
    logger.info(f"\n✅  Pipeline complet terminé en {total}s")
    logger.info(f"    Résultats dans : reports/ | data/warehouse/ | models/")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="E-Commerce ML — Pipeline ETL + Clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python run_etl.py                          # Pipeline complet
  python run_etl.py --etl-only               # ETL uniquement
  python run_etl.py --cluster-only           # Clustering sur données existantes
  python run_etl.py --data data/raw/mon_fichier.csv
        """
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Chemin vers le fichier source CSV (défaut: data/raw/...)"
    )
    parser.add_argument(
        "--etl-only", action="store_true",
        help="Exécuter uniquement le pipeline ETL (sans clustering)"
    )
    parser.add_argument(
        "--cluster-only", action="store_true",
        help="Exécuter uniquement le clustering (données déjà transformées requises)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Créer le dossier reports avant le logging file handler
    os.makedirs("reports", exist_ok=True)

    args = parse_args()

    run_full_pipeline(
        data_path=args.data,
        skip_clustering=args.etl_only,
        skip_etl=args.cluster_only,
    )
