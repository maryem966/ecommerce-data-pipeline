"""
ETL Orchestrateur — PIPELINE
==============================
Implémente un orchestrateur DAG (Directed Acyclic Graph) simple qui :
  - Enchaîne les étapes Extract → Transform → Load
  - Mesure le temps d'exécution de chaque étape
  - Collecte et centralise les métriques de chaque étape
  - Gère les erreurs avec contexte explicite
  - Génère un rapport de run complet

Analogie avec les outils pro :
  - Ce fichier joue le rôle d'un DAG Apache Airflow
  - Chaque PipelineStep = un Task dans Airflow / un Stage dans Spark
  - Le PipelineRunner = le Scheduler/Executor
"""

import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from etl.extract   import extract, save_extract_report
from etl.transform import transform
from etl.load      import load

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Modèle d'une étape de pipeline
# ─────────────────────────────────────────────
@dataclass
class PipelineStep:
    """
    Représente une étape atomique du pipeline ETL.

    Attributs :
        name        : nom lisible de l'étape (ex: "Extract", "Transform")
        fn          : fonction à appeler
        status      : 'pending' | 'running' | 'success' | 'failed'
        duration_s  : durée d'exécution en secondes
        output      : résultat retourné par la fonction
        error       : message d'erreur si l'étape a échoué
    """
    name:       str
    fn:         Callable
    status:     str = "pending"
    duration_s: float = 0.0
    output:     Any = None
    error:      Optional[str] = None

    def run(self, *args, **kwargs) -> Any:
        """Exécute la fonction de l'étape et met à jour le statut."""
        self.status = "running"
        logger.info(f"[PIPELINE] ▶ Étape '{self.name}' — démarrage")
        t0 = time.perf_counter()
        try:
            self.output    = self.fn(*args, **kwargs)
            self.status    = "success"
            self.duration_s = round(time.perf_counter() - t0, 4)
            logger.info(f"[PIPELINE] ✅ Étape '{self.name}' — succès ({self.duration_s:.2f}s)")
            return self.output
        except Exception as e:
            self.status     = "failed"
            self.duration_s = round(time.perf_counter() - t0, 4)
            self.error      = str(e)
            logger.error(
                f"[PIPELINE] ❌ Étape '{self.name}' — ÉCHEC ({self.duration_s:.2f}s)\n"
                f"{traceback.format_exc()}"
            )
            raise


# ─────────────────────────────────────────────
# Rapport de run
# ─────────────────────────────────────────────
@dataclass
class PipelineRunReport:
    """Rapport global d'exécution du pipeline."""
    run_id:         str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    started_at:     str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at:   str = ""
    total_duration: float = 0.0
    status:         str = "pending"
    steps:          dict = field(default_factory=dict)
    extract_meta:   dict = field(default_factory=dict)
    transform_meta: dict = field(default_factory=dict)
    load_meta:      dict = field(default_factory=dict)
    error:          Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "run_id":         self.run_id,
            "started_at":     self.started_at,
            "completed_at":   self.completed_at,
            "total_duration_s": self.total_duration,
            "status":         self.status,
            "steps_summary":  {
                name: {
                    "status":     s["status"],
                    "duration_s": s["duration_s"],
                    "error":      s.get("error"),
                }
                for name, s in self.steps.items()
            },
            "extract_meta":   self.extract_meta,
            "transform_meta": self.transform_meta,
            "load_meta":      self.load_meta,
            "error":          self.error,
        }


# ─────────────────────────────────────────────
# Orchestrateur principal
# ─────────────────────────────────────────────
class ETLPipeline:
    """
    Orchestrateur DAG du pipeline ETL E-commerce.

    Enchaîne :
      Extract ──► Transform ──► Load

    Usage :
        pipeline = ETLPipeline()
        report   = pipeline.run()

    Chaque étape est loggée, chronométrée et son rapport est collecté.
    En cas d'échec d'une étape, le pipeline s'arrête et logue l'erreur avec contexte.
    """

    def __init__(self, data_path: str = None, db_path: str = None, csv_path: str = None):
        self.data_path = data_path
        self.db_path   = db_path   or "data/warehouse/ecommerce.db"
        self.csv_path  = csv_path  or "data/processed/processed_data.csv"

    # ──────────────────────────────────────────
    # Point d'entrée
    # ──────────────────────────────────────────
    def run(self) -> PipelineRunReport:
        """
        Lance le pipeline ETL complet.

        Retourne un PipelineRunReport contenant :
          - Le statut global (success / failed)
          - Les métriques de chaque étape
          - Les temps d'exécution
        """
        report = PipelineRunReport()
        t_global = time.perf_counter()

        self._print_banner()
        logger.info(f"[PIPELINE] Run ID : {report.run_id}")

        # ── ÉTAPE 1 : EXTRACT ──────────────────
        step_extract = PipelineStep(name="Extract", fn=self._run_extract)
        try:
            df_raw, extract_meta = step_extract.run()
            report.extract_meta = extract_meta
        except Exception as e:
            return self._finalize_report(report, "failed", t_global, step_extract, str(e))

        report.steps["Extract"] = {
            "status":     step_extract.status,
            "duration_s": step_extract.duration_s,
            "rows":       extract_meta.get("quality", {}).get("total_rows"),
            "quality_score": extract_meta.get("quality", {}).get("quality_score"),
        }

        # ── ÉTAPE 2 : TRANSFORM ────────────────
        step_transform = PipelineStep(name="Transform", fn=self._run_transform)
        try:
            df_clean, transform_meta = step_transform.run(df_raw)
            report.transform_meta = transform_meta
        except Exception as e:
            return self._finalize_report(report, "failed", t_global, step_transform, str(e))

        report.steps["Transform"] = {
            "status":           step_transform.status,
            "duration_s":       step_transform.duration_s,
            "input_rows":       transform_meta.get("input_shape", [None])[0],
            "output_rows":      transform_meta.get("output_shape", [None])[0],
            "features_created": transform_meta.get("new_features_count"),
        }

        # ── ÉTAPE 3 : LOAD ─────────────────────
        combined_meta = {
            "run_id":       report.run_id,
            "extract":      extract_meta,
            "transform":    transform_meta,
        }
        step_load = PipelineStep(name="Load", fn=self._run_load)
        try:
            load_meta = step_load.run(df_clean, combined_meta)
            report.load_meta = load_meta
        except Exception as e:
            return self._finalize_report(report, "failed", t_global, step_load, str(e))

        report.steps["Load"] = {
            "status":     step_load.status,
            "duration_s": step_load.duration_s,
            "rows_in_db": load_meta.get("load", {}).get("sqlite", {}).get("rows_in_db"),
            "report_path": load_meta.get("report_path"),
        }

        return self._finalize_report(report, "success", t_global)

    # ──────────────────────────────────────────
    # Fonctions internes de chaque étape
    # ──────────────────────────────────────────
    def _run_extract(self):
        return extract(self.data_path)

    def _run_transform(self, df_raw):
        return transform(df_raw)

    def _run_load(self, df_clean, meta):
        return load(df_clean, meta, db_path=self.db_path, csv_path=self.csv_path)

    # ──────────────────────────────────────────
    # Finalisation du rapport
    # ──────────────────────────────────────────
    def _finalize_report(
        self,
        report: PipelineRunReport,
        status: str,
        t_global: float,
        failed_step: PipelineStep = None,
        error_msg: str = None,
    ) -> PipelineRunReport:
        report.status         = status
        report.total_duration = round(time.perf_counter() - t_global, 4)
        report.completed_at   = datetime.now().isoformat()
        report.error          = error_msg

        if failed_step:
            report.steps[failed_step.name] = {
                "status":     failed_step.status,
                "duration_s": failed_step.duration_s,
                "error":      failed_step.error,
            }

        self._print_summary(report)
        return report

    # ──────────────────────────────────────────
    # Affichage console
    # ──────────────────────────────────────────
    def _print_banner(self):
        banner = """
+------------------------------------------------------+
|         E-COMMERCE ML -- PIPELINE ETL                |
|         Extract  ->  Transform  ->  Load             |
+------------------------------------------------------+
"""
        print(banner)

    def _print_summary(self, report: PipelineRunReport):
        status_tag = "[OK]" if report.status == "success" else "[FAILED]"
        print(f"\n{'='*56}")
        print(f"  {status_tag} Pipeline {report.status.upper()} -- {report.total_duration:.2f}s")
        print(f"{'='*56}")
        for step_name, info in report.steps.items():
            step_tag = "[OK]" if info["status"] == "success" else "[ERR]"
            print(f"  {step_tag} {step_name:<12} {info['duration_s']:.2f}s")
        print(f"{'='*56}\n")
