"""
Monitoring — Pipeline Logger
=============================
Collecte, agrège et affiche les métriques de performance de la pipeline ETL.

Métriques suivies :
  - Temps d'exécution de chaque étape (Extract, Transform, Load, Clustering)
  - Volume de données à chaque step (lignes, colonnes)
  - Score de qualité des données (% nulls, doublons)
  - Résumé visuel du pipeline run

Affiche également un dashboard ASCII dans le terminal lors de l'exécution.
"""

import logging
import os
import json
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Collecteur de métriques
# ─────────────────────────────────────────────
class PipelineMetricsCollector:
    """
    Collecte les métriques de performance du pipeline ETL en temps réel.

    Usage :
        collector = PipelineMetricsCollector()
        collector.record_step("Extract", duration=1.2, rows_in=5000, rows_out=5000)
        collector.record_step("Transform", duration=2.1, rows_in=5000, rows_out=4980)
        collector.print_dashboard()
    """

    def __init__(self):
        self.run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.started_at = datetime.now().isoformat()
        self.steps: list[dict] = []
        self.data_quality: dict = {}
        self.cluster_metrics: dict = {}

    # ──────────────────────────────────────────
    # Enregistrement d'une étape
    # ──────────────────────────────────────────
    def record_step(
        self,
        step_name: str,
        duration_s: float,
        rows_in: Optional[int] = None,
        rows_out: Optional[int] = None,
        cols_in: Optional[int] = None,
        cols_out: Optional[int] = None,
        extra: Optional[dict] = None,
    ):
        """Enregistre les métriques d'une étape du pipeline."""
        entry = {
            "step":       step_name,
            "duration_s": round(duration_s, 4),
            "rows_in":    rows_in,
            "rows_out":   rows_out,
            "cols_in":    cols_in,
            "cols_out":   cols_out,
            "timestamp":  datetime.now().isoformat(),
        }
        if extra:
            entry.update(extra)
        self.steps.append(entry)
        logger.info(
            f"[MONITORING] ⏱  {step_name:<12} | "
            f"{duration_s:.2f}s | "
            f"Lignes: {rows_in} → {rows_out} | "
            f"Colonnes: {cols_in} → {cols_out}"
        )

    # ──────────────────────────────────────────
    # Enregistrement qualité données
    # ──────────────────────────────────────────
    def record_data_quality(self, quality_report: dict):
        """Enregistre les métriques de qualité des données brutes."""
        self.data_quality = {
            "quality_score":  quality_report.get("quality_score"),
            "total_rows":     quality_report.get("total_rows"),
            "duplicate_pct":  quality_report.get("duplicate_pct"),
            "avg_null_pct":   quality_report.get("avg_null_pct"),
        }
        logger.info(
            f"[MONITORING] 📊 Qualité données — "
            f"Score: {self.data_quality['quality_score']}/100 | "
            f"Nulls: {self.data_quality['avg_null_pct']}% | "
            f"Doublons: {self.data_quality['duplicate_pct']}%"
        )

    # ──────────────────────────────────────────
    # Enregistrement clustering
    # ──────────────────────────────────────────
    def record_clustering(self, clustering_report: dict):
        """Enregistre les métriques du clustering."""
        self.cluster_metrics = {
            "k_optimal":        clustering_report.get("k_optimal"),
            "silhouette_score": clustering_report.get("final_silhouette"),
            "duration_s":       clustering_report.get("clustering_time_s"),
            "distribution":     clustering_report.get("cluster_distribution"),
        }
        logger.info(
            f"[MONITORING] 🔵 Clustering — "
            f"K={self.cluster_metrics['k_optimal']} | "
            f"Silhouette={self.cluster_metrics['silhouette_score']}"
        )

    # ──────────────────────────────────────────
    # Dashboard ASCII
    # ──────────────────────────────────────────
    def print_dashboard(self):
        """Affiche un dashboard de métriques dans le terminal."""
        total_duration = sum(s["duration_s"] for s in self.steps)

        print(f"\n{'='*62}")
        print(f"  PIPELINE MONITORING DASHBOARD -- Run {self.run_id}")
        print(f"{'='*62}")

        # Étapes
        print(f"\n  {'Etape':<14} {'Duree':>8}  {'Lignes In':>10}  {'Lignes Out':>10}")
        print(f"  {'-'*55}")
        for step in self.steps:
            bar_len = min(20, max(1, int(step["duration_s"] / max(total_duration, 0.01) * 20)))
            bar = "#" * bar_len + "." * (20 - bar_len)
            r_in  = str(step["rows_in"])  if step["rows_in"]  is not None else "-"
            r_out = str(step["rows_out"]) if step["rows_out"] is not None else "-"
            print(
                f"  {step['step']:<14} "
                f"{step['duration_s']:>6.2f}s  "
                f"{r_in:>10}  {r_out:>10}  [{bar}]"
            )

        print(f"  {'-'*55}")
        print(f"  {'TOTAL':<14} {total_duration:>6.2f}s")

        # Qualité
        if self.data_quality:
            dq = self.data_quality
            score = dq.get("quality_score", 0)
            bar_len = min(20, int(score / 100 * 20))
            quality_bar = "#" * bar_len + "." * (20 - bar_len)
            print(f"\n  Qualite des donnees : [{quality_bar}] {score}/100")
            print(f"    Nulls moyen : {dq.get('avg_null_pct', '?')}%  |  "
                  f"Doublons : {dq.get('duplicate_pct', '?')}%  |  "
                  f"Lignes : {dq.get('total_rows', '?')}")

        # Clustering
        if self.cluster_metrics:
            cm = self.cluster_metrics
            print(f"\n  Clustering : K={cm['k_optimal']}  |  "
                  f"Silhouette={cm['silhouette_score']}  |  "
                  f"Duree={cm['duration_s']}s")
            if cm.get("distribution"):
                dist_str = "  ".join(
                    f"C{k}:{v}" for k, v in sorted(cm["distribution"].items())
                )
                print(f"    Distribution : {dist_str}")

        print(f"\n{'='*62}\n")

    # ──────────────────────────────────────────
    # Export JSON
    # ──────────────────────────────────────────
    def export_metrics(self, output_dir: str = "reports") -> str:
        """Exporte toutes les métriques collectées dans un fichier JSON."""
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "run_id":          self.run_id,
            "started_at":      self.started_at,
            "exported_at":     datetime.now().isoformat(),
            "steps":           self.steps,
            "data_quality":    self.data_quality,
            "cluster_metrics": self.cluster_metrics,
            "total_duration_s": round(sum(s["duration_s"] for s in self.steps), 4),
        }
        path = os.path.join(output_dir, f"monitoring_{self.run_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"[MONITORING] Métriques exportées → {path}")
        return path


# ─────────────────────────────────────────────
# Context manager pour chronométrer une étape
# ─────────────────────────────────────────────
class StepTimer:
    """
    Context manager pour mesurer le temps d'exécution d'un bloc de code.

    Usage :
        with StepTimer("Mon étape") as t:
            do_something()
        print(t.duration_s)
    """

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.duration_s = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        logger.info(f"[TIMER] ▶ {self.step_name}")
        return self

    def __exit__(self, *args):
        self.duration_s = round(time.perf_counter() - self._start, 4)
        logger.info(f"[TIMER] ■ {self.step_name} — {self.duration_s:.2f}s")
