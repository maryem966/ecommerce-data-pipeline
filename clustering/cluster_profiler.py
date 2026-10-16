"""
Clustering — Profiler de clusters
===================================
Génère des personas lisibles et des recommandations business
à partir des profils de clusters KMeans.

Objectif : transformer des statistiques techniques en insights actionnables
→ directement utilisable dans une présentation / entretien business.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import os
import json

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Nommage automatique des clusters
# ─────────────────────────────────────────────
def assign_cluster_personas(cluster_profiles: pd.DataFrame) -> dict:
    """
    Attribue automatiquement un nom de persona et une recommandation
    à chaque cluster en fonction de ses caractéristiques moyennes.

    Logique de nommage basée sur :
      - Frequency     : clients très actifs vs inactifs
      - MonetaryTotal : gros acheteurs vs petits acheteurs
      - SatisfactionScore : satisfaits vs insatisfaits
      - EngagementScore : engagés vs désengagés

    Retourne un dict {cluster_id: {"persona": ..., "description": ..., "action": ...}}
    """
    personas = {}

    # Normaliser pour comparer
    profiles = cluster_profiles.copy()
    available_cols = profiles.columns.tolist()

    def get_val(row, col, default=0):
        return row[col] if col in available_cols else default

    for cluster_id, row in profiles.iterrows():
        freq    = get_val(row, "Frequency")
        monetary = get_val(row, "MonetaryTotal")
        satisfaction = get_val(row, "SatisfactionScore", 3)
        engagement = get_val(row, "EngagementScore", 0)
        tenure  = get_val(row, "CustomerTenureDays", 0)
        return_r = get_val(row, "ReturnRatio", 0)

        # Seuils relatifs aux médianes du dataset
        freq_med    = float(profiles["Frequency"].median()) if "Frequency" in available_cols else 1
        monetary_med = float(profiles["MonetaryTotal"].median()) if "MonetaryTotal" in available_cols else 1

        is_high_freq     = freq > freq_med
        is_high_monetary = monetary > monetary_med
        is_satisfied     = satisfaction >= 3.5
        is_veteran       = tenure > 200
        is_returner      = return_r > 0.15

        # Assignation du persona
        if is_high_freq and is_high_monetary and is_satisfied:
            persona = "[Champion]"
            description = (
                "Clients tres actifs, gros acheteurs et satisfaits. "
                "Ce sont les meilleurs ambassadeurs de la marque."
            )
            action = "Fideliser avec un programme VIP, ambassadeurs de marque."

        elif is_high_freq and is_high_monetary and not is_satisfied:
            persona = "[Client Frustre Actif]"
            description = (
                "Achete beaucoup mais exprime une insatisfaction. "
                "Risque de churn eleve malgre le volume."
            )
            action = "Contacter le service client en priorite, offrir des avantages de fidelite."

        elif not is_high_freq and not is_high_monetary and not is_satisfied:
            persona = "[Client a Risque]"
            description = (
                "Peu actif, faibles achats et insatisfait. "
                "Fort risque de churn a court terme."
            )
            action = "Campagne de reactivation urgente, offre promotionnelle ciblee."

        elif is_veteran and not is_high_freq:
            persona = "[Client Dormant]"
            description = (
                "Anciennete elevee mais faible activite recente. "
                "Fidele par le passe, en phase de decrochage."
            )
            action = "Campagne de re-engagement, rappel des benefices du programme de fidelite."

        elif not is_veteran and is_high_freq:
            persona = "[Nouveau Client Prometteur]"
            description = (
                "Recemment arrive mais deja tres actif. "
                "Fort potentiel de valeur a long terme."
            )
            action = "Accueillir avec un programme d onboarding, cross-selling cible."

        elif is_returner:
            persona = "[Retourneur Frequent]"
            description = (
                "Taux de retour eleve -- potentiellement insatisfait "
                "de la qualite ou du service livraison."
            )
            action = "Analyser les motifs de retour, ameliorer l experience produit."

        else:
            persona = "[Client Ordinaire]"
            description = "Comportement moyen sans signal fort positif ou negatif."
            action = "Segmentation affinee recommandee, cross-sell sur produits populaires."

        cid_key = int(cluster_id) if str(cluster_id).isdigit() else str(cluster_id)
        personas[cid_key] = {
            "persona":     persona,
            "description": description,
            "action":      action,
            "stats": {
                "avg_frequency":     round(float(freq), 2),
                "avg_monetary":      round(float(monetary), 2),
                "avg_satisfaction":  round(float(satisfaction), 2),
                "avg_tenure_days":   round(float(tenure), 1),
                "avg_return_ratio":  round(float(return_r), 4),
            },
        }

        logger.info(f"[PROFILER] Cluster {cluster_id} → {persona}")

    return personas


# ─────────────────────────────────────────────
# Rapport complet des personas
# ─────────────────────────────────────────────
def generate_cluster_report(
    cluster_profiles: pd.DataFrame,
    cluster_distribution: dict,
    output_dir: str = "reports",
) -> str:
    """
    Génère un rapport JSON complet avec les personas,
    les statistiques et les recommandations business.

    Args:
        cluster_profiles     : DataFrame des moyennes par cluster
        cluster_distribution : {cluster_id: nombre de clients}
        output_dir           : dossier de sortie

    Returns:
        Chemin du fichier JSON généré
    """
    personas = assign_cluster_personas(cluster_profiles)

    # Enrichir avec la distribution
    total_customers = sum(cluster_distribution.values())
    report = {
        "total_clusters": len(personas),
        "total_customers_segmented": total_customers,
        "clusters": {},
    }

    for cluster_id, persona_info in personas.items():
        count = cluster_distribution.get(cluster_id, 0)
        pct   = round(count / total_customers * 100, 1) if total_customers > 0 else 0
        report["clusters"][str(cluster_id)] = {
            **persona_info,
            "customer_count": count,
            "pct_of_total":   pct,
        }

    # Sauvegarde JSON
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cluster_personas_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"[PROFILER] Rapport personas sauvegardé → {path}")

    # Affichage console
    _print_personas(report)

    return path


# ─────────────────────────────────────────────
# Affichage console des personas
# ─────────────────────────────────────────────
def _print_personas(report: dict):
    print(f"\n{'='*60}")
    print(f"  SEGMENTATION CLIENTS -- {report['total_clusters']} Clusters")
    print(f"  {report['total_customers_segmented']} clients segmentes")
    print(f"{'='*60}")
    for cid, info in report["clusters"].items():
        print(f"\n  Cluster {cid} -- {info['persona']}")
        print(f"  {'-'*50}")
        print(f"  [Clients] {info['customer_count']} clients ({info['pct_of_total']}%)")
        print(f"  [Profil]  {info['description']}")
        print(f"  [Action]  {info['action']}")
        s = info["stats"]
        print(
            f"  [Stats]   Freq={s['avg_frequency']} | "
            f"Monetary={s['avg_monetary']} EUR | "
            f"Satisfaction={s['avg_satisfaction']}/5 | "
            f"Anciennete={s['avg_tenure_days']}j"
        )
    print(f"\n{'='*60}\n")
