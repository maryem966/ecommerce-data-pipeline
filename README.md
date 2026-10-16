# 🛒 E-Commerce ML — Data Engineering Showcase

> **Prédiction du Churn client + Segmentation KMeans**  
> Pipeline ETL industrialisée · Clustering non supervisé · Data Warehouse SQLite · Monitoring · Tests unitaires

---

## 🏗️ Architecture

```
E-commerceML/
│
├── etl/                          ← Pipeline ETL (Extract → Transform → Load)
│   ├── extract.py                   Chargement + validation schéma + rapport qualité
│   ├── transform.py                 Nettoyage · Feature Eng · Leakage · Corrélations
│   ├── load.py                      SQLite Data Warehouse + CSV + requêtes SQL
│   └── pipeline.py                  Orchestrateur DAG (PipelineStep + rapport de run)
│
├── clustering/                   ← Segmentation non supervisée
│   ├── kmeans_segmentation.py       KMeans · Elbow · Silhouette · PCA 2D · Heatmap
│   └── cluster_profiler.py          Personas business + recommandations actionnables
│
├── monitoring/                   ← Observabilité de la pipeline
│   └── pipeline_logger.py           Métriques par étape · Dashboard ASCII · Export JSON
│
├── src/                          ← Code ML supervisé (inchangé)
│   ├── preprocessing.py             Pipeline Scikit-Learn (Imputer · Scaler · OHE · PCA)
│   ├── train_model.py               GridSearchCV sur LR + RandomForest
│   ├── predict.py                   Classe ChurnPredictor (inférence)
│   └── utils.py                     Logging · load/save data & models
│
├── tests/                        ← Tests unitaires pytest
│   ├── test_extract.py              Tests validation schéma + qualité données
│   ├── test_transform.py            Tests chaque transformation individuellement
│   └── test_clustering.py           Tests KMeans + personas
│
├── app/                          ← API Flask de démonstration
│   ├── routes.py                    Endpoints GET/POST
│   └── templates/index.html         Interface web
│
├── data/
│   ├── raw/                         Données sources brutes
│   ├── processed/                   Données transformées (CSV)
│   ├── warehouse/                   Data Warehouse (SQLite)
│   └── train_test/                  Splits 80/20 pour l'entraînement ML
│
├── models/                       ← Modèles sérialisés (.pkl)
├── reports/                      ← Graphiques + rapports JSON + logs
│
├── run_etl.py                    ← Point d'entrée ETL + Clustering
└── run.py                        ← Point d'entrée API Flask
```

---

## 🔄 Pipeline ETL — Flux de données

```
┌─────────────────────────────────────────────────────────────────────┐
│                          PIPELINE ETL                               │
│                                                                     │
│  ┌──────────┐    ┌─────────────────────────┐    ┌───────────────┐  │
│  │ EXTRACT  │───▶│       TRANSFORM         │───▶│     LOAD      │  │
│  │          │    │                         │    │               │  │
│  │ • CSV/   │    │ 1. Outlier correction   │    │ • SQLite DB   │  │
│  │   XLSX   │    │ 2. Feature Engineering  │    │ • CSV export  │  │
│  │ • Schema │    │    - Date parsing        │    │ • SQL valid.  │  │
│  │   valid. │    │    - IP features         │    │ • JSON report │  │
│  │ • Quality│    │    - AvgBasketValue      │    │               │  │
│  │   report │    │    - EngagementScore     │    │               │  │
│  │          │    │    - TenureFreqRatio     │    │               │  │
│  │ Output:  │    │ 3. Leakage removal      │    │ Output:       │  │
│  │ Raw DF   │    │ 4. Corr. filtering      │    │ ecommerce.db  │  │
│  │ + Metadata    │                         │    │ + report.json │  │
│  └──────────┘    │ Output: Clean DF        │    └───────────────┘  │
│                  │ + Metadata              │                        │
│                  └─────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       CLUSTERING KMeans                             │
│                                                                     │
│  Features RFM ──▶ StandardScaler ──▶ Elbow + Silhouette ──▶ KMeans │
│                                                                     │
│  Sorties :                                                          │
│    • Graphique Elbow/Silhouette (sélection K optimal)              │
│    • Visualisation PCA 2D                                           │
│    • Heatmap des profils de clusters                                │
│    • Personas business (Champion, Client à risque, ...)            │
│    • Labels sauvegardés dans SQLite                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Installation

```bash
# 1. Créer un environnement virtuel
python -m venv venv

# 2. Activer l'environnement
# Windows :
venv\Scripts\activate
# Mac/Linux :
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Guide d'utilisation

### Pipeline ETL + Clustering (nouveau)

```bash
# Pipeline complet (ETL + Clustering + Monitoring)
python run_etl.py

# ETL uniquement (sans clustering)
python run_etl.py --etl-only

# Clustering sur données déjà transformées
python run_etl.py --cluster-only

# Fichier source personnalisé
python run_etl.py --data data/raw/mon_fichier.csv
```

### Pipeline ML supervisé (original)

```bash
# Étape 1 : Prétraitement
python -m src.preprocessing

# Étape 2 : Entraînement
python -m src.train_model

# Étape 3 : API Flask
python run.py
# → http://127.0.0.1:5000/
```

### Tests unitaires

```bash
# Tous les tests
pytest tests/ -v

# Par module
pytest tests/test_extract.py -v
pytest tests/test_transform.py -v
pytest tests/test_clustering.py -v

# Avec rapport de couverture
pytest tests/ -v --tb=short
```

---

## 📊 Sorties produites

| Fichier | Description |
|---|---|
| `data/warehouse/ecommerce.db` | Data Warehouse SQLite (tables: `customers_transformed`, `customers_clustered`, `pipeline_runs`) |
| `data/processed/processed_data.csv` | Dataset nettoyé et enrichi |
| `reports/pipeline_run_YYYYMMDD.json` | Rapport complet du run ETL (métriques, volumes, qualité) |
| `reports/extract_report_*.json` | Rapport de qualité des données brutes |
| `reports/cluster_personas_report.json` | Personas et recommandations business |
| `reports/clustering_elbow_silhouette.png` | Graphique de sélection du K optimal |
| `reports/clustering_pca_2d.png` | Visualisation PCA 2D des clusters |
| `reports/clustering_profiles_heatmap.png` | Heatmap des profils de clusters |
| `reports/monitoring_*.json` | Métriques de performance de la pipeline |
| `models/kmeans_model.pkl` | Modèle KMeans sérialisé |
| `models/best_model.pkl` | Meilleur modèle supervisé (LR ou RF) |

---

## 🎯 Points forts Data Engineering

### 1. Pipeline ETL structurée
- **Séparation stricte** Extract / Transform / Load
- Chaque étape est une **fonction pure et testable**
- **Rapport de qualité** automatique à l'Extract (% nulls, doublons, score)
- **Orchestrateur DAG** dans `etl/pipeline.py` (inspiré d'Apache Airflow)

### 2. Clustering non supervisé
- **Sélection automatique de K** via méthode Elbow + Silhouette Score
- Visualisation PCA 2D pour inspection visuelle des clusters
- **Personas business** générés automatiquement
- Labels persistés dans le Data Warehouse

### 3. Data Warehouse SQLite
- Données transformées **requêtables en SQL**
- Table d'**audit** (`pipeline_runs`) : traçabilité de chaque exécution
- Requêtes de validation post-chargement

### 4. Monitoring de pipeline
- **Temps d'exécution** de chaque étape
- **Volume de données** à chaque step (lignes en entrée / sortie)
- **Score de qualité** des données (0–100)
- Dashboard ASCII + export JSON

### 5. Tests unitaires
- **Tests par fonction** sur chaque transformation
- **Fixtures pytest** réutilisables
- Vérification de l'absence de data leakage en output
- Tests de robustesse (valeurs manquantes, cas limites)

---

## 🗃️ Data Warehouse — Requêtes SQL exemples

```sql
-- Distribution du churn
SELECT Churn, COUNT(*) as n FROM customers_transformed GROUP BY Churn;

-- Profil moyen par cluster
SELECT Cluster,
       ROUND(AVG(Frequency), 2)        as avg_freq,
       ROUND(AVG(MonetaryTotal), 2)     as avg_monetary,
       ROUND(AVG(SatisfactionScore), 2) as avg_satisfaction,
       COUNT(*) as n
FROM customers_clustered
GROUP BY Cluster
ORDER BY avg_monetary DESC;

-- Journal des runs ETL
SELECT run_id, rows_loaded, loaded_at FROM pipeline_runs ORDER BY loaded_at DESC;

-- Clients à haut risque par cluster
SELECT Cluster, COUNT(*) as churners
FROM customers_clustered
WHERE Churn = 1
GROUP BY Cluster
ORDER BY churners DESC;
```

---

## 📦 Stack technique

| Couche | Technologie |
|---|---|
| ETL & Data Prep | Python · Pandas · NumPy |
| Clustering | Scikit-Learn (KMeans, PCA, StandardScaler, Silhouette) |
| ML Supervisé | Scikit-Learn (LogisticRegression, RandomForest, GridSearchCV, SMOTE) |
| Data Warehouse | SQLite3 |
| Visualisation | Matplotlib · Seaborn |
| API | Flask |
| Tests | Pytest |
| Sérialisation | Joblib |
