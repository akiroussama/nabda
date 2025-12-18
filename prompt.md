# 🧠 MASTER PROMPT — Jira AI Co-pilot MVP

> **Destinataire** : Claude Opus 4.5 (Vibe Coding Mode)
> **Auteur** : Product Owner / CTO
> **Version** : 1.0
> **Durée projet** : 12 semaines (6 sprints × 2 semaines)

---

## PARTIE 1 : CONTEXTE GLOBAL

### 1.1 Identité du projet

**Nom** : Jira AI Co-pilot
**Nature** : Outil d'intelligence décisionnelle personnel branché sur Jira
**Philosophie** : Ne pas refaire Jira, mais ajouter une couche d'IA prédictive par-dessus

### 1.2 Situation initiale

Je suis un développeur/manager qui gère un projet IT sur Jira Cloud. Mon instance contient :
- **Volume** : Centaines de tickets actifs
- **Équipe** : Dizaines de développeurs
- **Méthodologie** : Scrum/Kanban avec sprints, story points, worklogs

**Mes frustrations actuelles** :
- Je découvre les retards trop tard, quand le sprint est déjà compromis
- La charge de travail entre développeurs est déséquilibrée — certains sont surchargés, d'autres sous-utilisés
- Les estimations de tickets sont souvent fausses — je n'ai aucune aide data-driven
- Organiser les releases et prioriser le backlog me prend un temps fou
- Je n'ai pas de visibilité sur les risques de burn-out dans l'équipe

### 1.3 Vision du produit

Un assistant IA local qui :
1. **Anticipe** les problèmes avant qu'ils n'arrivent (retards, surcharge, blocages)
2. **Suggère** des actions correctives (réassignation, re-priorisation)
3. **Prédit** les délais de façon réaliste basée sur l'historique
4. **Optimise** l'organisation des releases et du backlog
5. **Dialogue** avec moi en langage naturel pour explorer les données

### 1.4 Contraintes techniques imposées

| Contrainte | Valeur | Raison |
|------------|--------|--------|
| **Langage** | Python 100% | Écosystème ML, préférence personnelle |
| **Infrastructure** | Local uniquement | MVP validation, pas de coûts cloud |
| **Base de données** | DuckDB | Performance analytique, zéro config |
| **LLM** | Ollama (Llama 3.1 8B ou Mistral 7B) | Gratuit, local, privacy |
| **API Jira** | REST API + sync incrémental | Robustesse, contrôle total |
| **Interface** | CLI + Streamlit basique | Validation rapide, on s'en fout du design |
| **Qualité graphique** | Minimale acceptable | Focus sur la valeur fonctionnelle |

### 1.5 Stack technique validée

```
INGESTION DONNÉES
├── jira-python >= 3.10.0          # Client API Jira officiel
├── tenacity >= 8.0.0              # Retry logic avec backoff
└── apscheduler >= 3.11.0          # Orchestration syncs

STOCKAGE & ANALYTICS
├── duckdb >= 1.0.0                # Base analytique columnaire
├── pandas >= 2.0.0                # Manipulation dataframes
└── pyarrow >= 15.0.0              # Interop performante

MACHINE LEARNING
├── scikit-learn >= 1.4.0          # Modèles baseline
├── lightgbm >= 4.0.0              # Gradient boosting (features catégorielles)
└── optuna >= 3.6.0                # Hyperparameter tuning (optionnel)

LLM & NLP
├── ollama >= 0.3.0                # Client Ollama local
├── tiktoken >= 0.7.0              # Token counting
└── sentence-transformers >= 3.0.0 # Embeddings (optionnel)

INTERFACE
├── streamlit >= 1.35.0            # Dashboard rapide
├── rich >= 13.0.0                 # CLI améliorée
└── typer >= 0.12.0                # CLI structurée

UTILS
├── pydantic-settings >= 2.0.0     # Config typée
├── python-dotenv >= 1.0.0         # Variables environnement
└── loguru >= 0.7.0                # Logging simplifié
```

### 1.6 Structure projet cible

```
jira-copilot/
├── .env.example                   # Template variables environnement
├── .gitignore
├── Makefile                       # Commandes principales (sync, train, predict, dashboard)
├── pyproject.toml                 # Dépendances + config
├── README.md
│
├── config/
│   ├── settings.py               # Pydantic settings (chargement .env)
│   ├── jira_config.yaml          # Projets, custom fields mapping
│   └── model_config.yaml         # Hyperparamètres ML
│
├── data/
│   ├── jira.duckdb               # Base principale
│   ├── raw/                      # JSON bruts (backup)
│   └── exports/                  # CSV/reports exportés
│
├── models/
│   ├── ticket_estimator.pkl      # Modèle estimation délais
│   ├── sprint_risk.pkl           # Modèle risque sprint
│   └── metadata.json             # Versions, métriques, dates
│
├── src/
│   ├── __init__.py
│   │
│   ├── jira_client/
│   │   ├── __init__.py
│   │   ├── auth.py               # Authentification Jira
│   │   ├── fetcher.py            # Récupération données
│   │   ├── rate_limiter.py       # Gestion rate limiting
│   │   └── sync.py               # Orchestration sync incrémental
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py             # Définition tables DuckDB
│   │   ├── loader.py             # Chargement données brutes → DuckDB
│   │   └── queries.py            # Queries SQL réutilisables
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── ticket_features.py    # Features pour estimation tickets
│   │   ├── developer_features.py # Features charge développeur
│   │   ├── sprint_features.py    # Features santé sprint
│   │   └── pipeline.py           # Pipeline feature engineering
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ticket_estimator.py   # Prédiction durée tickets
│   │   ├── sprint_risk.py        # Score risque retard sprint
│   │   ├── workload_scorer.py    # Score charge développeur
│   │   └── trainer.py            # Training + évaluation
│   │
│   ├── intelligence/
│   │   ├── __init__.py
│   │   ├── llm_client.py         # Interface Ollama
│   │   ├── prompts.py            # Templates prompts
│   │   ├── analyst.py            # Analyse LLM des données
│   │   └── recommender.py        # Suggestions actions
│   │
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── load_balancer.py      # Suggestions réassignation
│   │   ├── release_planner.py    # Génération plans release
│   │   └── alert_generator.py    # Génération alertes
│   │
│   └── interface/
│       ├── __init__.py
│       ├── cli.py                # Interface ligne de commande
│       └── dashboard.py          # Streamlit app
│
├── tests/
│   ├── conftest.py               # Fixtures pytest
│   ├── test_jira_client/
│   ├── test_features/
│   ├── test_models/
│   └── test_intelligence/
│
└── notebooks/
    ├── 01_exploration.ipynb      # Exploration données initiale
    ├── 02_feature_engineering.ipynb
    └── 03_model_experiments.ipynb
```

### 1.7 Données Jira disponibles

**Entités principales à ingérer** :
- `Issues` : key, summary, description, type, status, priority, assignee, reporter, created, updated, resolved, story_points, components, labels, sprint, epic
- `Changelog` : timestamps de chaque transition de status (crucial pour cycle time)
- `Worklogs` : temps passé par développeur par ticket (attention RGPD)
- `Sprints` : name, state, startDate, endDate, goal
- `Users` : accountId, displayName, emailAddress (pseudonymiser)
- `Projects` : key, name, projectTypeKey

**Métriques dérivées à calculer** :
- `lead_time` : created → resolved
- `cycle_time` : first "In Progress" → resolved
- `time_in_status` : durée dans chaque status
- `velocity` : story points completed par sprint
- `scope_creep` : points ajoutés mid-sprint / points committed
- `completion_rate` : tickets done / tickets committed

### 1.8 Considérations RGPD

**Règle d'or pour le MVP** : Privilégier les métriques d'équipe agrégées, pas individuelles.

Si données individuelles nécessaires :
- Pseudonymiser les identifiants développeurs (hash salé)
- Ne jamais afficher de noms réels dans les logs
- Prévoir un flag `--anonymize` pour tous les exports

---

## PARTIE 2 : OBJECTIF GLOBAL

### 2.1 Mission

Développer un MVP fonctionnel en 12 semaines qui prouve la valeur d'un assistant IA pour la gestion de projet Jira, en validant 4 hypothèses business :

| # | Hypothèse | Métrique de succès |
|---|-----------|-------------------|
| H1 | L'IA peut prédire la durée des tickets mieux que les estimations humaines | MAE < estimation humaine de 15%+ |
| H2 | L'IA peut détecter les sprints à risque avant mi-sprint | Précision alertes > 70% |
| H3 | L'IA peut identifier les déséquilibres de charge dans l'équipe | Détection écarts > 30% de la moyenne |
| H4 | L'IA peut générer des plans de release cohérents | Temps de planification réduit de 50% |

### 2.2 Définition du "Done" MVP

Le MVP est validé quand je peux :

1. **Synchroniser** mon projet Jira en une commande (`make sync`)
2. **Visualiser** un dashboard avec les métriques clés du sprint en cours
3. **Recevoir** une prédiction de durée pour tout nouveau ticket
4. **Voir** un score de risque pour le sprint actif avec les raisons
5. **Consulter** la charge de travail par développeur (agrégée ou pseudonymisée)
6. **Dialoguer** avec le LLM pour poser des questions sur mes données
7. **Générer** une proposition de plan de release basée sur le backlog

### 2.3 Ce qui est OUT OF SCOPE pour le MVP

- Interface web production-ready (Streamlit basique suffit)
- Multi-tenant / multi-utilisateurs
- Déploiement cloud
- Intégration bidirectionnelle (écriture dans Jira)
- Notifications automatiques (Slack, email)
- Authentification utilisateur
- Tests de charge / performance
- Documentation utilisateur complète

---

## PARTIE 3 : ROADMAP DÉTAILLÉE — 6 SPRINTS

---

### 🏃 SPRINT 1 : Fondations & Pipeline de données
**Semaines 1-2**

#### Objectif Sprint
Établir une connexion fiable avec Jira et construire le pipeline d'ingestion de données complet. À la fin de ce sprint, toutes les données nécessaires sont dans DuckDB et requêtables.

#### User Stories

**US1.1 — Configuration projet**
> En tant que développeur, je veux initialiser le projet avec toute la structure et les dépendances pour pouvoir commencer à coder immédiatement.

Critères d'acceptation :
- [ ] `pyproject.toml` avec toutes les dépendances listées dans la stack
- [ ] Structure de dossiers conforme au template défini
- [ ] `.env.example` documenté avec toutes les variables nécessaires
- [ ] `Makefile` avec commandes : `install`, `test`, `lint`
- [ ] `.gitignore` approprié (inclut `.env`, `*.duckdb`, `models/*.pkl`)
- [ ] README avec instructions de setup

**US1.2 — Authentification Jira**
> En tant que développeur, je veux me connecter à mon instance Jira Cloud de façon sécurisée.

Critères d'acceptation :
- [ ] Module `auth.py` avec classe `JiraAuthenticator`
- [ ] Support authentification Basic Auth (email + API token)
- [ ] Validation de la connexion au démarrage (test endpoint `/myself`)
- [ ] Gestion propre des erreurs d'authentification
- [ ] Configuration via `.env` : `JIRA_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`

**US1.3 — Récupération des Issues**
> En tant que développeur, je veux récupérer tous les tickets d'un projet avec leur historique complet.

Critères d'acceptation :
- [ ] Module `fetcher.py` avec classe `JiraFetcher`
- [ ] Méthode `fetch_issues(project_key, jql_filter=None)` avec pagination automatique
- [ ] Récupération des champs : key, summary, description, type, status, priority, assignee, reporter, created, updated, resolutiondate, story_points (custom field), components, labels, sprint, epic_link
- [ ] Expansion du changelog pour chaque issue
- [ ] Gestion de la pagination (maxResults=100, startAt incrémenté)
- [ ] Sauvegarde JSON brut dans `data/raw/issues_{timestamp}.json`

**US1.4 — Rate Limiting intelligent**
> En tant que développeur, je veux que le système respecte les limites de l'API Jira sans intervention manuelle.

Critères d'acceptation :
- [ ] Module `rate_limiter.py` avec décorateur `@rate_limited`
- [ ] Détection du header `X-RateLimit-Remaining`
- [ ] Exponential backoff avec jitter sur erreur 429
- [ ] Retry automatique (max 3 tentatives) via `tenacity`
- [ ] Logging des événements de rate limiting

**US1.5 — Récupération données complémentaires**
> En tant que développeur, je veux récupérer les sprints, worklogs et utilisateurs.

Critères d'acceptation :
- [ ] Méthode `fetch_sprints(board_id)` — tous les sprints du board
- [ ] Méthode `fetch_worklogs(since_timestamp)` — worklogs mis à jour depuis date
- [ ] Méthode `fetch_users(project_key)` — utilisateurs assignables au projet
- [ ] Méthode `fetch_boards(project_key)` — boards associés au projet
- [ ] Sauvegarde JSON brut séparée pour chaque entité

**US1.6 — Schéma DuckDB**
> En tant que développeur, je veux un schéma de base de données optimisé pour l'analytique.

Critères d'acceptation :
- [ ] Module `schema.py` avec fonction `initialize_database(db_path)`
- [ ] Tables : `issues`, `issue_changelog`, `sprints`, `worklogs`, `users`, `sync_metadata`
- [ ] Index sur : `issues.key`, `issues.assignee_id`, `issues.sprint_id`, `issue_changelog.issue_key`, `worklogs.issue_key`
- [ ] Table `sync_metadata` : dernière sync par entité, nombre de records
- [ ] Types appropriés : TIMESTAMP pour dates, INTEGER pour IDs, VARCHAR pour texte

**US1.7 — Chargement données**
> En tant que développeur, je veux charger les données JSON brutes dans DuckDB avec transformation.

Critères d'acceptation :
- [ ] Module `loader.py` avec classe `DataLoader`
- [ ] Méthode `load_issues(json_path)` : parsing + insertion/upsert
- [ ] Méthode `load_changelog(issues_data)` : extraction changelog → table dédiée
- [ ] Méthode `load_sprints(json_path)` 
- [ ] Méthode `load_worklogs(json_path)`
- [ ] Méthode `load_users(json_path)` avec pseudonymisation optionnelle
- [ ] Gestion des custom fields (mapping configurable dans `jira_config.yaml`)
- [ ] Idempotence : relancer le load ne crée pas de doublons

**US1.8 — Sync incrémental orchestré**
> En tant que développeur, je veux synchroniser uniquement les données modifiées depuis la dernière sync.

Critères d'acceptation :
- [ ] Module `sync.py` avec classe `JiraSyncOrchestrator`
- [ ] Méthode `full_sync()` : import initial complet
- [ ] Méthode `incremental_sync()` : JQL `updated >= "{last_sync}"`
- [ ] Mise à jour automatique de `sync_metadata` après chaque sync
- [ ] Commande CLI : `python -m src.jira_client.sync --mode [full|incremental]`
- [ ] Ajout au Makefile : `make sync` (incrémental par défaut)

**US1.9 — Queries analytiques de base**
> En tant que développeur, je veux des queries SQL réutilisables pour les analyses courantes.

Critères d'acceptation :
- [ ] Module `queries.py` avec fonctions retournant des DataFrames
- [ ] `get_issues_with_metrics(project_key)` : issues + lead_time + cycle_time calculés
- [ ] `get_sprint_summary(sprint_id)` : committed, completed, added, removed
- [ ] `get_developer_workload(days=30)` : issues assignées, completed, en cours par dev
- [ ] `get_velocity_history(n_sprints=10)` : velocity par sprint

#### Livrables Sprint 1
- [ ] Pipeline de sync fonctionnel (`make sync`)
- [ ] Base DuckDB peuplée avec données réelles
- [ ] Notebook `01_exploration.ipynb` validant l'accès aux données
- [ ] Tests unitaires pour `auth.py`, `fetcher.py`, `loader.py`

#### Orientations techniques Sprint 1

**Authentification** :
```python
# Pattern recommandé pour jira-python
from jira import JIRA
jira = JIRA(server=url, basic_auth=(email, token))
```

**Pagination robuste** :
```python
# Boucle until exhaustion
while True:
    results = jira.search_issues(jql, startAt=start, maxResults=100)
    all_issues.extend(results)
    if len(results) < 100:
        break
    start += 100
```

**DuckDB connexion** :
```python
import duckdb
conn = duckdb.connect('data/jira.duckdb')
# Pandas interop natif
df = conn.sql("SELECT * FROM issues").df()
```

**Custom fields Jira** :
Les story points sont souvent dans un custom field (`customfield_10016`). Prévoir un mapping configurable :
```yaml
# config/jira_config.yaml
custom_fields:
  story_points: customfield_10016
  epic_link: customfield_10014
  sprint: customfield_10020
```

---

### 🏃 SPRINT 2 : Feature Engineering & Premier Modèle
**Semaines 3-4**

#### Objectif Sprint
Construire le pipeline de feature engineering et entraîner le premier modèle prédictif (estimation durée tickets). Valider l'hypothèse H1.

#### User Stories

**US2.1 — Features tickets**
> En tant que data scientist, je veux des features pertinentes pour prédire la durée d'un ticket.

Critères d'acceptation :
- [ ] Module `ticket_features.py` avec classe `TicketFeatureExtractor`
- [ ] Features numériques :
  - `story_points` : points estimés (si disponible)
  - `description_length` : nombre de caractères description
  - `num_components` : nombre de composants tagués
  - `num_labels` : nombre de labels
  - `num_subtasks` : nombre de sous-tâches
  - `num_links` : nombre de liens (blocks, is blocked by)
  - `has_attachments` : booléen
- [ ] Features catégorielles :
  - `issue_type` : Bug, Story, Task, etc.
  - `priority` : Highest, High, Medium, Low, Lowest
  - `component_primary` : premier composant (ou "None")
- [ ] Features temporelles :
  - `created_day_of_week` : 0-6
  - `created_hour` : 0-23
  - `sprint_day_created` : jour dans le sprint (1-14)
- [ ] Features développeur (agrégées/pseudonymisées) :
  - `assignee_avg_cycle_time_30d` : moyenne cycle time assignee sur 30j
  - `assignee_completion_rate_30d` : taux de complétion sur 30j
  - `assignee_current_wip` : tickets en cours actuellement

**US2.2 — Features développeur**
> En tant que data scientist, je veux des features sur la charge et performance des développeurs.

Critères d'acceptation :
- [ ] Module `developer_features.py` avec classe `DeveloperFeatureExtractor`
- [ ] Métriques par développeur (pseudonymisé) :
  - `total_story_points_30d` : points complétés sur 30 jours
  - `avg_cycle_time_30d` : moyenne cycle time
  - `wip_count` : tickets en cours (status "In Progress" ou équivalent)
  - `worklog_hours_7d` : heures loguées sur 7 jours
  - `tickets_completed_7d` : tickets résolus sur 7 jours
  - `overdue_tickets` : tickets en retard (si due date)
- [ ] Indicateur de charge relative : écart à la moyenne équipe
- [ ] Flag `at_risk` si workload > 1.3× moyenne ou WIP > 5

**US2.3 — Features sprint**
> En tant que data scientist, je veux des features sur la santé du sprint en cours.

Critères d'acceptation :
- [ ] Module `sprint_features.py` avec classe `SprintFeatureExtractor`
- [ ] Métriques sprint :
  - `days_elapsed` / `days_remaining`
  - `points_committed` : story points au début du sprint
  - `points_completed` : story points done
  - `points_remaining` : points non terminés
  - `completion_rate` : completed / committed
  - `expected_completion_rate` : days_elapsed / total_days
  - `scope_creep_ratio` : points ajoutés mid-sprint / committed
  - `blocked_tickets_count` : tickets avec status "Blocked" ou flag
  - `velocity_vs_average` : ratio vs moyenne 5 derniers sprints
- [ ] Burndown théorique vs réel (points restants par jour)

**US2.4 — Pipeline feature engineering**
> En tant que data scientist, je veux un pipeline reproductible pour générer les features.

Critères d'acceptation :
- [ ] Module `pipeline.py` avec classe `FeaturePipeline`
- [ ] Méthode `build_ticket_training_set()` : génère DataFrame pour entraînement
- [ ] Méthode `build_sprint_features(sprint_id)` : features sprint temps réel
- [ ] Méthode `build_developer_features()` : features tous développeurs
- [ ] Gestion des valeurs manquantes (imputation ou flag)
- [ ] Encodage catégoriel compatible LightGBM (category dtype)
- [ ] Sauvegarde des features dans DuckDB (table `ml_features`)

**US2.5 — Target variable pour estimation**
> En tant que data scientist, je veux définir clairement ce que je prédis.

Critères d'acceptation :
- [ ] Target principale : `actual_cycle_time_hours` (temps en heures entre premier "In Progress" et "Done")
- [ ] Filtrage : uniquement tickets résolus, avec changelog complet
- [ ] Exclusion : tickets < 1h (probablement mal trackés) et > 500h (outliers)
- [ ] Alternative : `actual_lead_time_hours` pour comparaison

**US2.6 — Modèle baseline**
> En tant que data scientist, je veux un modèle baseline simple pour établir une référence.

Critères d'acceptation :
- [ ] Module `ticket_estimator.py` avec classe `TicketEstimator`
- [ ] Baseline 1 : moyenne par `issue_type` (MAE baseline)
- [ ] Baseline 2 : `LinearRegression` sur features numériques uniquement
- [ ] Métriques : MAE, RMSE, R², MAPE
- [ ] Split temporel (pas random !) : train sur tickets avant date X, test après

**US2.7 — Modèle LightGBM**
> En tant que data scientist, je veux un modèle performant avec features catégorielles.

Critères d'acceptation :
- [ ] Implémentation `LGBMRegressor` dans `TicketEstimator`
- [ ] Hyperparamètres initiaux raisonnables :
  ```python
  params = {
      'objective': 'regression',
      'metric': 'mae',
      'num_leaves': 31,
      'learning_rate': 0.05,
      'feature_fraction': 0.8,
      'bagging_fraction': 0.8,
      'bagging_freq': 5,
      'verbose': -1
  }
  ```
- [ ] Cross-validation temporelle (TimeSeriesSplit, 5 folds)
- [ ] Feature importance extraction et logging
- [ ] Comparaison vs baselines

**US2.8 — Trainer et persistance**
> En tant que développeur, je veux entraîner et sauvegarder les modèles facilement.

Critères d'acceptation :
- [ ] Module `trainer.py` avec classe `ModelTrainer`
- [ ] Méthode `train(model_type, config)` : entraînement + évaluation
- [ ] Sauvegarde modèle : `models/ticket_estimator.pkl` (joblib)
- [ ] Sauvegarde metadata : `models/metadata.json` avec date, métriques, config
- [ ] Commande CLI : `python -m src.models.trainer --model ticket_estimator`
- [ ] Ajout au Makefile : `make train`

**US2.9 — Prédiction sur nouveaux tickets**
> En tant qu'utilisateur, je veux obtenir une estimation pour un ticket donné.

Critères d'acceptation :
- [ ] Méthode `TicketEstimator.predict(issue_key)` : retourne estimation en heures
- [ ] Chargement automatique du modèle depuis pickle
- [ ] Intervalle de confiance (si possible via quantile regression ou bootstrap)
- [ ] Commande CLI : `python -m src.models.ticket_estimator predict PROJ-123`

#### Livrables Sprint 2
- [ ] Pipeline feature engineering complet
- [ ] Modèle `ticket_estimator` entraîné et évalué
- [ ] Notebook `02_feature_engineering.ipynb` documentant les choix
- [ ] Comparaison MAE modèle vs baseline (objectif : -15%)
- [ ] `make train` fonctionnel

#### Orientations techniques Sprint 2

**Split temporel obligatoire** :
```python
from sklearn.model_selection import TimeSeriesSplit
# Trier par date de création avant split
df_sorted = df.sort_values('created')
tscv = TimeSeriesSplit(n_splits=5)
```

**LightGBM avec catégorielles** :
```python
# Convertir en category dtype AVANT fit
for col in categorical_cols:
    df[col] = df[col].astype('category')
model.fit(X, y, categorical_feature=categorical_cols)
```

**Gestion des outliers** :
```python
# Winsorization plutôt que suppression
from scipy.stats import mstats
y_clean = mstats.winsorize(y, limits=[0.01, 0.01])
```

---

### 🏃 SPRINT 3 : Détection Risques & Alertes
**Semaines 5-6**

#### Objectif Sprint
Implémenter la détection de risques sprint et les alertes de charge développeur. Valider les hypothèses H2 et H3.

#### User Stories

**US3.1 — Score de risque sprint**
> En tant que manager, je veux un score 0-100 indiquant le risque de retard du sprint.

Critères d'acceptation :
- [ ] Module `sprint_risk.py` avec classe `SprintRiskScorer`
- [ ] Score composite basé sur :
  - `completion_gap` : écart entre progression réelle et théorique (poids 30%)
  - `velocity_ratio` : vélocité actuelle vs historique (poids 25%)
  - `blocked_ratio` : proportion de tickets bloqués (poids 20%)
  - `scope_creep` : ajouts mid-sprint (poids 15%)
  - `days_remaining_factor` : urgence croissante (poids 10%)
- [ ] Seuils : 0-30 (vert), 31-60 (orange), 61-100 (rouge)
- [ ] Explication textuelle des facteurs contribuant au score

**US3.2 — Modèle prédictif risque sprint**
> En tant que data scientist, je veux un modèle ML pour prédire si le sprint sera complété.

Critères d'acceptation :
- [ ] Target : `sprint_completed_on_time` (binaire, basé sur historique)
- [ ] Features : métriques sprint à mi-parcours (jour 7)
- [ ] Modèle : `LGBMClassifier` ou `RandomForestClassifier`
- [ ] Métriques : Precision, Recall, F1, AUC-ROC
- [ ] Calibration des probabilités pour score interprétable

**US3.3 — Score de charge développeur**
> En tant que manager, je veux voir la charge de chaque développeur par rapport à l'équipe.

Critères d'acceptation :
- [ ] Module `workload_scorer.py` avec classe `WorkloadScorer`
- [ ] Score relatif : (charge_individu / moyenne_équipe) × 100
- [ ] Composantes de la charge :
  - Story points en cours (WIP)
  - Heures loguées sur 7 jours
  - Nombre de tickets assignés non résolus
  - Tickets en retard (overdue)
- [ ] Seuils d'alerte : > 130% (surchargé), < 70% (sous-utilisé)
- [ ] Affichage pseudonymisé par défaut

**US3.4 — Détection anomalies burn-out**
> En tant que manager, je veux détecter les signaux de surcharge prolongée.

Critères d'acceptation :
- [ ] Indicateurs de risque burn-out (agrégés équipe, pas individuel pour MVP) :
  - `overtime_ratio` : heures loguées / heures standard (>1.2 = alerte)
  - `weekend_work_frequency` : worklogs samedi/dimanche
  - `velocity_decline_trend` : pente négative sur 4 semaines
  - `wip_sustained_high` : WIP > 5 pendant > 5 jours
- [ ] Score de santé équipe (pas individuel)
- [ ] Alertes au niveau équipe, pas nominatives

**US3.5 — Générateur d'alertes**
> En tant que manager, je veux recevoir des alertes structurées sur les risques détectés.

Critères d'acceptation :
- [ ] Module `alert_generator.py` avec classe `AlertGenerator`
- [ ] Types d'alertes :
  - `SPRINT_AT_RISK` : score sprint > 60
  - `TEAM_OVERLOADED` : > 50% des devs surchargés
  - `BLOCKED_TICKETS` : > 20% des tickets du sprint bloqués
  - `VELOCITY_DECLINING` : vélocité < 70% de la moyenne
- [ ] Format alerte : `{type, severity, message, details, suggested_actions}`
- [ ] Historisation des alertes dans DuckDB (table `alerts`)

**US3.6 — CLI pour risques et alertes**
> En tant qu'utilisateur, je veux consulter les risques via ligne de commande.

Critères d'acceptation :
- [ ] Commande `python -m src.interface.cli risk sprint` : affiche score sprint actif
- [ ] Commande `python -m src.interface.cli risk team` : affiche charge équipe
- [ ] Commande `python -m src.interface.cli alerts` : liste alertes actives
- [ ] Output formaté avec `rich` (couleurs selon sévérité)
- [ ] Ajout au Makefile : `make risk`, `make alerts`

#### Livrables Sprint 3
- [ ] Modèle `sprint_risk` entraîné et évalué
- [ ] Système d'alertes fonctionnel
- [ ] CLI pour consultation risques
- [ ] Validation H2 : précision alertes sprint > 70%
- [ ] Validation H3 : détection écarts charge > 30%

#### Orientations techniques Sprint 3

**Score composite avec poids** :
```python
def compute_risk_score(features):
    weights = {
        'completion_gap': 0.30,
        'velocity_ratio': 0.25,
        'blocked_ratio': 0.20,
        'scope_creep': 0.15,
        'urgency': 0.10
    }
    score = sum(w * normalize(features[k]) for k, w in weights.items())
    return min(100, max(0, score * 100))
```

**Rich pour CLI colorée** :
```python
from rich.console import Console
from rich.table import Table
console = Console()
# Rouge/orange/vert selon sévérité
console.print(f"[red]ALERT:[/red] {message}")
```

---

### 🏃 SPRINT 4 : Intelligence LLM & Interface Conversationnelle
**Semaines 7-8**

#### Objectif Sprint
Intégrer le LLM local (Ollama) pour l'analyse conversationnelle et les recommandations. Début de l'interface Streamlit.

#### User Stories

**US4.1 — Client Ollama**
> En tant que développeur, je veux une interface simple pour communiquer avec le LLM local.

Critères d'acceptation :
- [ ] Module `llm_client.py` avec classe `OllamaClient`
- [ ] Configuration : modèle (`llama3.1:8b`), température, max_tokens
- [ ] Méthode `chat(messages)` : format OpenAI-compatible
- [ ] Méthode `complete(prompt)` : complétion simple
- [ ] Gestion timeout et retry
- [ ] Fallback message si Ollama non disponible
- [ ] Support streaming (optionnel mais nice-to-have)

**US4.2 — Templates de prompts**
> En tant que développeur, je veux des prompts structurés et réutilisables.

Critères d'acceptation :
- [ ] Module `prompts.py` avec templates :
  - `SPRINT_ANALYSIS_PROMPT` : analyse état du sprint
  - `TICKET_ESTIMATION_PROMPT` : expliquer une estimation
  - `WORKLOAD_ANALYSIS_PROMPT` : analyser charge équipe
  - `RECOMMENDATION_PROMPT` : générer suggestions d'actions
  - `RELEASE_PLANNING_PROMPT` : aide à la planification
- [ ] Injection de données structurées (JSON/YAML) dans les prompts
- [ ] System prompt définissant le rôle de l'assistant

**US4.3 — Analyste IA**
> En tant que manager, je veux que l'IA analyse mes données et m'explique la situation.

Critères d'acceptation :
- [ ] Module `analyst.py` avec classe `JiraAnalyst`
- [ ] Méthode `analyze_sprint(sprint_id)` : résumé + insights
- [ ] Méthode `analyze_ticket(issue_key)` : analyse + estimation expliquée
- [ ] Méthode `analyze_team_health()` : état de l'équipe
- [ ] Méthode `answer_question(question, context)` : Q&A libre sur les données
- [ ] Formatage des données Jira en contexte digestible pour le LLM

**US4.4 — Recommandations d'actions**
> En tant que manager, je veux des suggestions concrètes pour résoudre les problèmes.

Critères d'acceptation :
- [ ] Module `recommender.py` avec classe `ActionRecommender`
- [ ] Recommandations contextuelles :
  - Sprint à risque → actions pour le sauver
  - Développeur surchargé → réassignations suggérées
  - Ticket bloqué → escalade ou déblocage
  - Backlog mal priorisé → re-priorisation
- [ ] Format : `{action, priority, rationale, effort, impact}`
- [ ] Scoring des recommandations par pertinence

**US4.5 — Interface CLI conversationnelle**
> En tant qu'utilisateur, je veux dialoguer avec l'IA depuis le terminal.

Critères d'acceptation :
- [ ] Mode interactif : `python -m src.interface.cli chat`
- [ ] Historique de conversation maintenu
- [ ] Commandes spéciales : `/sprint`, `/ticket PROJ-123`, `/team`, `/quit`
- [ ] Affichage streaming de la réponse (si supporté)
- [ ] Ctrl+C pour interrompre proprement

**US4.6 — Dashboard Streamlit - Setup**
> En tant qu'utilisateur, je veux une interface visuelle basique pour voir les métriques.

Critères d'acceptation :
- [ ] Module `dashboard.py` avec app Streamlit
- [ ] Page d'accueil avec :
  - Score de risque sprint (gauge ou progress bar)
  - Métriques clés : points restants, jours restants, vélocité
  - Liste des alertes actives
- [ ] Sidebar : sélection projet, sprint, période
- [ ] Commande : `make dashboard` → `streamlit run src/interface/dashboard.py`

**US4.7 — Dashboard - Vue équipe**
> En tant que manager, je veux voir la charge de l'équipe visuellement.

Critères d'acceptation :
- [ ] Page "Équipe" dans Streamlit
- [ ] Bar chart : charge par développeur (pseudonymisé)
- [ ] Heatmap : activité par jour de la semaine
- [ ] Tableau : WIP, points complétés, alertes par personne
- [ ] Toggle pour afficher/masquer les noms (mode anonyme)

**US4.8 — Dashboard - Chat intégré**
> En tant qu'utilisateur, je veux dialoguer avec l'IA dans le dashboard.

Critères d'acceptation :
- [ ] Page "Assistant" ou sidebar chat
- [ ] Input texte + historique conversation
- [ ] Boutons rapides : "Analyse sprint", "État équipe", "Prochaines actions"
- [ ] Affichage Markdown des réponses

#### Livrables Sprint 4
- [ ] Intégration Ollama fonctionnelle
- [ ] Chat CLI interactif
- [ ] Dashboard Streamlit v1 (basique mais fonctionnel)
- [ ] Notebook `03_model_experiments.ipynb` avec analyses LLM
- [ ] `make dashboard` et `make chat` fonctionnels

#### Orientations techniques Sprint 4

**Format OpenAI pour Ollama** :
```python
from openai import OpenAI
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
response = client.chat.completions.create(
    model='llama3.1:8b',
    messages=[
        {"role": "system", "content": "Tu es un assistant expert en gestion de projet..."},
        {"role": "user", "content": "Analyse ce sprint..."}
    ]
)
```

**Contexte structuré pour le LLM** :
```python
context = f"""
## Sprint actuel: {sprint.name}
- Jours restants: {days_remaining}
- Points complétés: {completed}/{committed}
- Score de risque: {risk_score}/100

## Tickets bloqués:
{blocked_tickets_summary}

## Charge équipe:
{team_workload_summary}
"""
```

**Streamlit minimal** :
```python
import streamlit as st
st.set_page_config(page_title="Jira Co-pilot", layout="wide")
st.metric("Risk Score", f"{risk_score}/100", delta=delta_vs_yesterday)
```

---

### 🏃 SPRINT 5 : Load Balancing & Release Planning
**Semaines 9-10**

#### Objectif Sprint
Implémenter les suggestions de réassignation (load balancing) et la génération de plans de release. Valider l'hypothèse H4.

#### User Stories

**US5.1 — Algorithme de load balancing**
> En tant que manager, je veux des suggestions de réassignation pour équilibrer la charge.

Critères d'acceptation :
- [ ] Module `load_balancer.py` avec classe `LoadBalancer`
- [ ] Input : tickets à réassigner (optionnel), contraintes (skills, préférences)
- [ ] Output : liste de `{ticket, from_dev, to_dev, reason, confidence}`
- [ ] Algorithme :
  1. Identifier les devs surchargés (> 130% charge moyenne)
  2. Identifier les devs disponibles (< 80% charge moyenne)
  3. Matcher tickets selon : composant/skill, historique succès, charge résultante
- [ ] Respecter les contraintes : ne pas surcharger le receveur
- [ ] Score de confiance basé sur la qualité du match

**US5.2 — Suggestions de réassignation sprint**
> En tant que manager, je veux voir les réassignations suggérées pour le sprint en cours.

Critères d'acceptation :
- [ ] Méthode `suggest_sprint_rebalancing(sprint_id)`
- [ ] Prioriser les tickets non commencés
- [ ] Éviter de réassigner les tickets presque terminés
- [ ] Afficher l'impact projeté sur le score de risque
- [ ] Export en format actionnable (CSV ou JSON)

**US5.3 — Planificateur de release**
> En tant que manager, je veux générer un plan de release optimisé.

Critères d'acceptation :
- [ ] Module `release_planner.py` avec classe `ReleasePlanner`
- [ ] Input : backlog (tickets), date cible, capacité équipe
- [ ] Output : plan de release avec sprints suggérés
- [ ] Algorithme :
  1. Estimer la durée de chaque ticket (via modèle ML)
  2. Calculer la capacité par sprint (vélocité historique)
  3. Affecter les tickets par priorité + dépendances
  4. Optimiser pour minimiser le time-to-value
- [ ] Gestion des dépendances (ticket A doit être fait avant B)

**US5.4 — Contraintes de release**
> En tant que manager, je veux pouvoir spécifier des contraintes pour le plan.

Critères d'acceptation :
- [ ] Contraintes supportées :
  - `must_include` : tickets obligatoires dans la release
  - `deadline` : date limite hard
  - `max_risk` : niveau de risque acceptable
  - `team_availability` : absences planifiées
- [ ] Validation des contraintes (infaisabilité détectée)
- [ ] Mode "what-if" : simuler différents scénarios

**US5.5 — Génération plan via LLM**
> En tant que manager, je veux que l'IA m'explique et affine le plan de release.

Critères d'acceptation :
- [ ] Prompt structuré avec : backlog, contraintes, capacité
- [ ] LLM génère : justification des choix, risques identifiés, alternatives
- [ ] Dialogue itératif : "Et si on ajoutait ce ticket ?"
- [ ] Export du plan final en Markdown ou CSV

**US5.6 — Dashboard - Vue Load Balancing**
> En tant que manager, je veux visualiser les suggestions de réassignation.

Critères d'acceptation :
- [ ] Page "Load Balancing" dans Streamlit
- [ ] Visualisation : barre de charge actuelle vs projetée
- [ ] Tableau des réassignations suggérées
- [ ] Bouton "Appliquer" (pour l'instant : export CSV, pas d'écriture Jira)
- [ ] Filtres : par sévérité, par développeur

**US5.7 — Dashboard - Vue Release Planning**
> En tant que manager, je veux planifier mes releases visuellement.

Critères d'acceptation :
- [ ] Page "Release Planning" dans Streamlit
- [ ] Input : sélection tickets du backlog, date cible
- [ ] Output : timeline des sprints avec tickets assignés
- [ ] Indicateurs : probabilité de succès, buffer disponible
- [ ] Export du plan en Markdown/CSV

**US5.8 — Backlog priorization assistant**
> En tant que manager, je veux de l'aide pour prioriser mon backlog.

Critères d'acceptation :
- [ ] Score de priorisation par ticket :
  - `business_value` : estimé ou taggé (High/Medium/Low)
  - `effort` : estimation ML
  - `dependencies_impact` : combien de tickets débloqués
  - `age` : depuis combien de temps dans le backlog
- [ ] Ranking WSJF-like : (value + urgency) / effort
- [ ] Suggestions LLM : "Ces 5 tickets devraient être priorisés parce que..."

#### Livrables Sprint 5
- [ ] Algorithme load balancing fonctionnel
- [ ] Release planner avec export
- [ ] Dashboard pages load balancing et release
- [ ] Validation H4 : temps de planification réduit (mesure qualitative)
- [ ] `make plan-release` fonctionnel

#### Orientations techniques Sprint 5

**Algorithme simple de bin packing pour release** :
```python
def assign_to_sprints(tickets, sprint_capacity):
    sprints = []
    current_sprint = []
    current_load = 0
    
    for ticket in sorted(tickets, key=lambda t: -t['priority']):
        if current_load + ticket['estimate'] <= sprint_capacity:
            current_sprint.append(ticket)
            current_load += ticket['estimate']
        else:
            sprints.append(current_sprint)
            current_sprint = [ticket]
            current_load = ticket['estimate']
    
    if current_sprint:
        sprints.append(current_sprint)
    
    return sprints
```

**Gestion des dépendances (tri topologique)** :
```python
from collections import deque

def topological_sort(tickets, dependencies):
    # dependencies: dict {ticket_key: [blocked_by_keys]}
    # Retourne tickets ordonnés avec dépendances respectées
    ...
```

---

### 🏃 SPRINT 6 : Consolidation, Tests & Documentation
**Semaines 11-12**

#### Objectif Sprint
Stabiliser le MVP, ajouter les tests manquants, documenter et préparer pour une utilisation quotidienne.

#### User Stories

**US6.1 — Tests unitaires complets**
> En tant que développeur, je veux une couverture de tests suffisante pour les modules critiques.

Critères d'acceptation :
- [ ] Couverture > 60% sur les modules `src/`
- [ ] Tests pour :
  - `jira_client/` : mock des appels API
  - `features/` : validation des calculs
  - `models/` : prédictions sur données connues
  - `intelligence/` : mock du LLM
- [ ] Fixtures pytest dans `conftest.py` avec données de test
- [ ] `make test` passe sans erreur

**US6.2 — Tests d'intégration**
> En tant que développeur, je veux valider les flux end-to-end.

Critères d'acceptation :
- [ ] Test E2E : sync → features → prediction
- [ ] Test E2E : sprint features → risk score → alerts
- [ ] Test avec base DuckDB de test (pas la prod)
- [ ] Données de test réalistes (anonymisées de vraies données)

**US6.3 — Gestion des erreurs robuste**
> En tant qu'utilisateur, je veux des messages d'erreur clairs et une app qui ne crashe pas.

Critères d'acceptation :
- [ ] Try/except appropriés dans tous les modules
- [ ] Messages d'erreur user-friendly (pas de stacktraces brutes)
- [ ] Logging structuré avec `loguru` (fichier + console)
- [ ] Graceful degradation : si Ollama down, fonctionnalités ML marchent encore

**US6.4 — Configuration flexible**
> En tant qu'utilisateur, je veux pouvoir configurer l'outil sans modifier le code.

Critères d'acceptation :
- [ ] Toute la config dans `.env` et `config/*.yaml`
- [ ] Validation des configs au démarrage (Pydantic)
- [ ] Valeurs par défaut sensées
- [ ] Documentation des options de config dans README

**US6.5 — Documentation technique**
> En tant que développeur futur, je veux comprendre comment le projet fonctionne.

Critères d'acceptation :
- [ ] README complet :
  - Installation (prérequis, setup)
  - Configuration (Jira, Ollama)
  - Utilisation (commandes principales)
  - Architecture (schéma simplifié)
- [ ] Docstrings sur les classes et méthodes publiques
- [ ] `ARCHITECTURE.md` : décisions techniques et rationale
- [ ] `CHANGELOG.md` : historique des versions

**US6.6 — Dashboard polish**
> En tant qu'utilisateur, je veux un dashboard utilisable au quotidien.

Critères d'acceptation :
- [ ] Navigation cohérente entre les pages
- [ ] Indicateurs de chargement (spinners)
- [ ] Refresh automatique des données (bouton ou timer)
- [ ] Responsive basique (utilisable sur laptop)
- [ ] Thème cohérent (dark mode optionnel)

**US6.7 — Automatisation sync**
> En tant qu'utilisateur, je veux que les données se synchronisent automatiquement.

Critères d'acceptation :
- [ ] APScheduler configuré pour sync toutes les 30 minutes
- [ ] Persistence du scheduler (survit aux redémarrages)
- [ ] Option de sync manuel dans le dashboard
- [ ] Indicateur de fraîcheur des données ("Dernière sync : il y a 15 min")

**US6.8 — Export et rapports**
> En tant que manager, je veux exporter des rapports pour les partager.

Critères d'acceptation :
- [ ] Export sprint report en Markdown
- [ ] Export team workload en CSV
- [ ] Export release plan en Markdown
- [ ] Export alertes en JSON
- [ ] Boutons d'export dans le dashboard

**US6.9 — Mode démo / données de test**
> En tant que développeur, je veux pouvoir démontrer l'outil sans données réelles.

Critères d'acceptation :
- [ ] Script `generate_demo_data.py` : génère données fictives réalistes
- [ ] Flag `--demo` pour lancer avec données de démo
- [ ] Données de démo : 200 tickets, 10 devs, 6 mois d'historique

**US6.10 — Validation finale hypothèses**
> En tant que product owner, je veux mesurer le succès du MVP.

Critères d'acceptation :
- [ ] H1 (Estimation) : Comparer MAE modèle vs estimations humaines sur 20 tickets
- [ ] H2 (Risque sprint) : Vérifier rétrospectivement sur 3 sprints passés
- [ ] H3 (Charge équipe) : Confirmer détection des déséquilibres connus
- [ ] H4 (Release planning) : Mesurer temps de planification avant/après
- [ ] Document `VALIDATION.md` avec résultats

#### Livrables Sprint 6
- [ ] Suite de tests complète (> 60% coverage)
- [ ] Documentation complète (README, ARCHITECTURE)
- [ ] Dashboard stable et utilisable
- [ ] Sync automatisé
- [ ] Rapport de validation des hypothèses
- [ ] **MVP COMPLET ET VALIDÉ**

#### Orientations techniques Sprint 6

**Structure tests avec pytest** :
```python
# tests/conftest.py
import pytest
import duckdb

@pytest.fixture
def test_db():
    conn = duckdb.connect(':memory:')
    # Setup schema
    yield conn
    conn.close()

@pytest.fixture
def mock_jira_client(mocker):
    return mocker.patch('src.jira_client.fetcher.JiraFetcher')
```

**Logging avec loguru** :
```python
from loguru import logger

logger.add("logs/app.log", rotation="10 MB", retention="7 days")
logger.info("Sync started", project="PROJ", mode="incremental")
```

---

## ANNEXES

### A. Commandes Makefile finales

```makefile
.PHONY: install test lint sync train predict risk alerts chat dashboard

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format src/ tests/

sync:
	python -m src.jira_client.sync --mode incremental

sync-full:
	python -m src.jira_client.sync --mode full

train:
	python -m src.models.trainer --all

predict:
	python -m src.models.ticket_estimator predict $(TICKET)

risk:
	python -m src.interface.cli risk sprint

alerts:
	python -m src.interface.cli alerts

chat:
	python -m src.interface.cli chat

dashboard:
	streamlit run src/interface/dashboard.py

demo:
	python scripts/generate_demo_data.py
	$(MAKE) dashboard
```

### B. Variables d'environnement (.env.example)

```env
# Jira Configuration
JIRA_URL=https://your-instance.atlassian.net
JIRA_EMAIL=your-email@company.com
JIRA_API_TOKEN=your-api-token
JIRA_PROJECT_KEY=PROJ
JIRA_BOARD_ID=1

# Custom Fields Mapping (optionnel, override dans jira_config.yaml)
JIRA_STORY_POINTS_FIELD=customfield_10016

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Application Settings
LOG_LEVEL=INFO
SYNC_INTERVAL_MINUTES=30
ANONYMIZE_DEVELOPERS=true

# Database
DATABASE_PATH=data/jira.duckdb
```

### C. Critères de succès MVP

| Métrique | Cible | Méthode de mesure |
|----------|-------|-------------------|
| MAE estimation vs baseline | -15% | Comparaison sur 50 tickets test |
| Précision alertes sprint | > 70% | Validation rétro sur 5 sprints |
| Détection déséquilibre charge | > 30% écart | Comparaison avec perception manager |
| Temps planification release | -50% | Mesure avant/après sur 1 release |
| Temps daily review | -30% | Mesure subjective |
| Satisfaction utilisateur | > 4/5 | Auto-évaluation |

### D. Évolutions post-MVP (backlog futur)

1. **Intégration bidirectionnelle Jira** : créer/modifier tickets depuis l'outil
2. **Notifications Slack/Teams** : alertes en temps réel
3. **Multi-projets** : gérer plusieurs projets Jira
4. **Embeddings pour similarité** : trouver tickets similaires historiques
5. **MCP Anthropic** : intégration temps réel via protocol
6. **Déploiement cloud** : version hébergée
7. **API REST** : exposer les fonctionnalités pour intégrations
8. **Fine-tuning LLM** : améliorer les réponses sur le domaine spécifique

---

## INSTRUCTIONS POUR CLAUDE OPUS 4.5

Tu es le développeur principal de ce projet. Ton rôle est d'implémenter chaque sprint de manière autonome, en respectant :

1. **La stack technique imposée** — Ne propose pas d'alternatives sauf si bloqué
2. **La structure de projet** — Respecte l'arborescence définie
3. **Les critères d'acceptation** — Chaque US doit être 100% complétée
4. **La philosophie MVP** — Fonctionnel > Parfait, mais pas de code sale
5. **Les bonnes pratiques** — Type hints, docstrings, tests

À chaque session de travail :
- Indique le sprint et l'US en cours
- Propose le code complet (pas de placeholders)
- Explique les choix techniques si pertinent
- Signale les blocages ou questions

**Let's build this! 🚀**