# Guide Complet d'Installation et d'Utilisation
## Jira AI Co-Pilot - Votre Assistant Intelligent de Gestion de Projet

---

> **Pour qui est ce guide ?**
> Ce guide est conçu pour les utilisateurs **débutants** qui n'ont aucune expérience technique préalable. Chaque étape est expliquée en détail avec des captures d'écran et des conseils.

---

## Table des Matières

1. [Introduction - Qu'est-ce que Jira AI Co-Pilot ?](#1-introduction)
2. [Prérequis - Ce dont vous avez besoin](#2-prérequis)
3. [Étape 1 - Accéder à Antigravity (l'éditeur cloud Google)](#3-étape-1---accéder-à-antigravity)
4. [Étape 2 - Importer le projet dans Antigravity](#4-étape-2---importer-le-projet-dans-antigravity)
5. [Étape 3 - Configurer votre token Jira](#5-étape-3---configurer-votre-token-jira)
6. [Étape 4 - Configurer Google Gemini (IA)](#6-étape-4---configurer-google-gemini-ia)
7. [Étape 5 - Installer les dépendances](#7-étape-5---installer-les-dépendances)
8. [Étape 6 - Lancer l'application](#8-étape-6---lancer-lapplication)
9. [Utilisation du Dashboard](#9-utilisation-du-dashboard)
10. [Ajouter un nouveau module](#10-ajouter-un-nouveau-module)
11. [Dépannage - Problèmes courants](#11-dépannage---problèmes-courants)
12. [Glossaire - Termes techniques expliqués](#12-glossaire)

---

## 1. Introduction

### Qu'est-ce que Jira AI Co-Pilot ?

**Jira AI Co-Pilot** est une application intelligente qui se connecte à votre compte Jira pour vous aider à :

| Fonctionnalité | Description |
|----------------|-------------|
| **Prédire les délais** | L'IA estime combien de temps prendra chaque tâche |
| **Détecter les risques** | Identifie les sprints à risque avant qu'il ne soit trop tard |
| **Analyser la charge de travail** | Visualise qui est surchargé dans l'équipe |
| **Générer des rapports** | Crée des rapports automatiques pour vos réunions |
| **Alerter sur le burnout** | Détecte les signes de surcharge chez les membres de l'équipe |

### Comment ça marche ?

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   Votre     │ ──► │  Jira AI        │ ──► │  Dashboard       │
│   Jira      │     │  Co-Pilot       │     │  Interactif      │
└─────────────┘     └─────────────────┘     └──────────────────┘
       │                    │                        │
       │                    ▼                        │
       │            ┌───────────────┐                │
       └──────────► │   Base de     │ ◄──────────────┘
                    │   Données     │
                    │   Locale      │
                    └───────────────┘
```

L'application se connecte à votre Jira, récupère vos données, les analyse avec de l'intelligence artificielle, et vous présente les résultats dans un tableau de bord visuel.

---

## 2. Prérequis

Avant de commencer, assurez-vous d'avoir :

### Matériel nécessaire
- Un ordinateur avec un **navigateur web moderne** (Chrome recommandé)
- Une connexion internet stable

> **Bonne nouvelle !** Avec Antigravity, vous n'avez **rien à installer** sur votre ordinateur. Tout fonctionne dans le navigateur !

### Comptes nécessaires
- Un compte **Google** (obligatoire pour Antigravity et Gemini)
- Un compte **Jira Cloud** (Atlassian)

### Temps estimé
| Étape | Durée |
|-------|-------|
| Création du workspace Antigravity | 5 minutes |
| Import du projet | 5 minutes |
| Configuration Jira & Gemini | 15 minutes |
| Installation des dépendances | 10 minutes |
| **Total** | **~35 minutes** |

> **Avantage d'Antigravity** : Pas besoin d'installer Python, VS Code, ou quoi que ce soit ! Tout est pré-configuré dans le cloud Google.

---

## 3. Étape 1 - Accéder à Antigravity

### Qu'est-ce qu'Antigravity ?

**Antigravity** est un éditeur de code **dans le cloud** créé par Google. Il fonctionne entièrement dans votre navigateur web.

**Avantages pour vous** :
- ✅ **Aucune installation** sur votre ordinateur
- ✅ **Python pré-installé** (pas besoin de le configurer)
- ✅ **Accessible partout** depuis n'importe quel ordinateur
- ✅ **Sauvegarde automatique** dans le cloud Google
- ✅ **Terminal intégré** pour exécuter les commandes

### Accéder à Antigravity

1. **Ouvrez votre navigateur** (Chrome recommandé)

2. **Allez sur Antigravity** :
   ```
   https://antigravity.google/
   ```

3. **Connectez-vous avec votre compte Google**

   ```
   ┌─────────────────────────────────────────────────────────┐
   │                                                         │
   │           🚀 Antigravity                                │
   │                                                         │
   │     Bienvenue ! Connectez-vous pour continuer          │
   │                                                         │
   │     [  Se connecter avec Google  ]  ← Cliquez ici      │
   │                                                         │
   └─────────────────────────────────────────────────────────┘
   ```

4. **Autorisez l'accès** si demandé

5. **Vous arrivez sur la page d'accueil d'Antigravity**

```
┌─────────────────────────────────────────────────────────────────────┐
│  🚀 Antigravity                                    [votre-email]   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Mes Workspaces                                                    │
│                                                                     │
│   ┌─────────────────┐   ┌─────────────────┐                        │
│   │                 │   │                 │                        │
│   │  + Nouveau      │   │  Importer       │                        │
│   │    Workspace    │   │  depuis GitHub  │                        │
│   │                 │   │                 │                        │
│   └─────────────────┘   └─────────────────┘                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Félicitations !** Vous avez accès à Antigravity.

---

## 4. Étape 2 - Importer le projet dans Antigravity

### Option A : Importer depuis GitHub (Recommandé)

Si le projet est sur GitHub :

1. **Sur la page d'accueil d'Antigravity**, cliquez sur **"Import from GitHub"** ou **"Importer depuis GitHub"**

2. **Connectez votre compte GitHub** si demandé

3. **Recherchez le projet** "jira-copilot" ou "nabda"

4. **Cliquez sur "Import"**

5. **Attendez** que le workspace se crée (1-2 minutes)

### Option B : Créer un nouveau workspace et uploader les fichiers

Si vous avez téléchargé le projet en ZIP :

1. **Cliquez sur "+ New Workspace"** ou **"+ Nouveau Workspace"**

2. **Choisissez "Python"** comme type de projet

3. **Donnez un nom** : `jira-copilot`

4. **Cliquez sur "Create"**

5. **Une fois le workspace créé**, vous pouvez glisser-déposer les fichiers du projet

### L'interface Antigravity

Une fois le projet ouvert, vous verrez cette interface :

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🚀 Antigravity - jira-copilot                              [Menu]     │
├───────────────┬─────────────────────────────────────────────────────────┤
│               │                                                         │
│  📁 EXPLORER  │   [Zone d'édition du code]                             │
│               │                                                         │
│  ▼ nabda      │   # Fichier ouvert                                     │
│    ▶ config/  │   contenu du fichier...                                │
│    ▶ data/    │                                                         │
│    ▶ src/     │                                                         │
│    ▶ tests/   │                                                         │
│    📄 .env    │                                                         │
│    📄 ...     │                                                         │
│               │                                                         │
├───────────────┴─────────────────────────────────────────────────────────┤
│  TERMINAL                                                               │
│  user@workspace:~/jira-copilot$ _                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Éléments importants** :
- **Explorer** (gauche) : Liste de tous les fichiers du projet
- **Éditeur** (centre) : Zone pour modifier le code
- **Terminal** (bas) : Pour taper des commandes

### Ouvrir le terminal

Si le terminal n'est pas visible :

1. **Cliquez sur le menu** (icône ≡ ou "View")
2. **Sélectionnez "Terminal"** ou appuyez sur `` Ctrl + ` ``

Le terminal s'ouvre en bas de l'écran. C'est ici que vous taperez les commandes.

---

## 5. Étape 3 - Configurer votre token Jira

### Qu'est-ce qu'un token API ?

Un **token API** est comme un mot de passe spécial qui permet à l'application de se connecter à votre Jira de manière sécurisée.

> **Sécurité** : Le token ne donne pas accès à votre compte Atlassian complet, seulement à l'API Jira. Vous pouvez le révoquer à tout moment.

### Étape 6.1 - Créer votre token Jira

1. **Connectez-vous à Atlassian** :
   ```
   https://id.atlassian.com/manage-profile/security/api-tokens
   ```

2. **Cliquez sur "Create API token"** (Créer un token API)
   ```
   ┌─────────────────────────────────────────────────────────┐
   │  API tokens                                             │
   │                                                         │
   │  [  Create API token  ]  ← Cliquez ici                 │
   │                                                         │
   └─────────────────────────────────────────────────────────┘
   ```

3. **Donnez un nom à votre token** :
   - Label : `Jira AI Copilot`
   - Cliquez sur "Create"

4. **IMPORTANT - Copiez le token** :
   ```
   ┌─────────────────────────────────────────────────────────┐
   │  Your new API token                                     │
   │                                                         │
   │  ╔═══════════════════════════════════════════════════╗  │
   │  ║  ABcD1234efGH5678ijKL9012mnOP...                  ║  │
   │  ╚═══════════════════════════════════════════════════╝  │
   │                                                         │
   │  [  Copy  ]     ← CLIQUEZ ICI POUR COPIER              │
   │                                                         │
   │  ⚠️ You won't be able to see this token again!         │
   └─────────────────────────────────────────────────────────┘
   ```

   > ⚠️ **ATTENTION** : Vous ne pourrez plus voir ce token après avoir fermé cette fenêtre ! Copiez-le maintenant et gardez-le dans un endroit sûr (temporairement).

### Étape 6.2 - Trouver les informations de votre Jira

Vous aurez besoin de ces informations :

| Information | Où la trouver | Exemple |
|-------------|---------------|---------|
| **URL Jira** | L'adresse de votre Jira | `https://votreentreprise.atlassian.net` |
| **Email** | L'email de votre compte Atlassian | `votre.email@entreprise.com` |
| **Clé du projet** | Dans l'URL de votre projet Jira | `PROJ` |
| **ID du tableau** | Dans l'URL de votre tableau Kanban/Scrum | `123` |

#### Comment trouver la clé du projet ?

1. Ouvrez votre projet Jira
2. Regardez l'URL :
   ```
   https://votreentreprise.atlassian.net/jira/software/projects/PROJ/boards/123
                                                             ^^^^
                                                             C'est la clé !
   ```

#### Comment trouver l'ID du tableau ?

1. Dans la même URL :
   ```
   https://votreentreprise.atlassian.net/jira/software/projects/PROJ/boards/123
                                                                            ^^^
                                                                            C'est l'ID !
   ```

### Étape 5.3 - Créer le fichier .env

Le fichier `.env` contient toutes vos informations de configuration secrètes.

1. **Dans Antigravity**, regardez la liste des fichiers à gauche (Explorer)

2. **Trouvez le fichier `.env.example`**

3. **Faites un clic droit** dessus et sélectionnez **"Rename"** (Renommer)

4. **Renommez-le en** `.env` (supprimez `.example`)

   > **Note** : Dans Antigravity, les fichiers commençant par `.` sont toujours visibles.

5. **Ouvrez le fichier `.env`** en double-cliquant dessus

6. **Modifiez le contenu** avec vos informations :

```env
# ═══════════════════════════════════════════════════════════════
# CONFIGURATION JIRA
# ═══════════════════════════════════════════════════════════════

# L'URL de votre instance Jira Cloud
# Exemple : https://monentreprise.atlassian.net
JIRA_URL=https://votreentreprise.atlassian.net

# Votre email Atlassian (celui que vous utilisez pour vous connecter)
JIRA_EMAIL=votre.email@entreprise.com

# Le token API que vous avez créé à l'étape précédente
# ⚠️ GARDEZ CE TOKEN SECRET - Ne le partagez jamais !
JIRA_API_TOKEN=votre_token_api_ici

# La clé de votre projet (ex: PROJ, DEV, MARKETING)
JIRA_PROJECT_KEY=PROJ

# L'ID de votre tableau Kanban/Scrum
JIRA_BOARD_ID=123

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION IA (Google Gemini)
# ═══════════════════════════════════════════════════════════════

# Votre clé API Google (voir étape suivante)
GOOGLE_API_KEY=votre_cle_google_ici

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION BASE DE DONNÉES (optionnel - laissez par défaut)
# ═══════════════════════════════════════════════════════════════

# Chemin vers la base de données locale
DATABASE_PATH=data/jira.duckdb

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION APPLICATION (optionnel)
# ═══════════════════════════════════════════════════════════════

# Niveau de logs (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Intervalle de synchronisation automatique (en minutes)
SYNC_INTERVAL_MINUTES=30

# Activer les fonctionnalités IA
ENABLE_LLM_FEATURES=true
```

7. **Sauvegardez le fichier** : `Ctrl + S` ou `Cmd + S` (le fichier est aussi auto-sauvegardé par Antigravity)

### Exemple concret

Voici à quoi pourrait ressembler votre fichier `.env` rempli :

```env
JIRA_URL=https://acme-corp.atlassian.net
JIRA_EMAIL=jean.dupont@acme.com
JIRA_API_TOKEN=ABcD1234efGH5678ijKL9012mnOPqrST
JIRA_PROJECT_KEY=ACME
JIRA_BOARD_ID=42
GOOGLE_API_KEY=AIzaSyC...votreCleGoogle...xyz
DATABASE_PATH=data/jira.duckdb
LOG_LEVEL=INFO
SYNC_INTERVAL_MINUTES=30
ENABLE_LLM_FEATURES=true
```

> **Sécurité** : Ne partagez JAMAIS ce fichier ! Il contient vos informations confidentielles.

---

## 6. Étape 4 - Configurer Google Gemini (IA)

### Pourquoi Gemini ?

**Google Gemini** est l'intelligence artificielle qui alimente les fonctionnalités avancées de l'application :
- Résumés automatiques
- Analyse des risques en langage naturel
- Suggestions intelligentes
- Briefings quotidiens

### Créer une clé API Google

1. **Allez sur Google AI Studio** :
   ```
   https://aistudio.google.com/app/apikey
   ```

2. **Connectez-vous** avec votre compte Google

3. **Cliquez sur "Create API Key"**
   ```
   ┌─────────────────────────────────────────────────────────┐
   │  API keys                                               │
   │                                                         │
   │  [  Create API key  ]  ← Cliquez ici                   │
   │                                                         │
   └─────────────────────────────────────────────────────────┘
   ```

4. **Sélectionnez "Create API key in new project"**

5. **Copiez la clé générée** :
   ```
   ┌─────────────────────────────────────────────────────────┐
   │  API key created                                        │
   │                                                         │
   │  AIzaSyC...votre-longue-cle-api...xyz                  │
   │                                                         │
   │  [  Copy  ]                                             │
   └─────────────────────────────────────────────────────────┘
   ```

6. **Collez cette clé** dans votre fichier `.env` à la ligne `GOOGLE_API_KEY=`

### Tarification

- L'API Gemini 2.0 Flash est **gratuite** pour un usage modéré
- Limite gratuite : ~60 requêtes par minute
- Pour un usage intensif, des forfaits payants sont disponibles

---

## 7. Étape 5 - Installer les dépendances

### Qu'est-ce qu'une dépendance ?

Les **dépendances** sont des bibliothèques (des outils pré-fabriqués) dont l'application a besoin pour fonctionner.

### Installer toutes les dépendances dans Antigravity

1. **Ouvrez le terminal** dans Antigravity (en bas de l'écran, ou `Ctrl + ù`)

2. **Tapez cette commande** et appuyez sur Entrée :
   ```bash
   pip install -e ".[dev]"
   ```

3. **Attendez** - L'installation prend **3 à 10 minutes**.

   Vous verrez beaucoup de texte défiler, c'est normal :
   ```
   Collecting streamlit>=1.35.0
     Downloading streamlit-1.35.0-py2.py3-none-any.whl (8.5 MB)
   Collecting pandas>=2.0.0
     ...
   Successfully installed ...
   ```

4. **Vérifiez l'installation** en tapant :
   ```bash
   pip list | head -20
   ```

> **Note** : Dans Antigravity, Python est déjà installé et configuré. Pas besoin d'environnement virtuel !

### Principales dépendances installées

| Package | Rôle |
|---------|------|
| `streamlit` | Crée le tableau de bord interactif |
| `pandas` | Manipule les données |
| `jira` | Se connecte à votre Jira |
| `duckdb` | Base de données locale ultra-rapide |
| `lightgbm` | Intelligence artificielle pour les prédictions |
| `google-generativeai` | Connexion à l'IA Gemini de Google |
| `plotly` | Graphiques interactifs |

---

## 8. Étape 6 - Lancer l'application

### Initialiser la base de données

Avant la première utilisation, initialisez la base de données :

1. **Dans le terminal Antigravity** (en bas de l'écran)

2. **Tapez cette commande** :
   ```bash
   jira-copilot init
   ```

   Vous devriez voir :
   ```
   ✅ Database initialized at data/jira.duckdb
   ✅ Configuration validated
   ✅ System ready!
   ```

### Synchroniser les données Jira

Maintenant, récupérons vos données depuis Jira :

```bash
jira-copilot sync full
```

> **Première synchronisation** : Elle peut prendre **5 à 30 minutes** selon la quantité de données dans votre Jira.

Vous verrez la progression :
```
🔄 Connecting to Jira...
✅ Connected to https://votreentreprise.atlassian.net
🔄 Fetching issues... 234/500
🔄 Fetching sprints... 12/12
🔄 Processing worklogs...
✅ Sync complete! 500 issues synchronized.
```

### Entraîner les modèles IA (optionnel mais recommandé)

Pour activer les prédictions intelligentes :

```bash
jira-copilot train
```

Cela prend généralement **2-5 minutes**.

### Lancer le dashboard

C'est le moment tant attendu ! Lancez le tableau de bord :

```bash
jira-copilot dashboard
```

Ou directement avec Streamlit :

```bash
streamlit run src/dashboard/app.py
```

### Que se passe-t-il ?

1. **Le terminal affiche** :
   ```

     You can now view your Streamlit app in your browser.

     Local URL: http://localhost:8501

   ```

2. **Dans Antigravity** : Une notification apparaît pour ouvrir l'aperçu web

3. **Cliquez sur "Open in Browser"** ou **"Preview"** pour voir le dashboard

4. **Alternative** : Antigravity peut afficher un onglet "Web Preview" à côté de votre code

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🚀 Antigravity                                                         │
├────────────────┬────────────────────────────────────────────────────────┤
│                │  [Code]  [Web Preview]  ← Cliquez ici                 │
│  📁 EXPLORER   │  ┌────────────────────────────────────────────────┐   │
│                │  │  🏆 Jira AI Co-Pilot                           │   │
│  ▼ nabda       │  │                                                │   │
│    ▶ src/      │  │  Bienvenue !                                   │   │
│    📄 .env     │  │  [Dashboard s'affiche ici]                     │   │
│                │  │                                                │   │
│                │  └────────────────────────────────────────────────┘   │
├────────────────┴────────────────────────────────────────────────────────┤
│  TERMINAL                                                               │
│  streamlit run src/dashboard/app.py                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Aperçu du dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🏆 Jira AI Co-Pilot                                          [Menu ▼] │
├───────────────┬─────────────────────────────────────────────────────────┤
│               │                                                         │
│  📊 Overview  │     Bienvenue dans votre Dashboard !                   │
│  📋 Board     │                                                         │
│  🏃 Sprint    │     ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  👥 Team      │     │ 42      │  │ 8       │  │ 95%     │             │
│  🎯 Predict   │     │ Issues  │  │ Sprint  │  │ Santé   │             │
│  🎲 Forecast  │     └─────────┘  └─────────┘  └─────────┘             │
│  🕯️ Burnout   │                                                         │
│  🌅 Morning   │     [Graphique de vélocité]                            │
│               │                                                         │
└───────────────┴─────────────────────────────────────────────────────────┘
```

### Arrêter l'application

Pour arrêter le dashboard :
- Retournez dans le terminal
- Appuyez sur `Ctrl + C`

---

## 9. Utilisation du Dashboard

### Navigation

Le menu de gauche contient toutes les pages disponibles :

| Page | Icône | Description |
|------|-------|-------------|
| **Executive Cockpit** | 🏆 | Vue d'ensemble pour les dirigeants |
| **Overview** | 📊 | Statistiques générales du projet |
| **Board** | 📋 | Vue Kanban de vos tickets |
| **Sprint Health** | 🏃 | Santé du sprint en cours |
| **Team Workload** | 👥 | Charge de travail par personne |
| **Predictions** | 🎯 | Prédictions IA sur les délais |
| **Strategic Gap** | 🎯 | Alignement stratégique |
| **Burnout Risk** | 🕯️ | Détection des risques de burnout |
| **Delivery Forecast** | 🎲 | Prévisions de livraison Monte Carlo |
| **Good Morning** | 🌅 | Briefing quotidien automatique |
| **The Oracle** | 🔮 | Questions/réponses IA |
| **Project Weather** | 🌀 | Météo santé du projet |

### Page "Good Morning" - Votre briefing quotidien

Cette page génère automatiquement un résumé de ce qui s'est passé et ce qui est prévu :

```
┌─────────────────────────────────────────────────────────────┐
│  🌅 Bonjour Jean !                         Lundi 13 janvier │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📌 Résumé de hier                                          │
│  • 5 tickets terminés par l'équipe                          │
│  • 2 nouveaux bugs signalés                                 │
│  • Sprint à 73% de complétion                               │
│                                                             │
│  ⚠️ Points d'attention                                      │
│  • Marie a 8 tickets assignés (charge élevée)               │
│  • Le ticket PROJ-234 est bloqué depuis 3 jours            │
│                                                             │
│  🎯 Priorités du jour                                       │
│  • Débloquer PROJ-234 (critique)                           │
│  • Code review de PROJ-301                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Page "Burnout Risk" - Protégez votre équipe

Cette page analyse la charge de travail et détecte les signes de surmenage :

```
┌─────────────────────────────────────────────────────────────┐
│  🕯️ Analyse du Risque de Burnout                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Membre        │ Risque  │ Indicateurs                      │
│  ───────────────────────────────────────────────────────    │
│  Marie D.      │ 🔴 HAUT │ 12 tickets, +40% vs moyenne      │
│  Pierre L.     │ 🟡 MOYEN│ Heures sup, weekends             │
│  Sophie M.     │ 🟢 BAS  │ Charge équilibrée                │
│                                                             │
│  💡 Recommandations                                          │
│  • Redistribuer 3 tickets de Marie vers Sophie              │
│  • Planifier un 1:1 avec Pierre                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Rafraîchir les données

Pour mettre à jour les données depuis Jira :

1. Retournez dans le terminal
2. Tapez :
   ```bash
   jira-copilot sync full
   ```
3. Rafraîchissez la page du dashboard (F5 ou bouton actualiser)

---

## 10. Ajouter un nouveau module

### Structure des modules

Les modules sont organisés ainsi :

```
src/
├── features/          ← Logique métier (calculs, analyses)
│   ├── burnout_models.py
│   ├── delivery_forecast.py
│   └── votre_nouveau_module.py  ← Nouveau !
│
└── dashboard/
    └── pages/         ← Pages du dashboard
        ├── 8_🕯️_Burnout_Risk.py
        └── XX_📊_Votre_Page.py    ← Nouveau !
```

### Exemple : Créer un module "Analyse de la Qualité"

#### Étape 1 - Créer le fichier de logique

1. **Dans Antigravity**, dans l'Explorer à gauche, faites un clic droit sur le dossier `src/features/`
2. Sélectionnez **"New File"** (Nouveau fichier)
3. Nommez-le : `quality_analyzer.py`
4. Copiez ce contenu :

```python
"""
Module d'analyse de la qualité des tickets.
"""
from pathlib import Path
import pandas as pd
from src.data.loader import DataLoader


class QualityAnalyzer:
    """Analyse la qualité des tickets Jira."""

    def __init__(self, db_path: str = "data/jira.duckdb"):
        """
        Initialise l'analyseur.

        Args:
            db_path: Chemin vers la base de données
        """
        self.loader = DataLoader(db_path)

    def analyze_descriptions(self) -> pd.DataFrame:
        """
        Analyse la qualité des descriptions de tickets.

        Returns:
            DataFrame avec les scores de qualité
        """
        # Récupérer les tickets
        issues = self.loader.get_issues()

        # Calculer des métriques de qualité
        results = []
        for _, issue in issues.iterrows():
            description = issue.get('description', '') or ''

            # Score de qualité (exemple simple)
            score = 0

            # La description existe
            if len(description) > 0:
                score += 20

            # La description est suffisamment longue
            if len(description) > 50:
                score += 20

            # La description contient des critères d'acceptation
            if 'critère' in description.lower() or 'acceptance' in description.lower():
                score += 30

            # La description contient des étapes
            if any(word in description.lower() for word in ['étape', 'step', '1.', '2.']):
                score += 30

            results.append({
                'key': issue['key'],
                'summary': issue['summary'],
                'quality_score': score,
                'description_length': len(description)
            })

        return pd.DataFrame(results)

    def get_low_quality_tickets(self, threshold: int = 50) -> pd.DataFrame:
        """
        Retourne les tickets avec une qualité insuffisante.

        Args:
            threshold: Score minimum acceptable

        Returns:
            DataFrame des tickets sous le seuil
        """
        df = self.analyze_descriptions()
        return df[df['quality_score'] < threshold].sort_values('quality_score')
```

5. **Sauvegardez** (`Ctrl + S` - Antigravity sauvegarde aussi automatiquement)

#### Étape 2 - Créer la page du dashboard

1. **Dans Antigravity**, faites un clic droit sur le dossier `src/dashboard/pages/`
2. Sélectionnez **"New File"** (Nouveau fichier)
3. Nommez-le : `25_📝_Quality_Analysis.py`

   > **Note** : Le numéro au début (25) détermine l'ordre dans le menu. L'emoji rend le menu plus visuel.

4. Copiez ce contenu :

```python
"""
Page d'analyse de la qualité des tickets.
"""
import streamlit as st
import plotly.express as px
from src.features.quality_analyzer import QualityAnalyzer

# Configuration de la page
st.set_page_config(
    page_title="Analyse Qualité",
    page_icon="📝",
    layout="wide"
)

# Titre
st.title("📝 Analyse de la Qualité des Tickets")
st.markdown("---")

# Initialiser l'analyseur
@st.cache_data(ttl=300)  # Cache de 5 minutes
def load_quality_data():
    analyzer = QualityAnalyzer()
    return analyzer.analyze_descriptions()

# Charger les données
with st.spinner("Analyse en cours..."):
    df = load_quality_data()

# Métriques globales
col1, col2, col3 = st.columns(3)

with col1:
    avg_score = df['quality_score'].mean()
    st.metric(
        label="Score moyen de qualité",
        value=f"{avg_score:.0f}/100"
    )

with col2:
    low_quality = len(df[df['quality_score'] < 50])
    st.metric(
        label="Tickets à améliorer",
        value=low_quality,
        delta=f"-{low_quality}" if low_quality > 0 else "0",
        delta_color="inverse"
    )

with col3:
    high_quality = len(df[df['quality_score'] >= 80])
    st.metric(
        label="Tickets bien documentés",
        value=high_quality
    )

st.markdown("---")

# Graphique de distribution
st.subheader("Distribution des scores de qualité")
fig = px.histogram(
    df,
    x='quality_score',
    nbins=20,
    title="Répartition des scores",
    labels={'quality_score': 'Score de qualité', 'count': 'Nombre de tickets'}
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Tableau des tickets à améliorer
st.subheader("🚨 Tickets nécessitant une meilleure description")

low_quality_df = df[df['quality_score'] < 50].sort_values('quality_score')

if len(low_quality_df) > 0:
    st.dataframe(
        low_quality_df[['key', 'summary', 'quality_score', 'description_length']],
        use_container_width=True,
        column_config={
            'key': 'Ticket',
            'summary': 'Résumé',
            'quality_score': st.column_config.ProgressColumn(
                'Score',
                min_value=0,
                max_value=100,
                format="%d"
            ),
            'description_length': 'Longueur description'
        }
    )
else:
    st.success("✅ Tous les tickets ont une description de qualité acceptable !")

# Conseils
with st.expander("💡 Comment améliorer la qualité des tickets ?"):
    st.markdown("""
    ### Bonnes pratiques pour les descriptions de tickets

    1. **Description claire** : Expliquez le contexte et l'objectif
    2. **Critères d'acceptation** : Listez ce qui doit être fait pour considérer le ticket comme terminé
    3. **Étapes de reproduction** (pour les bugs) : Détaillez comment reproduire le problème
    4. **Captures d'écran** : Ajoutez des visuels si pertinent
    5. **Liens utiles** : Référencez la documentation ou les tickets liés
    """)
```

5. **Sauvegardez** (`Ctrl + S`)

#### Étape 3 - Tester votre nouveau module

1. **Relancez le dashboard** :
   ```bash
   streamlit run src/dashboard/app.py
   ```

2. **Cherchez votre nouvelle page** dans le menu de gauche : "📝 Quality Analysis"

3. **Cliquez dessus** pour voir votre module en action !

### Structure recommandée pour un nouveau module

```
┌─────────────────────────────────────────────────────────────┐
│  Nouveau Module                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. src/features/mon_module.py                              │
│     └─ Classe avec la logique métier                        │
│     └─ Méthodes pour calculer/analyser                      │
│     └─ Retourne des DataFrames                              │
│                                                             │
│  2. src/dashboard/pages/XX_📊_Ma_Page.py                    │
│     └─ Import du module de features                         │
│     └─ Configuration st.set_page_config()                   │
│     └─ Titre et métriques st.metric()                       │
│     └─ Graphiques Plotly                                    │
│     └─ Tableaux st.dataframe()                              │
│                                                             │
│  3. tests/test_features/test_mon_module.py (optionnel)      │
│     └─ Tests unitaires                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Dépannage - Problèmes courants

### Problème : "Module not found"

**Symptôme** : `ModuleNotFoundError: No module named 'streamlit'`

**Solution dans Antigravity** :
```bash
pip install -e ".[dev]"
```

### Problème : Erreur de connexion Jira

**Symptôme** : `JiraAuthenticationError` ou `401 Unauthorized`

**Solutions** :
1. Vérifiez votre URL Jira (doit se terminer par `.atlassian.net`)
2. Vérifiez que l'email est correct
3. Régénérez un nouveau token API sur https://id.atlassian.com/manage-profile/security/api-tokens
4. Vérifiez que vous avez accès au projet spécifié

### Problème : Le dashboard ne s'ouvre pas dans Antigravity

**Symptôme** : Pas de preview après `streamlit run`

**Solutions** :
1. Cherchez la notification "Open Preview" dans Antigravity
2. Cliquez sur l'onglet "Web Preview" en haut
3. Si rien ne marche, utilisez le port forwarding d'Antigravity

### Problème : Le workspace Antigravity est lent

**Symptôme** : L'éditeur ou le terminal répond lentement

**Solutions** :
1. Fermez les autres onglets de navigateur
2. Rafraîchissez la page (F5)
3. Redémarrez le workspace (Menu → Restart Workspace)

### Problème : Erreur Google API

**Symptôme** : `google.api_core.exceptions.InvalidArgument`

**Solutions** :
1. Vérifiez que la clé API est correcte dans `.env`
2. Vérifiez que l'API Gemini est activée dans votre projet Google Cloud
3. Attendez quelques minutes si vous venez de créer la clé

### Problème : Base de données corrompue

**Symptôme** : Erreurs DuckDB ou données incohérentes

**Solution** :
```bash
rm data/jira.duckdb
jira-copilot init
jira-copilot sync full
```

### Problème : Fichier .env non reconnu

**Symptôme** : L'application ne trouve pas vos configurations

**Solutions** :
1. Vérifiez que le fichier s'appelle bien `.env` (avec le point au début)
2. Vérifiez qu'il est à la racine du projet (pas dans un sous-dossier)
3. Redémarrez le terminal dans Antigravity

---

## 12. Glossaire

### Termes généraux

| Terme | Explication |
|-------|-------------|
| **Antigravity** | Éditeur de code cloud de Google, accessible via navigateur |
| **Workspace** | Espace de travail dans Antigravity contenant votre projet |
| **API** | Interface de programmation - permet à deux logiciels de communiquer |
| **Token** | Clé secrète qui authentifie votre accès |
| **Terminal** | Zone pour taper des commandes textuelles (en bas d'Antigravity) |
| **Dashboard** | Tableau de bord visuel |
| **Dépendance** | Logiciel tiers dont l'application a besoin |
| **Cloud** | Service accessible via internet, pas installé sur votre ordinateur |

### Termes Python

| Terme | Explication |
|-------|-------------|
| **pip** | Gestionnaire de packages Python (installe les bibliothèques) |
| **Module** | Fichier Python contenant du code réutilisable |
| **Package** | Collection de modules |
| **Streamlit** | Framework Python pour créer des dashboards web |

### Termes Jira

| Terme | Explication |
|-------|-------------|
| **Issue/Ticket** | Tâche, bug, ou story dans Jira |
| **Sprint** | Période de travail fixe (généralement 2 semaines) |
| **Board** | Tableau Kanban ou Scrum |
| **Story Points** | Estimation de la complexité d'une tâche |
| **JQL** | Jira Query Language - langage pour rechercher des tickets |

### Termes Streamlit

| Terme | Explication |
|-------|-------------|
| **st.** | Préfixe pour les fonctions Streamlit |
| **Widget** | Élément interactif (bouton, slider, etc.) |
| **Cache** | Stockage temporaire pour accélérer |
| **Page** | Une vue dans l'application multi-pages |

---

## Besoin d'aide ?

### Ressources supplémentaires

- **Documentation Streamlit** : https://docs.streamlit.io
- **Documentation Jira API** : https://developer.atlassian.com/cloud/jira/platform/rest/v3/
- **Google AI Studio** : https://aistudio.google.com

### Support

Si vous rencontrez un problème non couvert par ce guide :

1. Vérifiez les logs dans le terminal
2. Cherchez l'erreur sur Google
3. Contactez l'administrateur du projet

---

## Mémo des commandes (à taper dans le terminal Antigravity)

```bash
# ═══════════════════════════════════════════════════════════════
# INSTALLATION (une seule fois)
# ═══════════════════════════════════════════════════════════════

# Installer les dépendances
pip install -e ".[dev]"

# Initialiser la base de données
jira-copilot init

# ═══════════════════════════════════════════════════════════════
# UTILISATION QUOTIDIENNE
# ═══════════════════════════════════════════════════════════════

# Synchroniser les données depuis Jira
jira-copilot sync full

# Entraîner les modèles IA (après sync)
jira-copilot train

# Lancer le tableau de bord
jira-copilot dashboard
# ou
streamlit run src/dashboard/app.py

# Vérifier l'état du système
jira-copilot status

# ═══════════════════════════════════════════════════════════════
# RACCOURCIS ANTIGRAVITY
# ═══════════════════════════════════════════════════════════════

# Ouvrir/fermer le terminal
Ctrl + ù  (ou Ctrl + `)

# Sauvegarder
Ctrl + S

# Arrêter une commande en cours
Ctrl + C

# Rechercher dans les fichiers
Ctrl + Shift + F
```

---

## Résumé rapide - Démarrage en 5 étapes

| Étape | Action | Commande/URL |
|-------|--------|--------------|
| 1 | Ouvrir Antigravity | `https://antigravity.google/` |
| 2 | Importer le projet | Import from GitHub |
| 3 | Configurer `.env` | Copier `.env.example` → `.env` et remplir |
| 4 | Installer | `pip install -e ".[dev]"` |
| 5 | Lancer | `jira-copilot init && jira-copilot sync full && jira-copilot dashboard` |

---

**Félicitations !** Vous avez configuré avec succès Jira AI Co-Pilot !

Si vous suivez ce guide étape par étape, vous aurez une application fonctionnelle qui analyse vos données Jira et vous aide à mieux gérer vos projets.

### Besoin d'aide ?

- **Documentation Antigravity** : https://antigravity.google/docs
- **Documentation Streamlit** : https://docs.streamlit.io
- **Créer un token Jira** : https://id.atlassian.com/manage-profile/security/api-tokens
- **Créer une clé Gemini** : https://aistudio.google.com/app/apikey

---

*Guide créé pour Jira AI Co-Pilot v0.1.0*
*Dernière mise à jour : Janvier 2026*
*Optimisé pour Google Antigravity*
