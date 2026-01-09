# Guide Complet d'Installation et d'Utilisation
## Jira AI Co-Pilot - Votre Assistant Intelligent de Gestion de Projet

---

> **Pour qui est ce guide ?**
> Ce guide est conçu pour les utilisateurs **débutants** qui n'ont aucune expérience technique préalable. Chaque étape est expliquée en détail avec des captures d'écran et des conseils.

---

## Table des Matières

1. [Introduction - Qu'est-ce que Jira AI Co-Pilot ?](#1-introduction)
2. [Prérequis - Ce dont vous avez besoin](#2-prérequis)
3. [Étape 1 - Installer l'éditeur de code (Antigravity/VS Code)](#3-étape-1---installer-léditeur-de-code)
4. [Étape 2 - Installer Python](#4-étape-2---installer-python)
5. [Étape 3 - Télécharger le projet](#5-étape-3---télécharger-le-projet)
6. [Étape 4 - Créer un environnement virtuel](#6-étape-4---créer-un-environnement-virtuel)
7. [Étape 5 - Installer les dépendances](#7-étape-5---installer-les-dépendances)
8. [Étape 6 - Configurer votre token Jira](#8-étape-6---configurer-votre-token-jira)
9. [Étape 7 - Configurer Google Gemini (IA)](#9-étape-7---configurer-google-gemini-ia)
10. [Étape 8 - Lancer l'application](#10-étape-8---lancer-lapplication)
11. [Utilisation du Dashboard](#11-utilisation-du-dashboard)
12. [Ajouter un nouveau module](#12-ajouter-un-nouveau-module)
13. [Dépannage - Problèmes courants](#13-dépannage---problèmes-courants)
14. [Glossaire - Termes techniques expliqués](#14-glossaire)

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
- Un ordinateur (Windows, Mac, ou Linux)
- Une connexion internet stable
- Au moins 4 Go de RAM disponible
- 2 Go d'espace disque libre

### Comptes nécessaires
- Un compte **Jira Cloud** (Atlassian)
- Un compte **Google** (pour l'IA Gemini - gratuit)

### Temps estimé
| Étape | Durée |
|-------|-------|
| Installation de l'éditeur | 10 minutes |
| Installation de Python | 10 minutes |
| Configuration du projet | 20 minutes |
| Configuration Jira | 15 minutes |
| **Total** | **~1 heure** |

---

## 3. Étape 1 - Installer l'éditeur de code

### Qu'est-ce qu'un éditeur de code ?

Un **éditeur de code** est un programme qui permet de voir et modifier les fichiers du projet. C'est comme Microsoft Word, mais pour le code informatique.

### Nous recommandons : Visual Studio Code (VS Code)

VS Code est **gratuit**, **facile à utiliser**, et très populaire.

### Instructions d'installation

#### Pour Windows :

1. **Ouvrez votre navigateur** (Chrome, Firefox, Edge...)

2. **Allez sur le site officiel** :
   ```
   https://code.visualstudio.com/
   ```

3. **Cliquez sur le bouton bleu "Download for Windows"**

   ```
   ┌────────────────────────────────────────────┐
   │                                            │
   │    [  Download for Windows  ]              │
   │         ↑                                  │
   │    Cliquez ici                             │
   │                                            │
   └────────────────────────────────────────────┘
   ```

4. **Attendez** que le fichier `VSCodeUserSetup-x64-X.XX.X.exe` se télécharge

5. **Double-cliquez** sur le fichier téléchargé

6. **Suivez l'assistant d'installation** :
   - Acceptez les termes de la licence ✓
   - Laissez le dossier par défaut ✓
   - **IMPORTANT** : Cochez ces options :
     - ☑️ "Ajouter à PATH"
     - ☑️ "Ajouter l'action 'Ouvrir avec Code' au menu contextuel"
   - Cliquez sur "Installer"

7. **Redémarrez votre ordinateur** après l'installation

#### Pour Mac :

1. Allez sur `https://code.visualstudio.com/`
2. Cliquez sur "Download for Mac"
3. Ouvrez le fichier `.zip` téléchargé
4. Glissez l'icône VS Code dans le dossier "Applications"
5. Ouvrez VS Code depuis le Launchpad

#### Pour Linux (Ubuntu/Debian) :

```bash
sudo apt update
sudo apt install code
```

### Vérifier l'installation

1. Ouvrez VS Code (cherchez "Visual Studio Code" dans vos applications)
2. Vous devriez voir cette interface :

```
┌─────────────────────────────────────────────────────────────┐
│  Visual Studio Code                               _ □ X    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐                                              │
│  │ Explorer │    Welcome                                   │
│  │          │                                              │
│  │ (vide)   │    Start                                     │
│  │          │    • New File                                │
│  │          │    • Open Folder                             │
│  │          │                                              │
│  └──────────┘                                              │
└─────────────────────────────────────────────────────────────┘
```

**Félicitations !** L'éditeur est installé.

---

## 4. Étape 2 - Installer Python

### Qu'est-ce que Python ?

**Python** est le langage de programmation utilisé par Jira AI Co-Pilot. Vous devez l'installer pour que l'application fonctionne.

### Quelle version ?

Vous avez besoin de **Python 3.11** ou plus récent.

### Instructions d'installation

#### Pour Windows :

1. **Allez sur le site officiel Python** :
   ```
   https://www.python.org/downloads/
   ```

2. **Cliquez sur "Download Python 3.12.x"** (ou la version la plus récente)

3. **Ouvrez le fichier téléchargé**

4. **TRÈS IMPORTANT - Cochez cette case** :
   ```
   ┌────────────────────────────────────────────────────────┐
   │  Install Python 3.12.x                                 │
   │                                                        │
   │  ☑️ Add python.exe to PATH   ← COCHEZ CETTE CASE !    │
   │                                                        │
   │  [  Install Now  ]                                     │
   │                                                        │
   └────────────────────────────────────────────────────────┘
   ```

   > ⚠️ **ATTENTION** : Si vous oubliez de cocher "Add python.exe to PATH", l'application ne fonctionnera pas !

5. Cliquez sur **"Install Now"**

6. Attendez la fin de l'installation

7. Cliquez sur **"Close"**

#### Pour Mac :

1. Ouvrez le **Terminal** (Recherchez "Terminal" dans Spotlight)

2. Installez Homebrew si ce n'est pas déjà fait :
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. Installez Python :
   ```bash
   brew install python@3.12
   ```

#### Pour Linux :

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

### Vérifier l'installation

1. **Ouvrez un terminal** :
   - Windows : Tapez "cmd" dans la barre de recherche et appuyez sur Entrée
   - Mac/Linux : Ouvrez l'application "Terminal"

2. **Tapez cette commande** et appuyez sur Entrée :
   ```bash
   python --version
   ```

   Ou sur certains systèmes :
   ```bash
   python3 --version
   ```

3. **Vous devriez voir** quelque chose comme :
   ```
   Python 3.12.1
   ```

   > Si vous voyez une version 3.11 ou supérieure, c'est parfait !

### Problème ? Python non reconnu ?

Si vous voyez une erreur comme `'python' n'est pas reconnu...` :

1. **Redémarrez votre ordinateur**
2. Réessayez la commande
3. Si ça ne marche toujours pas, réinstallez Python en vous assurant de cocher "Add to PATH"

---

## 5. Étape 3 - Télécharger le projet

### Option A : Télécharger depuis GitHub (Recommandé)

1. **Allez sur la page GitHub du projet** (l'URL vous sera fournie)

2. **Cliquez sur le bouton vert "Code"**
   ```
   ┌─────────────────────────┐
   │  < > Code  ▼            │
   └─────────────────────────┘
   ```

3. **Cliquez sur "Download ZIP"**
   ```
   ┌─────────────────────────────────┐
   │  Clone                          │
   │  ─────────────────────────────  │
   │  HTTPS   SSH   GitHub CLI       │
   │                                 │
   │  [  Download ZIP  ]  ← Ici      │
   └─────────────────────────────────┘
   ```

4. **Extrayez le fichier ZIP** :
   - Faites un clic droit sur le fichier téléchargé
   - Sélectionnez "Extraire tout..." ou "Extract All..."
   - Choisissez un emplacement facile à retrouver, par exemple :
     ```
     C:\Projets\jira-copilot
     ```
     ou sur Mac :
     ```
     ~/Documents/jira-copilot
     ```

### Option B : Si vous avez Git installé

Ouvrez un terminal et tapez :
```bash
git clone [URL_DU_PROJET]
cd nabda
```

### Ouvrir le projet dans VS Code

1. **Ouvrez VS Code**

2. **Cliquez sur "File" → "Open Folder"** (ou Fichier → Ouvrir un dossier)

3. **Naviguez** jusqu'au dossier extrait et sélectionnez-le

4. **Cliquez sur "Sélectionner un dossier"**

Vous devriez maintenant voir la structure du projet dans la barre latérale gauche :

```
┌─────────────────────────────────────────────────────────────┐
│  EXPLORER                                                   │
├─────────────────────────────────────────────────────────────┤
│  ▼ NABDA                                                    │
│    ▶ config/                                                │
│    ▶ data/                                                  │
│    ▶ models/                                                │
│    ▶ prompts/                                               │
│    ▶ scripts/                                               │
│    ▶ src/                                                   │
│    ▶ tests/                                                 │
│    📄 .env.example                                          │
│    📄 pyproject.toml                                        │
│    📄 README.md                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Étape 4 - Créer un environnement virtuel

### Qu'est-ce qu'un environnement virtuel ?

Un **environnement virtuel** est comme une "bulle" isolée pour votre projet. Il permet d'installer les outils nécessaires sans affecter les autres programmes de votre ordinateur.

### Créer l'environnement virtuel

1. **Ouvrez un terminal dans VS Code** :
   - Menu : `Terminal` → `New Terminal`
   - Ou raccourci : `Ctrl + ù` (Windows) / `Cmd + ù` (Mac)

2. **Assurez-vous d'être dans le bon dossier**. Le terminal devrait afficher quelque chose comme :
   ```
   C:\Projets\jira-copilot>
   ```
   ou
   ```
   ~/Documents/jira-copilot$
   ```

3. **Créez l'environnement virtuel** en tapant cette commande :

   **Windows** :
   ```bash
   python -m venv venv
   ```

   **Mac/Linux** :
   ```bash
   python3 -m venv venv
   ```

   > Cette commande crée un dossier `venv` qui contient l'environnement virtuel.

4. **Activez l'environnement virtuel** :

   **Windows (PowerShell)** :
   ```bash
   .\venv\Scripts\Activate.ps1
   ```

   **Windows (CMD)** :
   ```bash
   venv\Scripts\activate.bat
   ```

   **Mac/Linux** :
   ```bash
   source venv/bin/activate
   ```

5. **Vérifiez que c'est activé** - Vous devriez voir `(venv)` au début de la ligne :
   ```
   (venv) C:\Projets\jira-copilot>
   ```

### Problème avec PowerShell ?

Si vous voyez une erreur sur Windows PowerShell du type "l'exécution de scripts est désactivée" :

1. Ouvrez PowerShell **en tant qu'administrateur**
2. Tapez cette commande :
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. Tapez `O` pour confirmer
4. Réessayez d'activer l'environnement

---

## 7. Étape 5 - Installer les dépendances

### Qu'est-ce qu'une dépendance ?

Les **dépendances** sont des bibliothèques (des outils pré-fabriqués) dont l'application a besoin pour fonctionner. Par exemple : Streamlit (pour le tableau de bord), pandas (pour les données), etc.

### Installer toutes les dépendances

1. **Assurez-vous que l'environnement virtuel est activé** (vous voyez `(venv)` dans le terminal)

2. **Tapez cette commande** :
   ```bash
   pip install -e ".[dev]"
   ```

   > Cette commande lit le fichier `pyproject.toml` et installe automatiquement toutes les dépendances nécessaires.

3. **Attendez** - L'installation peut prendre **5 à 15 minutes** selon votre connexion internet.

   Vous verrez beaucoup de texte défiler, c'est normal :
   ```
   Collecting streamlit>=1.35.0
     Downloading streamlit-1.35.0-py2.py3-none-any.whl (8.5 MB)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.5/8.5 MB 2.1 MB/s
   Collecting pandas>=2.0.0
     ...
   Successfully installed ...
   ```

4. **Vérifiez l'installation** en tapant :
   ```bash
   pip list
   ```

   Vous devriez voir une longue liste de packages installés.

### Liste des principales dépendances installées

| Package | Rôle |
|---------|------|
| `streamlit` | Crée le tableau de bord interactif |
| `pandas` | Manipule les données |
| `jira` | Se connecte à votre Jira |
| `duckdb` | Base de données locale ultra-rapide |
| `lightgbm` | Intelligence artificielle pour les prédictions |
| `google-generativeai` | Connexion à l'IA Gemini de Google |
| `plotly` | Graphiques interactifs |
| `typer` | Interface en ligne de commande |

---

## 8. Étape 6 - Configurer votre token Jira

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

### Étape 6.3 - Créer le fichier .env

Le fichier `.env` contient toutes vos informations de configuration secrètes.

1. **Dans VS Code**, regardez la liste des fichiers à gauche

2. **Trouvez le fichier `.env.example`**

3. **Faites un clic droit** dessus et sélectionnez **"Rename"** (Renommer)

4. **Renommez-le en** `.env` (supprimez `.example`)

   > **Note** : Sur certains systèmes, les fichiers commençant par `.` sont cachés. Dans VS Code, ils sont toujours visibles.

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

7. **Sauvegardez le fichier** : `Ctrl + S` (Windows) ou `Cmd + S` (Mac)

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

## 9. Étape 7 - Configurer Google Gemini (IA)

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

## 10. Étape 8 - Lancer l'application

### Initialiser la base de données

Avant la première utilisation, initialisez la base de données :

1. **Ouvrez un terminal** dans VS Code (`Terminal` → `New Terminal`)

2. **Assurez-vous que l'environnement virtuel est activé** (vous voyez `(venv)`)

3. **Tapez cette commande** :
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
     Network URL: http://192.168.1.XX:8501

   ```

2. **Votre navigateur s'ouvre automatiquement** sur le dashboard

3. **Si le navigateur ne s'ouvre pas**, copiez l'URL `http://localhost:8501` et collez-la dans votre navigateur

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

## 11. Utilisation du Dashboard

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

## 12. Ajouter un nouveau module

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

1. **Dans VS Code**, faites un clic droit sur le dossier `src/features/`
2. Sélectionnez **"New File"**
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

5. **Sauvegardez** (`Ctrl + S`)

#### Étape 2 - Créer la page du dashboard

1. **Dans VS Code**, faites un clic droit sur le dossier `src/dashboard/pages/`
2. Sélectionnez **"New File"**
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

## 13. Dépannage - Problèmes courants

### Problème : "Python n'est pas reconnu"

**Symptôme** : Le terminal affiche `'python' n'est pas reconnu comme commande interne`

**Solutions** :
1. Redémarrez votre ordinateur
2. Réinstallez Python en cochant "Add to PATH"
3. Sur Windows, essayez `py` au lieu de `python`

### Problème : "Module not found"

**Symptôme** : `ModuleNotFoundError: No module named 'streamlit'`

**Solutions** :
1. Vérifiez que l'environnement virtuel est activé (vous voyez `(venv)`)
2. Réinstallez les dépendances :
   ```bash
   pip install -e ".[dev]"
   ```

### Problème : Erreur de connexion Jira

**Symptôme** : `JiraAuthenticationError` ou `401 Unauthorized`

**Solutions** :
1. Vérifiez votre URL Jira (doit se terminer par `.atlassian.net`)
2. Vérifiez que l'email est correct
3. Régénérez un nouveau token API
4. Vérifiez que vous avez accès au projet spécifié

### Problème : Le dashboard ne s'ouvre pas

**Symptôme** : Pas de fenêtre de navigateur après `streamlit run`

**Solutions** :
1. Ouvrez manuellement `http://localhost:8501` dans votre navigateur
2. Vérifiez qu'aucun autre programme n'utilise le port 8501
3. Essayez avec un port différent :
   ```bash
   streamlit run src/dashboard/app.py --server.port 8080
   ```

### Problème : "Permission denied" sur Mac/Linux

**Symptôme** : Erreur de permission lors de l'exécution

**Solution** :
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

### Problème : Erreur Google API

**Symptôme** : `google.api_core.exceptions.InvalidArgument`

**Solutions** :
1. Vérifiez que la clé API est correcte dans `.env`
2. Vérifiez que l'API Gemini est activée dans votre projet Google Cloud
3. Attendez quelques minutes si vous venez de créer la clé

### Problème : Base de données corrompue

**Symptôme** : Erreurs DuckDB ou données incohérentes

**Solution** :
1. Supprimez la base de données :
   ```bash
   rm data/jira.duckdb
   ```
2. Réinitialisez :
   ```bash
   jira-copilot init
   jira-copilot sync full
   ```

---

## 14. Glossaire

### Termes généraux

| Terme | Explication |
|-------|-------------|
| **API** | Interface de programmation - permet à deux logiciels de communiquer |
| **Token** | Clé secrète qui authentifie votre accès |
| **Terminal** | Fenêtre pour taper des commandes textuelles |
| **Dashboard** | Tableau de bord visuel |
| **Dépendance** | Logiciel tiers dont l'application a besoin |

### Termes Python

| Terme | Explication |
|-------|-------------|
| **pip** | Gestionnaire de packages Python (installe les bibliothèques) |
| **venv** | Environnement virtuel isolé |
| **Module** | Fichier Python contenant du code réutilisable |
| **Package** | Collection de modules |

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

## Mémo des commandes

```bash
# Activer l'environnement virtuel
# Windows PowerShell :
.\venv\Scripts\Activate.ps1

# Windows CMD :
venv\Scripts\activate.bat

# Mac/Linux :
source venv/bin/activate

# Commandes principales
jira-copilot init          # Initialiser la base de données
jira-copilot sync full     # Synchroniser toutes les données Jira
jira-copilot train         # Entraîner les modèles IA
jira-copilot dashboard     # Lancer le tableau de bord
jira-copilot status        # Vérifier l'état du système

# Lancer directement Streamlit
streamlit run src/dashboard/app.py

# Arrêter l'application
Ctrl + C
```

---

**Félicitations !** 🎉 Vous avez configuré avec succès Jira AI Co-Pilot !

Si vous suivez ce guide étape par étape, vous aurez une application fonctionnelle qui analyse vos données Jira et vous aide à mieux gérer vos projets.

---

*Guide créé pour Jira AI Co-Pilot v0.1.0*
*Dernière mise à jour : Janvier 2026*
