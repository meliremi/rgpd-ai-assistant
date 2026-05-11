# RGPD AI Assistant 🛡️

Assistant intelligent basé sur une architecture RAG (Retrieval Augmented Generation) permettant de répondre à des questions sur le RGPD de manière contextualisée à partir de documents juridiques.


```text
PROJET RGPD 
└── 📁 rgpd-ai-assistant/
    |--.zen/                     # Configuration ZenML
    ├── .venv/                  # Environnement virtuel
    ├── 📁 data_preparation/       # Scripts de traitement de données
    │   ├── 📄 alpaca_format.py
    │   ├── 📄 chunking.py
    │   ├── 📄 clean_filter.py
    │   ├── 📄 dataset_train.json
    │   ├── 📄 dataset_val.json
    │   ├── 📄 extract_text.py
    │   ├── 📄 generate_qa.py
    │   ├── 📄 RGPD.pdf            # Source PDF originale
    │   └── 📄 split_export.py
    ├── 📁 evaluation/             # Tests de performance
    │   └── 📄 ragas_eval.py
    ├── 📁  frontend/               # Interface utilisateur
    │   └── 📄 index.html
    ├── 📁 pipeline/               # Orchestration du workflow
    │   └── 📄 run_pipeline.py
    ├── 📁 rag/                    # Logique du Retrieval Augmented Generation
    │   ├── 📄 embeddings.py
    │   ├── 📄 prompt_templates.py
    │   └── 📄 retriever.py
    ├──📁  serving/                # Déploiement et API
    │   └── api.py
    ├──📁  training/               # (Dossier d'entraînement, contenu non visible)
    ├── .gitignore
    ├── 📄dataset_rgpd_50.json    # Dataset final ou de test
    ├── README.md               # Documentation du projet
    ├── 📄 requirements.txt        # Dépendances Python
    └── 📄rgpd_text.txt           # Extraction plein texte du RGPD
```
## 📋 Prérequis

*   Python 3.11+
*   Un environnement virtuel activé (`.venv`)
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

## 🚀 Installation

1. **Cloner le projet** :
   ```bash
   git clone <url-du-repo>
   cd rgpd-ai-assistant
pip install fastapi uvicorn transformers torch sentence-transformers faiss-cpu pdfplumber
*(Note : `pdfplumber` est nécessaire pour l'extraction des textes).*

## 🏗️ Préparation des données

Avant de lancer l'API, vous devez extraire le texte des documents et générer les embeddings :

1. **Extraction du texte** :
   ```bash
   python data_preparation/extract_text.py
## 🌐 Lancement de l'API

Pour démarrer le serveur de l'assistant (FastAPI), utilisez **uvicorn** à la racine du projet :

```bash
uvicorn serving.api:app --reload
API sera disponible sur : http://127.0.0.1:8000

Documentation interactive (Swagger) : http://127.0.0.1:8000/docs

🛠️ Utilisation (Endpoint principal)
Poser une question
URL : POST /ask

Corps de la requête (JSON) :
{
  "question": "Quels sont les droits d'une personne concernée ?"
}
