# RGPD AI Assistant 🛡️

Assistant intelligent basé sur le RAG (Retrieval Augmented Generation) pour répondre aux questions sur le RGPD.


```text
rgpd-ai-assistant/
├── 📁 .venv/                # Environnement virtuel (Masqué par défaut)
├── 📁 data/                 #  Déposez vos PDF sources ici
│   └── document_rgpd.pdf
├── 📁 data_preparation/     #  Étape 1 : Scripts de traitement
│   └── extract_text.py      # Extraction de texte (pdfplumber)
├── 📁 rag/                  #  Étape 2 : Moteur d'intelligence
│   ├── embeddings.py        # Vecteurs (Sentence-Transformers)
│   └── retriever.py         # Recherche sémantique (FAISS)
├── 📁 serving/              #  Étape 3 : Serveur API
│   └── api.py               # FastAPI & TinyLlama
├── 📄 requirements.txt      # Liste des bibliothèques
└── 📄 README.md             # Guide d'utilisation 
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
