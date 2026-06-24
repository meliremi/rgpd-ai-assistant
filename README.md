# Assistant Juridique RGPD/CNIL
**Projet M2 Deep Learning & MLOps — ZenML + RAG + QLoRA**
*Remila Melissa & Benzouaoua Selma — Mars 2026*

---

## Architecture

```
PDF RGPD  -->  [Step 1] Chunking  -->  [Step 2] FAISS Index
                                              |
                                       [Step 3] RAGAS Eval
                                              |
                                       [Step 4] Trigger?
                                              |
                                       [Step 5] QLoRA (si besoin)
                                              |
                                       [Step 6] Eval Modele
                                              |
                                       [Step 7] Model Registry
                                              |
                                       [Step 8] FastAPI Deploy
```

**Stack :** ZenML · sentence-transformers · FAISS · TinyLlama/Mistral-7B · RAGAS · QLoRA · FastAPI

---

## Installation

```bash
git clone <repo>
cd rgpd-ai-assistant

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

---

## Executer le Pipeline

### Option 1 — Pipeline complet ZenML (8 steps)
```bash
python pipeline/run_pipeline.py
```

### Option 2 — RAG seulement (sans ZenML, rapide)
```bash
python pipeline/run_pipeline.py --rag-only
```

### Option 3 — Tester le RAG interactivement
```bash
python pipeline/run_pipeline.py --test-rag
```

---

## Demarrer l'API

```bash
uvicorn serving.api:app --reload --port 8000
```

- Interface : http://localhost:8000
- Swagger UI : http://localhost:8000/docs

**Exemple de requete :**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quand faut-il nommer un DPO ?"}'
```

**Reponse :**
```json
{
  "question": "Quand faut-il nommer un DPO ?",
  "answer": "La designation d'un DPO est obligatoire (Art. 37 RGPD) pour les autorites publiques...",
  "sources": ["chunk1...", "chunk2...", "chunk3..."],
  "confidence": 0.84,
  "latency_ms": 320
}
```

---

## Structure du Projet

```
rgpd-ai-assistant/
├── data_preparation/
│   ├── extract_text.py     # Step 1 : extraction PDF
│   ├── chunking.py         # Step 2 : chunking semantique (400 mots, overlap 60)
│   ├── generate_qa.py      # Step 3 : generation dataset Q/R
│   ├── clean_filter.py     # Step 4 : nettoyage paires Q/R
│   ├── alpaca_format.py    # Step 5 : format Alpaca pour QLoRA
│   └── split_export.py     # Step 6 : split train/val
├── rag/
│   ├── embeddings.py       # Embeddings L2-normalises (MiniLM-L6-v2)
│   ├── retriever.py        # Index FAISS IndexFlatIP + search_with_scores()
│   └── prompt_templates.py # Templates de prompt RAG anti-hallucination
├── pipeline/
│   ├── zenml_pipeline.py   # 8 steps ZenML orchestrees
│   └── run_pipeline.py     # Point d'entree
├── evaluation/
│   └── ragas_eval.py       # RAGAS : Faithfulness / Relevancy / Recall
├── training/
│   ├── qlora_trainer.py    # QLoRA 4-bit NF4 (Mistral-7B / TinyLlama)
│   └── anti_forgetting.py  # Replay 20% donnees historiques
├── serving/
│   └── api.py              # FastAPI : /ask /health /model/info /feedback
├── artifacts/              # Artefacts generes (crees par le pipeline)
│   ├── chunks.json         # Chunks du PDF RGPD
│   ├── embeddings.npy      # Matrice embeddings (N x 384)
│   ├── faiss_index.bin     # Index FAISS binaire
│   ├── train.json          # Dataset train Alpaca
│   ├── val.json            # Dataset validation
│   └── feedback.jsonl      # Feedbacks utilisateurs
└── requirements.txt
```

---

## RAG — Details Techniques

### Ou sont stockees les donnees ?
| Artefact | Chemin | Format | Contenu |
|---|---|---|---|
| Texte brut | artifacts/raw_text.txt | TXT | Texte nettoye du PDF RGPD |
| Chunks | artifacts/chunks.json | JSON | Liste de N chunks (400 mots) |
| Embeddings | artifacts/embeddings.npy | NPY | Matrice (N x 384) float32 |
| Index FAISS | artifacts/faiss_index.bin | BIN | Index IndexFlatIP serialise |

### Quelle fonction de retrieval ?
`search_with_scores(query, index, chunks, k=5)` dans `rag/retriever.py`

### Comment on cherche ?
1. `embed_query(question)` — encode la question en vecteur 384-dims L2-normalise
2. `index.search(query_vector, k)` — produit scalaire avec tous les vecteurs (= cosine sur normalises)
3. Retourne les k indices + scores

### Comment on classe la pertinence ?
- **Score cosine** (0 a 1) : plus proche de 1 = plus pertinent
- **Seuil** : 0.30 minimum pour etre inclus dans les resultats
- **IndexFlatIP** : tri automatique par score decroissant par FAISS

---

## ZenML — Cloud / Docker

### Configuration Stack Local
```bash
zenml init
zenml stack register local_stack \
  -a default \
  -o default
zenml stack set local_stack
python pipeline/run_pipeline.py
```

### Configuration Stack Docker
```bash
zenml integration install docker
zenml stack register docker_stack \
  -a default \
  -o default \
  --container-registry=default
zenml stack set docker_stack
python pipeline/run_pipeline.py
```

### Configuration Stack GCP (Cloud)
```bash
zenml integration install gcp
zenml stack register gcp_stack \
  -a gcs://mon-bucket/artifacts \
  -o vertex_orchestrator \
  --container-registry=gcr
zenml stack set gcp_stack
python pipeline/run_pipeline.py
```

---

## Fine-tuning QLoRA (Colab GPU)

Le fine-tuning necessite un GPU. Utiliser Google Colab T4 :

1. Ouvrir Colab : https://colab.research.google.com
2. Runtime -> Change runtime type -> GPU T4
3. Cloner le repo et installer les dependances
4. Lancer : `python training/qlora_trainer.py`

**Configuration QLoRA :**
- Quantization 4-bit NF4 (bitsandbytes)
- LoRA : r=16, alpha=32, dropout=0.05
- Cibles : q_proj, k_proj, v_proj, o_proj
- Anti-forgetting : replay 20% donnees precedentes

---

## Evaluation RAGAS

| Metrique | Seuil | Description |
|---|---|---|
| Faithfulness | >= 0.75 | Reponse ancree dans les sources ? |
| Answer Relevancy | >= 0.70 | Reponse repond a la question ? |
| Context Recall | >= 0.65 | Contexte couvre la reponse attendue ? |

Si une metrique est sous le seuil -> Step 4 declenche automatiquement le fine-tuning QLoRA.

---

## Auteurs
- **Remila Melissa** — Pipeline ZenML, QLoRA, API FastAPI
- **Benzouaoua Selma** — Data preparation, RAG, RAGAS evaluation
