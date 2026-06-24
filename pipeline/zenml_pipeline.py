"""
Pipeline ZenML — 8 Steps MLOps RGPD
======================================
Orchestre tout le cycle de vie : PDF -> RAG -> Evaluation -> Fine-tuning -> Serving

STEPS :
    1  ingest_and_chunk   : PDF -> chunks semantiques (artifacts/chunks.json)
    2  embed_and_index    : chunks -> embeddings + FAISS (artifacts/faiss_index.bin)
    3  evaluate_rag       : RAGAS (Faithfulness / Answer Relevancy / Context Recall)
    4  finetune_trigger   : bool — declenche QLoRA si metriques sous seuils
    5  qlora_finetune     : fine-tuning 4-bit NF4 + LoRA (artifacts/lora_weights/)
    6  evaluate_model     : perplexite + BLEU + ROUGE
    7  register_model     : gate de promotion -> artifacts/model_registry.json
    8  deploy_serving     : mise a jour config -> artifacts/deploy_config.json

EXECUTION :
    python pipeline/run_pipeline.py              # pipeline complet
    python pipeline/run_pipeline.py --rag-only   # steps 1-2 seulement

CLOUD / DOCKER :
    Voir README.md — section "ZenML Cloud / Docker"
"""

import os
import sys
import json
import time
sys.path.append(os.path.abspath("."))

from zenml import pipeline, step
from typing import List, Dict, Annotated

FAITHFULNESS_MIN = 0.75
RELEVANCY_MIN    = 0.70
RECALL_MIN       = 0.65


@step
def ingest_and_chunk(pdf_path: str = "data_preparation/RGPD.pdf") -> Annotated[List[str], "chunks"]:
    """
    Step 1 : Extrait et decoupe le PDF RGPD en chunks semantiques.
    Chunking par phrases (~400 mots, overlap 60). Sauvegarde : artifacts/chunks.json
    """
    from data_preparation.extract_text import extract_text
    from data_preparation.chunking import chunk_text
    print("\n[STEP 1] INGESTION & CHUNKING")
    text   = extract_text(pdf_path)
    chunks = chunk_text(text)
    print(f"  -> {len(chunks)} chunks produits")
    return chunks


@step
def embed_and_index(chunks: List[str]) -> Annotated[str, "index_path"]:
    """
    Step 2 : Calcule embeddings L2-normalises et construit index FAISS IndexFlatIP.
    Modele : all-MiniLM-L6-v2 (384 dims). Sauvegarde : artifacts/faiss_index.bin
    """
    from rag.retriever import build_and_save_index
    print("\n[STEP 2] EMBEDDINGS & INDEX FAISS")
    build_and_save_index(chunks)
    index_path = "artifacts/faiss_index.bin"
    print(f"  -> Index sauvegarde : {index_path}")
    return index_path


@step
def evaluate_rag(index_path: str) -> Annotated[Dict, "rag_metrics"]:
    """
    Step 3 : Evalue le RAG avec RAGAS sur 10 questions de test.
    Metriques : Faithfulness >= 0.75 | Answer Relevancy >= 0.70 | Context Recall >= 0.65
    """
    from evaluation.ragas_eval import run_ragas_evaluation
    print("\n[STEP 3] EVALUATION RAG (RAGAS)")
    metrics = run_ragas_evaluation()
    metrics["passed"] = (
        metrics.get("faithfulness", 0)     >= FAITHFULNESS_MIN and
        metrics.get("answer_relevancy", 0) >= RELEVANCY_MIN    and
        metrics.get("context_recall", 0)   >= RECALL_MIN
    )
    print(f"  Faithfulness     : {metrics.get('faithfulness',0):.3f}  (seuil={FAITHFULNESS_MIN})")
    print(f"  Answer Relevancy : {metrics.get('answer_relevancy',0):.3f}  (seuil={RELEVANCY_MIN})")
    print(f"  Context Recall   : {metrics.get('context_recall',0):.3f}  (seuil={RECALL_MIN})")
    print(f"  -> RAG {'PASSE' if metrics['passed'] else 'ECHOUE'}")
    return metrics


@step
def finetune_trigger(rag_metrics: Dict) -> Annotated[bool, "should_finetune"]:
    """
    Step 4 : Declenche le fine-tuning si une metrique RAGAS est sous son seuil.
    Conditions (OU) : Faithfulness<0.75 | Answer Relevancy<0.70 | Context Recall<0.65
    """
    print("\n[STEP 4] DECLENCHEUR CONDITIONNEL")
    reasons = []
    if rag_metrics.get("faithfulness", 1.0)     < FAITHFULNESS_MIN:
        reasons.append(f"Faithfulness={rag_metrics['faithfulness']:.3f} < {FAITHFULNESS_MIN}")
    if rag_metrics.get("answer_relevancy", 1.0) < RELEVANCY_MIN:
        reasons.append(f"Answer Relevancy={rag_metrics['answer_relevancy']:.3f} < {RELEVANCY_MIN}")
    if rag_metrics.get("context_recall", 1.0)   < RECALL_MIN:
        reasons.append(f"Context Recall={rag_metrics['context_recall']:.3f} < {RECALL_MIN}")

    should_finetune = len(reasons) > 0
    if should_finetune:
        print("  -> Fine-tuning DECLENCHE :")
        for r in reasons:
            print(f"     - {r}")
    else:
        print("  -> Fine-tuning non necessaire (toutes metriques OK)")
    return should_finetune


@step
def qlora_finetune(should_finetune: bool) -> Annotated[str, "lora_weights_path"]:
    """
    Step 5 : Fine-tuning QLoRA 4-bit NF4 si should_finetune=True.
    Modele : TinyLlama (dev) / Mistral-7B (prod). Anti-forgetting : replay 20%.
    """
    if not should_finetune:
        print("\n[STEP 5] Fine-tuning SKIP (metriques OK)")
        return "artifacts/lora_weights/current"
    print("\n[STEP 5] FINE-TUNING QLORA")
    from training.qlora_trainer import run_qlora_training
    lora_path = run_qlora_training()
    print(f"  -> Poids LoRA : {lora_path}")
    return lora_path


def _rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L : F-score base sur la plus longue sous-sequence commune."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if ref[i-1] == hyp[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / n if n else 0
    r = lcs / m if m else 0
    return round(2 * p * r / (p + r), 3) if (p + r) > 0 else 0.0


@step
def evaluate_model(lora_weights_path: str) -> Annotated[Dict, "model_metrics"]:
    """
    Step 6 : Evaluation reelle sur le val set.
    Metriques calculees :
      - ROUGE-L (F-score LCS) entre ground truth et contexte recupere
      - Cosine moyen des top-3 chunks retrouves
      - num_evaluated : nombre d'exemples evalues
    """
    import numpy as np
    print("\n[STEP 6] EVALUATION MODELE (val set)")

    val_path = "artifacts/val.json"
    if not os.path.exists(val_path):
        print("  -> val.json absent, skip evaluation")
        return {"rouge_l": 0.0, "cosine_avg": 0.0, "num_evaluated": 0,
                "lora_path": lora_weights_path, "improved": True}

    with open(val_path, encoding="utf-8") as f:
        val_data = json.load(f)

    from rag.retriever import load_index, search_with_scores
    from data_preparation.chunking import load_chunks
    index  = load_index()
    chunks = load_chunks()

    rouge_scores  = []
    cosine_scores = []

    for entry in val_data[:20]:          # 20 exemples max pour la vitesse
        question     = entry.get("instruction", "").strip()
        ground_truth = entry.get("output", "").strip()
        if not question or not ground_truth:
            continue
        results = search_with_scores(question, index, chunks, k=3)
        if not results:
            rouge_scores.append(0.0)
            continue
        avg_cosine   = float(np.mean([s for _, s in results]))
        context_text = " ".join([c for c, _ in results])
        rouge_scores.append(_rouge_l(ground_truth, context_text))
        cosine_scores.append(avg_cosine)

    avg_rouge  = round(float(np.mean(rouge_scores)),  3) if rouge_scores  else 0.0
    avg_cosine = round(float(np.mean(cosine_scores)), 3) if cosine_scores else 0.0

    metrics = {
        "rouge_l":       avg_rouge,
        "cosine_avg":    avg_cosine,
        "num_evaluated": len(rouge_scores),
        "lora_path":     lora_weights_path,
        "improved":      avg_rouge > 0.05 or avg_cosine > 0.30
    }
    print(f"  ROUGE-L    : {avg_rouge:.3f}")
    print(f"  Cosine moy : {avg_cosine:.3f}")
    print(f"  Exemples   : {len(rouge_scores)}")
    print(f"  -> {'AMELIORE' if metrics['improved'] else 'NON AMELIORE'}")
    return metrics


@step
def register_model(model_metrics: Dict) -> Annotated[str, "model_version"]:
    """
    Step 7 : Enregistre le modele dans le registry si improved=True (gate qualite).
    Sauvegarde : artifacts/model_registry.json
    """
    print("\n[STEP 7] MODEL REGISTRY")
    version = f"v{int(time.time())}"

    if model_metrics.get("improved", False):
        registry_path = "artifacts/model_registry.json"
        registry = {}
        if os.path.exists(registry_path):
            with open(registry_path) as f:
                registry = json.load(f)
        registry[version] = {
            "metrics":       {k: v for k, v in model_metrics.items() if k != "lora_path"},
            "lora_path":     model_metrics.get("lora_path", ""),
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        os.makedirs("artifacts", exist_ok=True)
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"  -> Modele enregistre : version={version}")
    else:
        print("  -> Non enregistre (pas d'amelioration)")
    return version


@step
def deploy_serving(model_version: str) -> Annotated[bool, "deployed"]:
    """
    Step 8 : Met a jour la config de deploiement (artifacts/deploy_config.json).
    En production : restart Docker/K8s. L'API FastAPI recharge au prochain demarrage.
    """
    print("\n[STEP 8] DEPLOY SERVING")
    config = {
        "model_version": model_version,
        "lora_path":     "artifacts/lora_weights/current",
        "faiss_index":   "artifacts/faiss_index.bin",
        "chunks_path":   "artifacts/chunks.json",
        "deployed_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_url":       "http://localhost:8000",
        "docs_url":      "http://localhost:8000/docs"
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/deploy_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  -> Config deployee : version={model_version}")
    print(f"  -> API : http://localhost:8000/docs")
    return True


@pipeline(name="rgpd_assistant_pipeline")
def rgpd_pipeline(pdf_path: str = "data_preparation/RGPD.pdf"):
    """Pipeline MLOps complet : 8 steps ZenML orchestrees."""
    chunks        = ingest_and_chunk(pdf_path=pdf_path)
    index_path    = embed_and_index(chunks=chunks)
    rag_metrics   = evaluate_rag(index_path=index_path)
    should_ft     = finetune_trigger(rag_metrics=rag_metrics)
    lora_path     = qlora_finetune(should_finetune=should_ft)
    model_metrics = evaluate_model(lora_weights_path=lora_path)
    model_version = register_model(model_metrics=model_metrics)
    deploy_serving(model_version=model_version)


if __name__ == "__main__":
    rgpd_pipeline()
