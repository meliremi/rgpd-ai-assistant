from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

from rag.retriever import search, build_faiss_index
from data_preparation.chunking import load_chunks

app = FastAPI(title="RGPD RAG API", version="5.0")

# =========================
# LLM
# =========================
llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto"
)

# =========================
# EMBEDDINGS
# =========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    question: str

# =========================
# DATA
# =========================
chunks = load_chunks()
embeddings = embedder.encode(chunks, normalize_embeddings=True)
index = build_faiss_index(embeddings)

# =========================
# SCORE FILTER (IMPORTANT)
# =========================
def is_relevant(question, retrieved_chunks):
    q_emb = embedder.encode(question, normalize_embeddings=True)

    chunk_embs = embedder.encode(retrieved_chunks, normalize_embeddings=True)

    scores = np.dot(chunk_embs, q_emb)

    return float(max(scores)) > 0.35  # seuil IMPORTANT

# =========================
# PROMPT
# =========================
def build_prompt(question, context):
    return f"""
Tu es un assistant expert UNIQUEMENT en RGPD.

Règles strictes :
- utilise uniquement le contexte
- si contexte insuffisant → répond "Information non disponible dans le contexte"
- réponse courte (max 5 lignes)
- ne jamais inventer

CONTEXTE:
{context}

QUESTION:
{question}

REPONSE:
"""

# =========================
# GENERATION SAFE
# =========================
def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:
        return "Information non disponible dans le contexte."

    if not is_relevant(question, retrieved_chunks):
        return "Information non disponible dans le contexte."

    context = "\n".join(retrieved_chunks[:4])

    prompt = build_prompt(question, context)

    output = llm(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        temperature=0.0
    )[0]["generated_text"]

    if "REPONSE:" in output:
        answer = output.split("REPONSE:")[-1].strip()
    else:
        answer = output.strip()

    # cleanup hardcore
    answer = answer.split("QUESTION")[0].strip()
    answer = answer.split("CONTEXTE")[0].strip()

    if len(answer) < 5:
        return "Information non disponible dans le contexte."

    return answer

# =========================
# ENDPOINT
# =========================
@app.post("/ask")
def ask(req: QueryRequest):

    results = search(req.question, index, chunks)

    # anti duplication + top k
    results = list(dict.fromkeys(results))[:5]

    answer = generate_answer(req.question, results)

    return {
        "question": req.question,
        "answer": answer,
        "sources": results
    }

# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}