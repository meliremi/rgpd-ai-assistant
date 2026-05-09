from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from rag.retriever import search, build_faiss_index
from data_preparation.chunking import load_chunks

# =========================
# FASTAPI
# =========================
app = FastAPI(
    title="RGPD RAG API",
    version="4.0"
)

# =========================
# EMBEDDING MODEL
# =========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# REQUEST MODEL
# =========================
class QueryRequest(BaseModel):
    question: str

# =========================
# LOAD DATASET
# =========================
chunks = load_chunks()

# suppression des doublons
chunks = list(dict.fromkeys(chunks))

# embeddings
embeddings = embedder.encode(chunks)

# FAISS
index = build_faiss_index(embeddings)

# =========================
# BUILD ANSWER
# =========================
def build_answer(question, retrieved_chunks):

    if len(retrieved_chunks) == 0:
        return "Information non disponible dans le contexte."

    # meilleur chunk
    best_chunk = retrieved_chunks[0]

    # nettoyage
    best_chunk = best_chunk.replace("\n", " ").strip()

    # extraction après "?"
    if "?" in best_chunk:

        parts = best_chunk.split("?")

        if len(parts) > 1:

            answer = parts[1].strip()

            # sécurité
            if len(answer) > 3:
                return answer

    # fallback
    return best_chunk

# =========================
# ENDPOINT
# =========================
@app.post("/ask")
def ask(req: QueryRequest):

    # recherche FAISS
    results = search(req.question, index, chunks)

    # suppression doublons
    results = list(dict.fromkeys(results))

    # top 5
    results = results[:5]

    # génération réponse
    answer = build_answer(req.question, results)

    return {
        "question": req.question,
        "answer": answer,
        "sources": results
    }

# =========================
# HEALTHCHECK
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok"
    }