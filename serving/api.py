from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from rag.retriever import search, build_faiss_index

app = FastAPI()

# =========================
# LLM
# =========================
llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
# DATA
# =========================
chunks = [
    "Le RGPD protège les données personnelles des citoyens.",
    "La CNIL est l'autorité française de protection des données.",
    "Les entreprises doivent respecter la minimisation des données.",
    "Le droit à l'oubli permet la suppression des données.",
]

embeddings = embedder.encode(chunks)
index = build_faiss_index(embeddings)


# =========================
# BUILD ANSWER
# =========================
def build_answer(question, retrieved_chunks):

    context = "\n".join(retrieved_chunks)

    prompt = f"""
Contexte:
{context}

Question:
{question}

Réponse:
"""

    output = llm(
        prompt,
        max_new_tokens=50,
        do_sample=False,
        return_full_text=False
    )[0]["generated_text"]

    answer = output.strip().split("Question:")[0]

    return answer.strip()


# =========================
# ENDPOINT
# =========================
@app.post("/ask")
def ask(req: QueryRequest):

    results = search(req.question, index, chunks)

    answer = build_answer(req.question, results)

    return {
        "question": req.question,
        "answer": answer,
        "sources": results
    }