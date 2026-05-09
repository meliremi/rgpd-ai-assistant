from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from transformers import pipeline
from sentence_transformers import SentenceTransformer

import numpy as np

from rag.retriever import search, build_faiss_index
from data_preparation.chunking import load_chunks

# =========================
# APP
# =========================
app = FastAPI(
    title="RGPD RAG API",
    version="6.0"
)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LLM
# =========================
print("Loading TinyLlama...")

llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto"
)

print("TinyLlama loaded.")

# =========================
# EMBEDDINGS
# =========================
print("Loading embeddings model...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Embeddings model loaded.")

# =========================
# REQUEST
# =========================
class QueryRequest(BaseModel):
    question: str

# =========================
# DATA
# =========================
print("Loading chunks...")
chunks = load_chunks()
print(f"{len(chunks)} chunks loaded.")

print("Creating embeddings...")
embeddings = embedder.encode(
    chunks,
    normalize_embeddings=True,
    show_progress_bar=False
)

print("Building FAISS index...")
index = build_faiss_index(embeddings)

print("FAISS ready.")

# =========================
# RELEVANCE CHECK
# =========================
def is_relevant(question, retrieved_chunks):

    if not retrieved_chunks:
        return False

    rgpd_keywords = [
        "rgpd", "dpo", "cnil", "donnée", "données",
        "personnelle", "personnelles", "consentement",
        "traitement", "privacy", "vie privée", "effacement",
        "oubli", "protection", "responsable", "sous-traitant",
        "transfert", "union européenne", "ue", "aipd",
        "violation", "cookies", "fichier"
    ]

    q = question.lower()

    if not any(k in q for k in rgpd_keywords):
        return False

    # embeddings question
    q_emb = embedder.encode(question, normalize_embeddings=True)

    # embeddings chunks
    chunk_embs = embedder.encode(retrieved_chunks, normalize_embeddings=True)

    # sécurité anti crash
    if len(chunk_embs) == 0:
        return False

    scores = np.dot(chunk_embs, q_emb)

    best_score = float(np.max(scores)) if len(scores) > 0 else 0.0

    print("RELEVANCE SCORE:", best_score)

    return best_score > 0.40


# =========================
# PROMPT
# =========================
def build_prompt(question, context):

    return f"""
Tu es un assistant expert du RGPD.

RÈGLES STRICTES :
- utilise UNIQUEMENT le contexte
- si info absente → "Information non disponible dans le contexte."
- réponse courte (max 4 lignes)
- ne jamais inventer

CONTEXTE:
{context}

QUESTION:
{question}

REPONSE:
"""


# =========================
# GENERATION
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
        max_new_tokens=60,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.1
    )[0]["generated_text"]

    # extraction réponse propre
    if "REPONSE:" in output:
        answer = output.split("REPONSE:")[-1].strip()
    else:
        answer = output.strip()

    # nettoyage brut
    answer = answer.split("QUESTION")[0].strip()
    answer = answer.split("CONTEXTE")[0].strip()

    if len(answer) < 5:
        return "Information non disponible dans le contexte."

    # anti hallucination
    forbidden = ["je pense", "probablement", "peut-être", "selon moi"]

    if any(x in answer.lower() for x in forbidden):
        return "Information non disponible dans le contexte."

    return answer


# =========================
# API ENDPOINT
# =========================
@app.post("/ask")
def ask(req: QueryRequest):

    question = req.question.strip()

    print("\n====================")
    print("QUESTION:", question)

    # retrieval
    results = search(question, index, chunks)

    # sécurité
    results = list(dict.fromkeys(results))[:5]

    print("RESULTS COUNT:", len(results))

    # answer
    answer = generate_answer(question, results)

    print("ANSWER:", answer)

    # si pas bon
    if answer == "Information non disponible dans le contexte.":
        results = []

    return {
        "question": question,
        "answer": answer,
        "sources": results
    }


# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "TinyLlama",
        "rag": "active"
    }


# =========================
# FRONTEND
# =========================
app.mount(
    "/",
    StaticFiles(directory="frontend", html=True),
    name="frontend"
)