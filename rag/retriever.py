import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# MODEL EMBEDDINGS
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# BUILD FAISS INDEX
# =========================
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index


# =========================
# SEARCH FUNCTION
# =========================
def search(query, index, chunks, k=5):
    query_vector = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vector, k)

    results = [chunks[i] for i in indices[0]]

    return results


# =========================
# TEST LOCAL
# =========================
if __name__ == "__main__":

    chunks = [
        "Le RGPD protège les données personnelles des citoyens.",
        "La CNIL est l'autorité française de protection des données.",
        "Les entreprises doivent respecter la minimisation des données.",
        "Le droit à l'oubli permet la suppression des données.",
    ]

    embeddings = model.encode(chunks)

    index = build_faiss_index(embeddings)

    query = "Qu'est-ce que la CNIL ?"

    results = search(query, index, chunks)

    print("\n--- RESULTATS ---")
    for r in results:
        print("-", r)