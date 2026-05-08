from sentence_transformers import SentenceTransformer

# modèle léger et efficace
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks):
    """
    Transforme une liste de chunks en embeddings vectoriels.
    """
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings


if __name__ == "__main__":
    test_chunks = [
        "Le RGPD protège les données personnelles.",
        "La CNIL est l'autorité française."
    ]

    vectors = embed_chunks(test_chunks)

    print("Nb chunks:", len(test_chunks))
    print("Dimension embedding:", len(vectors[0]))