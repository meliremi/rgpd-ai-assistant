# =========================
# RAG CHUNKING MODULE
# =========================

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Découpe un texte en chunks pour RAG.

    Args:
        text (str): texte complet
        chunk_size (int): taille d’un chunk
        overlap (int): chevauchement entre chunks

    Returns:
        list[str]: liste de chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# =========================
# TEST LOCAL (OPTIONNEL)
# =========================

if __name__ == "__main__":
    sample_text = "RGPD est un règlement européen. " * 100

    chunks = chunk_text(sample_text)

    print("Nombre de chunks:", len(chunks))
    print("\nExemple chunk:")
    print(chunks[0])