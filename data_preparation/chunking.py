import json

# =========================
# LOAD DATASET (RAG BASE)
# =========================
def load_chunks(path="dataset_rgpd_50.json"):
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # on transforme en chunks utilisables pour FAISS
    chunks = [
        f"{item['question']} {item['answer']}"
        for item in dataset
    ]

    return chunks


# =========================
# OPTION PDF CHUNKING (INUTILE POUR L’INSTANT)
# =========================
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap

    return chunks


# =========================
# TEST
# =========================
if __name__ == "__main__":
    chunks = load_chunks()
    print("Nb chunks:", len(chunks))
    print(chunks[0])