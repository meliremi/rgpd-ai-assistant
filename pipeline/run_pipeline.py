from data_preparation.extract_text import extract_text
from data_preparation.chunking import chunk_text

def run_pipeline():
    text = extract_text()

    if not text:
        return

    chunks = chunk_text(text)

    print("Nombre de chunks:", len(chunks))
    print("\nExemple chunk:")
    print(chunks[0])


if __name__ == "__main__":
    run_pipeline()