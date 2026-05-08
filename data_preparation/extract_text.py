import pdfplumber
import os

def extract_text(pdf_path="data_preparation/RGPD.pdf"):
    if not os.path.exists(pdf_path):
        print("Fichier introuvable")
        return None

    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    return text


if __name__ == "__main__":
    text = extract_text()

    if text:
        with open("rgpd_text.txt", "w", encoding="utf-8") as f:
            f.write(text)

        print("Extraction OK")