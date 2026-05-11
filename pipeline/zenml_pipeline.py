from zenml import pipeline, step
import sys
import os

# =========================
# FIX IMPORT PATH (IMPORTANT)
# =========================
sys.path.append(os.path.abspath("."))

# =========================
# STEP 1 : EXTRACTION
# =========================
@step
def extract_step():
    from data_preparation.extract_text import extract_text
    text = extract_text()
    return text

# =========================
# STEP 2 : CHUNKING
# =========================
@step
def chunk_step(text):
    from data_preparation.chunking import chunk_text
    chunks = chunk_text(text)
    return chunks

# =========================
# PIPELINE MLOPS
# =========================
@pipeline
def rgpd_pipeline():
    text = extract_step()
    chunks = chunk_step(text)

# =========================
# EXECUTION
# =========================
if __name__ == "__main__":
    rgpd_pipeline()