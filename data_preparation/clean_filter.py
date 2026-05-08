import re

def clean_text(text):
    # Suppression des doubles sauts de ligne et espaces superflus
    text = re.sub(r'\s+', ' ', text)
    # Suppression des numéros de page isolés
    text = re.sub(r'\n\d+\n', '\n', text)
    return text.strip()