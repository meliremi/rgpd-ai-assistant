import json
import os
from extract_text import extract_text  
from clean_filter import clean_text    
from generate_qa import MANUAL_QA_DATA 

def to_alpaca_with_context(question, answer, context):
    return {
        "instruction": question,
        "input": f"Contexte : \n{context}",
        "output": answer
     }

def run_full_pipeline():
    print("--- Étape 1 : Extraction du PDF ---")
    # Changement ici : on utilise le nom direct du fichier
    raw_text = extract_text("RGPD.pdf") 
    
    if not raw_text:
        print("Erreur : Impossible d'extraire le texte.")
        return

    print("--- Étape 2 : Nettoyage du texte ---")
    cleaned_text = clean_text(raw_text)

    print("--- Étape 3 : Conversion au format Alpaca ---")
    alpaca_dataset = []
    for qa in MANUAL_QA_DATA:
        entry = to_alpaca_with_context(qa["question"], qa["answer"], cleaned_text[:2000])
        alpaca_dataset.append(entry)

    print("--- Étape 4 : Création des splits ---")
    train_data = alpaca_dataset[:40]
    val_data = alpaca_dataset[40:]

    with open("dataset_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open("dataset_val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=4, ensure_ascii=False)
            
    print(f"\n✅ Succès ! {len(train_data)} train, {len(val_data)} val.")

if __name__ == "__main__":
    run_full_pipeline()