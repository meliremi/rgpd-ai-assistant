import random
import json

def split_and_export(dataset, train_ratio=0.8, output_file="dataset_alpaca.json"):
    random.shuffle(dataset)
    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    print(f"Export terminé: {len(train_data)} train, {len(val_data)} val.")