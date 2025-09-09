from sentence_transformers import losses
from torch.utils.data import DataLoader
import os

from src.util import load_triplets, get_embedding_model

if __name__ == "__main__":
    # ---------- Config ----------
    triplet_file = "data/processed/legal_triplets.jsonl" 
    output_dir = "models/fine_tuned"
    batch_size = 16
    epochs = 3
    warmup_steps = 100

    # Embedding models to fine-tune
    # embedding_model_paths = {
    #     "KoSimCSE": "models/before_tuned/kosimcse",
    #     "BGE-Korean": "models/before_tuned/bge-korean",
    #     "E5": "models/before_tuned/e5-multilingual",

    #     "KoSimCSE-FT": "models/fine_tuned/kosimcse",
    #     "BGE-Korean-FT": "models/fine_tuned/bge-korean",
    #     "E5-FT": "models/fine_tuned/e5-multilingual"
    # }

    embedding_model_paths = {
        "KoSimCSE": "models/before_tuned/kosimcse"
    }

    triplet_examples = load_triplets(triplet_file)
    print(f"Loaded {len(triplet_examples)} triplets.")

    # ---------- Fine-tune Each Model ----------
    for model_name, model_path in embedding_model_paths.items():
        print(f"\n=== Fine-tuning {model_name} ===")
        
        model = get_embedding_model(model_path)
        train_dataloader = DataLoader(triplet_examples, shuffle=True,
                                      batch_size=batch_size)
        train_loss = losses.TripletLoss(model)
        
        model_output_path = os.path.join(output_dir, model_name + "-FT")
        os.makedirs(model_output_path, exist_ok=True)
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=model_output_path
        )
        
        print(f"Saved fine-tuned model to {model_output_path}")
