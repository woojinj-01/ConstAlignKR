import json

from src.util import dir_judgement, embed_long_text, get_embedding_model, cosine
from src.parser import extract_decision_type, extract_other_opinions, extract_decision_gst

target_panre_nums = ["2011헌바379", "2017헌바127"]

embedding_model_paths = {
        "KoSimCSE": "models/before_tuned/kosimcse"
    }

for model_name, model_path in embedding_model_paths.items():
    print(f"\n=== {model_name} ===")
    
    model = get_embedding_model(model_path)

    for panre_num in target_panre_nums:
        file_path = dir_judgement(panre_num)
            
        with open(file_path, "r", encoding="utf-8") as f:
            j = json.load(f)

            if extract_decision_type(j) == "각하":
                continue

            embs1 = embed_long_text(extract_decision_gst(j), model)
            embs2 = embed_long_text(extract_other_opinions(j), model)

            print(cosine(embs1, embs2))