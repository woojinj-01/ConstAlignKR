import os, json
from src.req_data import get_judgement_lists

case_type = ["헌가", "헌나", "헌다", "헌라", "헌마", "헌바"]

os.makedirs("./data/raw/judgement_list", exist_ok=True)

for t in case_type:
    judgements = get_judgement_lists(t)

    file_path = f"./data/raw/judgement_list/{t}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(judgements, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {file_path}")


