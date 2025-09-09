import json, os

from src.req_data import get_judgement
from src.parser import Precedent
from src.util import dir_triplet

dir_path = "./data/raw/judgement_list"

target_type = ["헌나", "헌다"]

for filename in os.listdir(dir_path):
    if filename.endswith(".json") and filename.split('.')[0] in target_type:
        file_path = os.path.join(dir_path, filename)

        print(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            judgements = data["body"]["items"]["item"]  
            for j in judgements:

                prcdnt = Precedent(j)

                if '각하' in j["rstaRsta"]:
                    continue

                event_num = str(j["eventNum"])
                panre_num = str(j["eventNo"])
                
                match j["panreType"]:
                    case "결정문":
                        panre_type = "01"
                    case "공보":
                        panre_type = "02"
                    case "판례집":
                        panre_type = "03"
                    case _:
                        continue

                doc = get_judgement(event_num=event_num, panre_type=panre_type)

                file_path = f"./data/raw/judgement/{panre_num}.json"

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)

                print(f"✅ Saved {file_path}")

                triplet = prcdnt.triplet()

                with open(dir_triplet(), "a") as f:
                    json.dump(list(triplet), f)
                    f.write("\n")


