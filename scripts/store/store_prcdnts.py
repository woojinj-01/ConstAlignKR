import json, os

from src.req_data import get_prcdnt
from src.util import path_prcdnt_list, path_prcdnt

target_type = ["헌바"]

for t in target_type:
    src_path = path_prcdnt_list(t)

    print(f"\n==={t}===")

    count = 0
    
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        judgements = data["body"]["items"]["item"]  

        for j in judgements:

            if '각하' in j["rstaRsta"]:
                continue

            if '전원재판부' not in j["jgdmtCort"]:
                continue

            event_num = str(j["eventNum"])
            panre_num = str(j["eventNo"])

            dst_path = path_prcdnt(t, panre_num)

            if os.path.exists(dst_path):
                continue
            
            match j["panreType"]:
                case "결정문":
                    panre_type = "01"
                case "공보":
                    panre_type = "02"
                case "판례집":
                    panre_type = "03"
                case _:
                    continue

            doc = get_prcdnt(event_num=event_num, panre_type=panre_type)

            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

            count += 1

            print(f"✅ Saved {panre_num} --- Count: {count}")

    print(f"🏁 {t}: {count} prcdnts saved")
        


