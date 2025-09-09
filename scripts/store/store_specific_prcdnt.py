from sentence_transformers import losses
from torch.utils.data import DataLoader
import json

from src.req_data import get_prcdnt
from src.util import path_prcdnt

target_panre_nums = ["2016헌가1"]
target_type = ["헌가"]
target_event_nums = ["46113"]
target_panre_types = ["03"]

if __name__ == "__main__":

    for i in range(len(target_panre_nums)):

        panre_num = target_panre_nums[i]
        type = target_type[i]
        event_num = target_event_nums[i]
        panre_type = target_panre_types[i]

        doc = get_prcdnt(event_num=event_num, panre_type=panre_type)

        file_path = path_prcdnt(type, panre_num)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {file_path}")

