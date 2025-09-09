import json, os
from rich import print

import src.parser as mine
from src.util import dir_prcdnt, PRCDNT_TYPES, get_embedding_model, embed, embed_long_text, cosine
from src.parser import print_marker_tree, Precedent, normalize_issue

target_type = ["헌바"]

stop = 5

model = get_embedding_model()

for t in target_type:
    count = 0

    for filename in os.listdir(dir_prcdnt(t)):
        filepath = os.path.join(dir_prcdnt(t), filename)

        if os.path.isfile(filepath):
            print(filename)
        else:
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            j = json.load(f)

            case = Precedent(j)
            # print(case.pansi_matt(as_list=True))
            result = case.triplet_dev()

            # print(result)

            for trip in result["triplets"]:

                print(trip)

                anc = trip["Anchor"]
                pos = trip["Positive"]
                neg = trip["Negative"]

                if anc is None or all(x is None for x in (pos, neg)):
                    continue

                emb_i = embed(anc, model)

                text = pos or neg
                emb_t = embed_long_text(text, model)

                # print("Anchor:\n", anc)
                # print("Text:\n", text)

                print(cosine(emb_i, emb_t))
                print("================\n")

        count += 1
        
        if count >= stop:
            break
    


