import json, os
from rich import print

import src.parser as mine
from src.util import dir_prcdnt, PRCDNT_TYPES
from src.parser import print_marker_tree, Precedent

target_type = ["헌바"]

stop = 15

for t in target_type:
    count = 0

    for filename in os.listdir(dir_prcdnt(t)):
        filepath = os.path.join(dir_prcdnt(t), filename)
        if os.path.isfile(filepath):
            print(filepath)   # do something with the file

        with open(filepath, "r", encoding="utf-8") as f:
            j = json.load(f)

            case = Precedent(j)
            # print(case.issues())

            if case.other_opinions() is None:
                continue

            print(case.issues())
            # print(case.other_opinions())
            # print(case.decision_gst())
            # print(case.reason(first_only=True))
            # print_marker_tree(case.other_opinions(), show_text=True)

            # print(case.match_sections_to_markers(case.issues(), case.other_opinions()))
            print(case.match_sections_to_markers_llm(case.issues(), case.other_opinions()))
            # print(case.other_opinions())

            # print_marker_tree(case.other_opinions(), show_text=True)

            # print(case.ref_provision(j))
            # print(case.ref_precedent(j))

            # print(case.test())

            count += 1
        
        if count >= stop:
            break
    


