from typing import Callable, Dict, List, Tuple
import re
from src.util import gpt4_call, cosine, get_embedding_model, embed

PATTERNS = [
    ("num", re.compile(r"^([1-9][0-9]{0,2}\.)\s+(.+)")),
    ("kor", re.compile(r"^([가-힣]\.)\s+(.+)")),
    ("paren_num", re.compile(r"^(\([1-9][0-9]{0,2}\))\s+(.+)")),
    ("paren_kor", re.compile(r"^(\([가-힣]\))\s+(.+)")),
    ("rparen_num", re.compile(r"^([1-9][0-9]{0,2}\))\s+(.+)")),
    ("rparen_kor", re.compile(r"^([가-힣]\))\s+(.+)")),
]

HIERARCHY = ["num", "kor", "paren_num", "paren_kor", "rparen_num", "rparen_kor"]

ISSUE_KEYWORDS = ["(적극)", "(소극)"]

VALID_SEQUENCES = {
    "num": [f"{i}." for i in range(1, 200)],   # "1.", "2.", ...
    "kor": [c + "." for c in "가나다라마바사아자차카타파하"],  # "가.", "나.", ...
    "paren_num": [f"({i})" for i in range(1, 200)],  # "(1)", "(2)", ...
}

_patterns = [
    r'(?:되는지|하는지|인지|한지)\s*여부\s*\([^)]*\)\s*$',  # 여부(소극) / 여부(적극)
    r'(?:되는지|하는지|인지|한지)\s*여부\s*$',              # 여부
]


class Precedent:
    def __init__(self, j) -> None:
        self.doc = j
        self.prcdnt_num = self.doc["body"]["items"]["item"][0]["eventNo"]

        if self.prcdnt_num[:4].isdigit():
            self.year = int(self.prcdnt_num[:4])
        elif self.prcdnt_num[:2].isdigit():
            year = int(self.prcdnt_num[:2])
            self.year = 1900 + year
        else:
            self.year = 0

    def test(self):

        # print(self.pansi_matt)
        # print(self.decision_gst)

        return self.gather_arguments(self.reason)
    
    def is_issue(self, title: str):
        """Check if this section looks like an 'issue'"""
        return any(kw in title for kw in ISSUE_KEYWORDS)
    
    def is_relevant(self, title: str, issue: str):
        prompt = f"You are given:\
                \nJudgement document title: {title}\
                \nTarget issue: {issue}\
                \nDetermine if the judgement document title is directly and clearly relevant to the target issue.\
                \nAnswer Yes only if the title clearly indicates it addresses the target issue.\
                \nAnswer No if the relevance is uncertain, indirect, or weak.\
                \nDo not provide explanations, reasoning, or any additional text. Respond strictly with Yes or No."

        return gpt4_call(prompt)
        
    def triplet(self):
        triplet = {"Anc": "Anchor",
                   "Pos": "Positive",
                   "Neg": "Negative"}
        
        return triplet

    def triplet_dev(self):

        issues = self.issues()
        markers = self.match_sections_to_markers(self.issues(), self.reason(first_only=True))
        reason = self.reason()

        result = {"prcdnt_num": self.prcdnt_num,
                  "year": self.year,
                  "other_opinions": False,
                  "triplets": []}

        for issue, marker in zip(issues, markers):
            triplet = {"Anchor": None,
                       "Positive": None,
                       "Negative": None}
            
            if not issue or not marker:
                continue

            path = marker.split('-')
            text = summarize(get_text_by_markers(reason, path), issue)

            if not text:
                continue
            
            triplet["Anchor"] = normalize_issue(issue)

            if issue[-3:-1] == "적극":
                triplet["Positive"] = text
            elif issue[-3:-1] == "소극":
                triplet["Negative"] = text
            else:
                continue

            result["triplets"].append(triplet)

        # if self.other_opinions():
        #     self.parse_other_opinions(result)

        return result
    
    def parse_other_opinions(self, result):
        pass

    def issues(self):
        return [p for p in self.pansi_matt(as_list=True) if self.is_issue(p)]
    
    def text_of_decision(self, first_only=False):    # 주문
        return parse_document(self.doc["body"]["items"]["item"][0]["textOfDecision"], first_only=first_only)

    def decision_type(self, first_only=False):   # 판결 종류 
        return parse_document(self.doc["body"]["items"]["item"][0]["rstaRsta"], first_only=first_only)

    def pansi_matt(self, as_list=False, first_only=False):  # 판시사항
        if as_list:
            pansi_matt = parse_document(self.doc["body"]["items"]["item"][0]["pansiMatt"], first_only=first_only)
            return [c['text'] for c in pansi_matt['children']]
        else:
            return parse_document(self.doc["body"]["items"]["item"][0]["pansiMatt"], first_only=first_only)

    def decision_gst(self, first_only=False):    # 결정요지
        return parse_document(self.doc["body"]["items"]["item"][0]["decisionGst"], first_only=first_only)

    def adj(self, first_only=False):  # 심판의 대상
        return parse_document(self.doc["body"]["items"]["item"][0]["otherOpinions"], first_only=first_only)

    def adj_provision(self):  # 심판대상조문
        return parse_document(self.doc["body"]["items"]["item"][0]["otherOpinions"])

    def event_summary(self):  # 사건의 개요
        return parse_document(self.doc["body"]["items"]["item"][0]["eventSummary"])

    def reason(self, first_only=False):  # 이유
        return parse_document(self.doc["body"]["items"]["item"][0]["reason"], first_only=first_only)

    def other_opinions(self, first_only=False):  # 별개/보충의견
        return parse_document(self.doc["body"]["items"]["item"][0]["otherOpinions"], first_only=first_only)

    def ref_provision(self):  # 참조조문
        return parse_document(self.doc["body"]["items"]["item"][0]["refrnPrvsn"])

    def ref_precedent(self):  # 참조판례
        return parse_document(self.doc["body"]["items"]["item"][0]["refrnPrcdnt"])

    def collect_marker_paths(self, node: Dict, prefix: str = "", top_level_only=False, depth=0) -> List[Tuple[str, str, int]]:
        """
        Traverse tree and collect (marker_path, text, depth) tuples.
        depth=0 means root, depth=1 means top-level markers like '1.', '2.', '3.'.
        """
        results = []
        marker = node.get("marker", "")
        text = node.get("text", "")

        # skip 'root' in path
        if marker == "root":
            full_marker = prefix
        else:
            full_marker = f"{prefix}-{marker}" if prefix else marker

        if text.strip():
            results.append((full_marker, text.strip(), depth))

        for child in node.get("children", []):
            results.extend(
                self.collect_marker_paths(
                    child,
                    prefix=full_marker,
                    top_level_only=top_level_only,
                    depth=depth + 1
                )
            )

        return results

    # def match_sections_to_markers(
    #     self,
    #     sections,
    #     tree,
    #     similarity_fn: Callable[[str, str], float] = cosine,
    #     threshold: float = 0.3
    # ) -> Dict[str, str]:
        
    #     if any(x is None for x in (sections, tree)):
    #         return None

    #     model = get_embedding_model("models/before_tuned/kosimcse")

    #     all_markers = self.collect_marker_paths(tree)

    #     # keep only depth == 1 (direct children of root)
    #     top_level_markers = [(m, t) for m, t, d in all_markers if d == 1 or d == 2]

    #     results = []
    #     for section in sections:
    #         best_marker, best_score = None, -1.0
    #         for marker, text in top_level_markers:
    #             section_emb = embed(section, model)
    #             text_emb = embed(text, model)

    #             score = similarity_fn(section_emb, text_emb)
    #             # print(section, text, score)
    #             if score > best_score:
    #                 best_marker, best_score = marker, score

    #         if best_marker and best_score >= threshold:
    #             results.append(best_marker)
    #         else:
    #             results.append(None)  # no good match

    #     return results  

    def match_sections_to_markers(
        self,
        sections,
        tree,
        similarity_fn: Callable[[str, str], float] = cosine,
        threshold: float = 0.3
    ) -> Dict[str, str]:
            
        if any(x is None for x in (sections, tree)):
            return None

        model = get_embedding_model("models/before_tuned/kosimcse")

        all_markers = self.collect_marker_paths(tree)

        # keep only depth == 1 or 2
        top_level_markers = [(m, t) for m, t, d in all_markers if d in (1, 2)]

        # ✅ include root text if available
        if tree.get("text"):
            print("ROOT FOUND")
            top_level_markers.append(("root", tree["text"]))

        results = []
        for section in sections:
            best_marker, best_score = None, -1.0
            section_emb = embed(section, model)

            for marker, text in top_level_markers:
                text_emb = embed(text, model)
                score = similarity_fn(section_emb, text_emb)
                if score > best_score:
                    best_marker, best_score = marker, score

            if best_marker and best_score >= threshold:
                results.append(best_marker)
            else:
                results.append(None)  # no good match

        return results
    
    def match_sections_to_markers_llm(
        self,
        sections,
        tree
    ) -> Dict[str, str]:
            
        if any(x is None for x in (sections, tree)):
            return None
        
        sys_prompt = "You are given:\
                    A list of legal issues.\
                    A document (text snippet) that discusses one of these issues.\
                    Task:\
                    Determine which issue the document most likely tackles.\
                    Return the result as the index of the single best-matching issue (0, 1, 2, …).\
                    If no clear match exists, return null.\
                    The output must be a single integer or null only — no explanations, no extra text."
        
        text_root = tree['text']
        text_child = tree['children'][0]['text'] if len(tree['children']) !=0 else None

        doc = text_child if text_root is None or len(text_root) == 0 else text_root

        print(doc)

        user_prompt = f"Disputed issues: {sections}\n\
                    Document: {text_root if text_child is None else text_root + text_child}"

        return gpt4_call(user_prompt, sys_prompt)


def parse_document(text: str, first_only=False):
    if text is None:
        return None

    def traverse_and_apply(node: dict):
        if "text" in node:
            node["text"] = get_first_sentence(node["text"])

        if "children" in node:
            for child in node["children"]:
                traverse_and_apply(child)

        return node

    lines = text.splitlines()
    root = {"marker": "root", "text": "", "children": [], "seen_markers": {}, "next_expected": {}}
    stack = [(None, root)]  # (label, node)
    found_index = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        matched = False
        for label, pattern in PATTERNS:
            m = pattern.match(stripped)
            if m:
                found_index = True
                marker, rest = m.groups()

                # --- IMPORTANT: determine hierarchy first (find correct parent) ---
                while stack and not is_parent(stack[-1][0], label):
                    stack.pop()
                # if the top has same label, pop it so new node becomes sibling
                if stack and stack[-1][0] == label:
                    stack.pop()

                parent_node = stack[-1][1]

                # initialize per-parent bookkeeping if needed
                parent_node.setdefault("seen_markers", {})
                parent_node.setdefault("next_expected", {})

                # prepare validation
                valid_markers = VALID_SEQUENCES.get(label, None)
                expected_idx = parent_node["next_expected"].get(label, 0)

                # If we don't have a VALID_SEQUENCES entry for this label type, accept everything.
                accept = False
                if valid_markers is None:
                    accept = True
                else:
                    # accept only when marker equals the expected next marker
                    if expected_idx < len(valid_markers) and marker == valid_markers[expected_idx]:
                        accept = True

                if not accept:
                    # skip this marker (out-of-sequence or restart like another "1.")
                    # mark as matched so this line is NOT treated as continuation text
                    matched = True
                    break

                # accept node: record seen marker and bump expected index
                parent_node["seen_markers"].setdefault(label, set()).add(marker)
                parent_node["next_expected"][label] = expected_idx + 1

                node = {
                    "marker": marker,
                    "text": rest.strip(),
                    "children": [],
                    "type": label,
                    "seen_markers": {},
                    "next_expected": {}
                }

                # attach and descend
                stack[-1][1]["children"].append(node)
                stack.append((label, node))
                matched = True
                break

        if not matched:
            # treat as continuation text for the current node
            if stack:
                cur = stack[-1][1]
                if cur.get("text"):
                    cur["text"] += "\n" + stripped
                else:
                    cur["text"] = stripped

    # if no markers found at all, wrap whole text as one block (original behavior)
    if not found_index:
        all_text = "\n".join([line for line in lines if line.strip()])
        root["children"].append({
            "marker": "본문",
            "text": all_text,
            "children": [],
            "seen_markers": {},
            "next_expected": {}
        })
        root["text"] = ""

    if first_only:
        return traverse_and_apply(root)
    else:
        return root


def is_parent(current_label, new_label):
    if current_label is None:
        return True
    return HIERARCHY.index(new_label) > HIERARCHY.index(current_label)


def get_first_sentence(text: str) -> str:

    def before_newline(s: str) -> str:
        return s.split("\n", 1)[0]

    if not text:
        return ""

    # Find newline index
    newline_index = text.find("\n")

    # Match punctuation that signals sentence end,
    # but ignore '.' if it's part of a date-like pattern (digit '.' digit)
    punctuation_match = re.search(r'(?<!\d)[.?!](?!\d)', text)

    # Determine earliest split point
    split_index = None
    if newline_index != -1 and punctuation_match:
        split_index = min(newline_index, punctuation_match.start())
    elif newline_index != -1:
        split_index = newline_index
    elif punctuation_match:
        split_index = punctuation_match.start()

    if split_index is not None:
        return before_newline(text[:split_index + 1].strip())  # include punctuation
    return before_newline(text.strip())


def print_marker_tree(node, indent=0, show_text=False, max_level=None):
    if node is None:
        return
    
    # if node["marker"] != "root":
    line = node["marker"]
    if show_text and node.get("text"):
        first_sentence = get_first_sentence(node["text"])
        if first_sentence:
            line += " " + first_sentence
    print("    " * indent + line)

    # stop recursion if max_level reached
    if max_level and node.get("type"):
        current_index = HIERARCHY.index(node["type"])
        max_index = HIERARCHY.index(max_level)
        if current_index >= max_index:
            return

    for child in node["children"]:
        print_marker_tree(child, indent + 1, show_text=show_text, max_level=max_level)


def get_text_by_markers(data, path):
    if not path:

        def gather_children(child):
            text = child.get('text')

            for grandchild in child.get("children", []):
                text += gather_children(grandchild)

            return text
        
        return gather_children(data)
    
    # print(data)
    next_marker = path[0]

    for child in data.get("children", []):
        # print(next_marker)
        # print(child)

        if child.get("marker") == next_marker:
            # print("HIT!!")
            return get_text_by_markers(child, path[1:])
    return None


def normalize_issue(s: str) -> str:

    out = s.strip()
    for pat in _patterns:
        out = re.sub(pat, '하다.', out)
   
    out = re.sub(r'여부\s*$', '하다.', out)
    return out


def summarize(text, issue):

    # sys_prompt = "You are given a document from the Constitutional Court of South Korea, along with the points of contention (disputed issues). Summarize the document in Korean in the formal judicial style of the original ruling.\
    #             Requirements:\
    #             Produce a concise summary of 3–4 sentences.\
    #             The summary must explicitly describe the disputed issue(s), how the court resolves them, and the logical reasoning behind the decision.\
    #             Avoid phrases like “this document discusses” or “the issue is”; write as if it were part of the ruling itself.\
    #             Appropriately preserve constitutional norms, legal terminology, and references to the law.\
    #             Output must be in Korean."
    
    sys_prompt = "You are given a document from the Constitutional Court of South Korea, along with the points of contention (disputed issues). Summarize the document in Korean in the formal judicial style of the original ruling.\
                Requirements:\
                Produce a concise summary of 3–4 sentences.\
                The summary must explicitly describe the disputed issue(s) and how the court resolves them.\
                Each point of the court’s conclusion must follow the structure:\
                '법원은 ~~~ 라고 밝혔다. 왜냐하면 ~~~ 하기 때문이다.'\
                This structure should clearly convey the court’s reasoning.\
                Avoid phrases like “this document discusses” or “the issue is”; write as if it were part of the ruling itself.\
                Appropriately preserve constitutional norms, legal terminology, and references to the law.\
                Output must be in Korean."
    
    user_prompt = f"Disputed issues: {issue}\n\
                    Document: {text}"
    
    return gpt4_call(user_prompt, sys_prompt)









