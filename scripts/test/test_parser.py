import re
import json

PATTERNS = [
    ("num", re.compile(r"^(\d+\.)\s")),              # 1. 2. 3.
    ("kor", re.compile(r"^([가-힣]\.)\s")),       # 가. 나. 다.
    ("paren_num", re.compile(r"^(\(\d+\))\s")),       # (1) (2) (3)
    ("paren_kor", re.compile(r"^(\([가-힣]\))\s")),   # (가) (나) (다)
    ("rparen_num", re.compile(r"^(\d+\))\s")),           # 1) 2) 3)
    ("rparen_kor", re.compile(r"^([가-힣]\))\s")),       # 가) 나) 다)
]


def parse_document(text: str):
    lines = text.splitlines()
    root = {"title": "root", "text": "", "children": []}
    stack = [(None, root)]  # (label, node)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        matched = False
        for label, pattern in PATTERNS:
            m = pattern.match(stripped)
            if m:
                # keep whole thing as title (e.g., "가) 추가 설명")
                node = {
                    "title": stripped,
                    "text": "",
                    "children": []
                }

                # Adjust stack depth according to hierarchy
                while stack and not is_parent(stack[-1][0], label):
                    stack.pop()

                stack[-1][1]["children"].append(node)
                stack.append((label, node))
                matched = True
                break

        if not matched:
            # plain text → append to current node's text
            stack[-1][1]["text"] += (("\n" if stack[-1][1]["text"] else "") + stripped)

    return root


def is_parent(current_label, new_label):
    hierarchy = ["num", "kor", "paren_num", "paren_kor", "rparen_num", "rparen_kor"]
    if current_label is None:
        return True
    return hierarchy.index(new_label) > hierarchy.index(current_label)


# Example usage
doc = """
1. 서론
가. 사실관계
(1) 첫 번째 사항
(가) 세부사항
1) 세부 하위 사항
가) 추가 설명
이 문장은 가) 추가 설명의 본문입니다.
나) 또 다른 설명
2) 두 번째 세부 사항
나. 법리 판단
"""

parsed = parse_document(doc)

print(parsed)
# print(json.dumps(parsed, ensure_ascii=False, indent=2))
