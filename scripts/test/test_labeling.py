import re
import json

# Keyword dictionaries (expand as needed)
SECTION_MAP = {
    "applicant": ["청구인의 주장", "청구취지", "청구이유"],
    "state": ["피청구인 주장", "정부의 의견", "국가의 항변", "국가의 주장"],
    "court": ["법리 판단", "이유", "재판부의 판단", "헌법재판소의 판단"]
}

ISSUE_KEYWORDS = ["쟁점", "문제", "여부", "위헌"]

def classify_section(title: str):
    """Return applicant/state/court if title matches, else None"""
    for label, keywords in SECTION_MAP.items():
        for kw in keywords:
            if kw in title:
                return label
    return None

def is_issue(title: str):
    """Check if this section looks like an 'issue'"""
    return any(kw in title for kw in ISSUE_KEYWORDS)

def gather_arguments(parsed_json):
    """Traverse parsed JSON and extract issue-by-issue summaries"""
    results = []

    def traverse(node, current_issue=None):
        title = node.get("title", "")
        text = node.get("text", "").strip()

        # Detect new issue
        if is_issue(title):
            current_issue = {
                "issue": title,
                "applicant": "",
                "state": "",
                "court": ""
            }
            results.append(current_issue)

        # Classify section
        label = classify_section(title)
        if label and current_issue:
            current_issue[label] += ("\n" + text if current_issue[label] else text)

        # If no explicit section label but there is text under issue root
        elif current_issue and text and not node.get("children"):
            # heuristic: if text exists directly under issue, treat as court reasoning
            current_issue["court"] += ("\n" + text if current_issue["court"] else text)

        # Recurse children
        for child in node.get("children", []):
            traverse(child, current_issue)

    traverse(parsed_json)
    return results


# === Example Usage ===
doc = {
    "title": "root",
    "text": "",
    "children": [
        {
            "title": "쟁점1: 인터넷 실명제 위헌 여부",
            "text": "",
            "children": [
                {"title": "청구인의 주장", "text": "표현의 자유 침해라고 주장한다.", "children": []},
                {"title": "국가의 항변", "text": "건전한 여론 형성을 위해 필요하다고 주장한다.", "children": []},
                {"title": "법리 판단", "text": "과잉금지원칙에 위배되어 위헌이다.", "children": []}
            ]
        },
        {
            "title": "쟁점2: 개인정보자기결정권 침해 여부",
            "text": "",
            "children": [
                {"title": "청구이유", "text": "개인정보 자기결정권 침해라고 본다.", "children": []},
                {"title": "헌법재판소의 판단", "text": "역시 위헌이다.", "children": []}
            ]
        }
    ]
}

issues = gather_arguments(doc)
print(json.dumps(issues, ensure_ascii=False, indent=2))
