import json, torch
from rich.tree import Tree
# from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.util import cos_sim
import os, openai
from dotenv import load_dotenv

PRCDNT_TYPES = ["헌가", "헌나", "헌다", "헌라", "헌마", "헌바"]


def load_openai_key():
    if not (hasattr(openai, "api_key") and openai.api_key):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")


def load_openapi_key():
    load_dotenv()
    return os.getenv("OPEN_API_KEY")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
        

def load_triplets(jsonl_path):
    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            examples.append(InputExample(
                texts=[obj["anchor"], obj["positive"], obj["negative"]]
            ))
    return examples


def dir_prcdnt(panre_type):
    dir_path = "./data/raw/prcdnt"
    return os.path.join(dir_path, panre_type)


def dir_prcdnt_list():
    dir_path = "./data/raw/prcdnt_list"
    return dir_path


def dir_processed():
    dir_path = "./data/processed"
    return dir_path


def path_prcdnt(panre_type, panre_num):
    return os.path.join(dir_prcdnt(panre_type), panre_num + ".json")


def path_keywords():
    return os.path.join(dir_processed(), "keywords" + ".csv")


def path_triplet():
    return os.path.join(dir_processed(), "triplet" + ".json")


def path_prcdnt_list(prcdnt_type):
    return os.path.join(dir_prcdnt_list(), prcdnt_type + ".json")


def dir_embed_model():
    embedding_model_paths = {
        "KoSimCSE": "models/before_tuned/kosimcse",
        "BGE-M3-KOR": "models/before_tuned/bge-m3-korean",
        "Multilingual-E5-Large": "models/before_tuned/multilingual-e5-large"
    }
    return embedding_model_paths


def cosine(vec1, vec2):
    return cos_sim(vec1, vec2).item()


def get_embedding_model(path_or_name="models/before_tuned/kosimcse"):
    return SentenceTransformer(path_or_name)


def embed(word, model):
    return model.encode(word, convert_to_tensor=True, normalize_embeddings=True)


def embed_long_text(text, model, chunk_size=250, pooling=True):

    # Split text into words
    words = text.split()
    embeddings = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        emb = embed(chunk, model)
        embeddings.append(emb)
        
    if pooling:
        return torch.stack(embeddings).mean(dim=0)
    else:
        return embeddings


def gpt4_call(user_prompt, sys_prompt=None, model="gpt-4"):

    load_openai_key()

    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}] if sys_prompt is None
        else [{"role": "system", "content": sys_prompt},
              {"role": "user", "content": user_prompt}]
    )
    return response.choices[0].message.content


def build_tree(data, tree=None):
    if tree is None:
        tree = Tree(data.get("title", "root"))

    # Show text if exists
    if data.get("text"):
        tree.add(f"[dim]{data['text']}[/dim]")

    # Recurse into children
    for child in data.get("children", []):
        subtree = tree.add(child["title"])
        build_tree(child, subtree)
    return tree


def iter_nodes(node, parent_code=None):
    yield {"code": node["code"], "name": node["name"], "parent": parent_code}
    for child in node.get("children", []):
        yield from iter_nodes(child, node["code"])
