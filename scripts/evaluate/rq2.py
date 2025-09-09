import csv, json

from src.util import path_keywords, dir_embed_model, embed, get_embedding_model
import src.evaluate as eval
import src.visualize as vis


def evaluation(texts, keywords):

    print("=== 1. Coocurrence Count ===")
    cooc_matrix = eval.cooccurrence_count(texts, keywords)
    vis.plot_cooccurrence_count(cooc_matrix)

    print("=== 2. Cooccurrence Jaccard ===")
    jaccard_matrix = eval.cooccurrence_jaccard(texts, keywords)
    vis.plot_cooccurrence_jaccard(jaccard_matrix)

    print("=== 3. Cooccurence PMI ===")
    pmi_matrix = eval.cooccurrence_pmi(texts, keywords)
    vis.plot_cooccurrence_pmi(pmi_matrix)

    print("=== 4. Conditional Cooccurrence ===")
    cond_matrix = eval.conditional_cooccurrence(texts, keywords)
    vis.plot_conditional_cooccurrence(cond_matrix)

    print("=== 5. Sentence Level Cooccurence Distribution ===")
    freq_list = eval.sentence_level_cooccurrence_distribution(texts, keywords)
    vis.plot_sentence_level_cooccurrence_distribution(freq_list)

    print("=== 6. Keyword Community Detection ===")
    G, node_to_comm = eval.keyword_community_detection(texts, keywords)
    vis.plot_keyword_community_detection(G, node_to_comm)


if __name__ == "__main__":
    with open(path_keywords(), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        keywords = [row[0] for row in reader if row]

    with open('./data/processed/texts.json', 'r', encoding='utf-8') as f:
        texts = json.load(f)

    evaluation(texts, keywords)

        