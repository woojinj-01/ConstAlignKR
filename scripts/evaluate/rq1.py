import csv

from src.util import path_keywords, dir_embed_model, embed, get_embedding_model
import src.evaluate as eval
import src.visualize as vis


def evaluation(model_name_base, model_name_tuned, keywords):

    dir_embed = dir_embed_model()

    model_1 = get_embedding_model(dir_embed[model_name_base])
    model_2 = get_embedding_model(dir_embed[model_name_tuned])

    embed_1 = [embed(word, model_1) for word in keywords]
    embed_2 = [embed(word, model_2) for word in keywords]

    print("=== 1. Direct Embedding Shift Analysis ===")
    shift_result = eval.embedding_shift(keywords, embed_1, embed_2)
    vis.plot_embedding_shift(shift_result, keywords)

    print("=== 3. Cluster Structure Preservation ===")
    cluster_result = eval.cluster_preservation(embed_1, embed_2)
    print(cluster_result)

    print("=== 4. Neighborhood Consistency ===")
    neigh_result = eval.neighborhood_consistency(keywords, embed_1, embed_2, k=5)
    vis.plot_neighborhood_consistency(neigh_result, keywords)

    print("=== 5. Dimensionality Reduction (PCA for numerical check) ===")
    pca_result = eval.pca_shift(embed_1, embed_2)
    vis.plot_pca_shift(pca_result, keywords)

    print("=== 6. Embedding Alignment (Procrustes) ===")
    print(eval.procrustes_alignment(embed_1, embed_2))


if __name__ == "__main__":
    with open(path_keywords(), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        keywords = [row[0] for row in reader if row]

        evaluation("KoSimCSE", "KoSimCSE")

        