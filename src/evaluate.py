import os, json
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import networkx as nx
import kss

from src.util import PRCDNT_TYPES, dir_prcdnt, path_prcdnt_list


# ----------------------------------------------------------- #
# Statistics                                                  #
# ----------------------------------------------------------- #


def prcdnt_by_year():

    case_types = PRCDNT_TYPES

    case_dict = {}

    for case_type in case_types:
        dir_path = dir_prcdnt(case_type)
        case_dict[case_type] = {}

        if not os.path.exists(dir_path):
            continue

        for filename in os.listdir(dir_path):
            if filename[:4].isdigit():
                year = int(filename[:4])
            elif filename[:2].isdigit():
                year = int(filename[:2])
                year = 1900 + year
            else:
                continue

            if year not in case_dict[case_type]:
                case_dict[case_type][year] = 0

            case_dict[case_type][year] += 1

    return case_dict


def num_valid_prcdnts():
    target_type = PRCDNT_TYPES

    count_sum = 0

    for prcdnt_type in target_type:
        print(f"\n==={prcdnt_type}===")
        file_path = path_prcdnt_list(prcdnt_type)
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            items = data["body"]["items"]["item"]

            count = sum(
                1 for case in items
                if case.get("jgdmtCort") == "전원재판부" and case.get("rstaRsta") != "각하"
            )
        
            count_sum += count

            print(f"Number of valid cases: {count}")

    print("\n===Total===")
    print(f"Number of valid cases: {count_sum}")


def num_valid_prcdnts_with_other_opinions():
    target_type = PRCDNT_TYPES

    count_sum = 0

    for prcdnt_type in target_type:
        print(f"\n==={prcdnt_type}===")
        file_path = dir_prcdnt(prcdnt_type)

        count = 0

        for filename in os.listdir(file_path):
            full_path = os.path.join(file_path, filename)

            if os.path.isfile(full_path):  # skip subdirectories if any
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if len(data["body"]["items"]["item"]) == 0:
                        continue

                    if data["body"]["items"]["item"][0]["otherOpinions"] is None:
                        continue

                    count += 1
        
        count_sum += count

        print(f"Number of valid cases: {count}")

    print("\n===Total===")
    print(f"Number of valid cases: {count_sum}")

# ----------------------------------------------------------- #
# Research Question 1                                         #
# ----------------------------------------------------------- #


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# ---------------------------------------------------
# 1. Direct Embedding Shift Analysis
# ---------------------------------------------------


def embedding_shift(keywords, emb_base, emb_tuned):

    emb_base = [to_numpy(e) for e in emb_base]
    emb_tuned = [to_numpy(e) for e in emb_tuned]

    sims = []
    for i in range(len(keywords)):
        sim = cosine_similarity([emb_base[i]], [emb_tuned[i]])[0][0]
        sims.append(sim)
    return {
        "mean_similarity": np.mean(sims),
        "std_similarity": np.std(sims),
        "per_keyword": dict(zip(keywords, sims))
    }

# ---------------------------------------------------
# 2. Intra-keyword Semantic Cohesion
# (requires groups of related keywords)
# groups = [["word1", "word2"], ["word3", "word4", "word5"]]
# ---------------------------------------------------


def intra_group_cohesion(groups, keywords, emb_base, emb_tuned):
    kw2idx = {kw: i for i, kw in enumerate(keywords)}

    def avg_similarity(group, embs):
        sims = []
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                sims.append(
                    cosine_similarity([embs[kw2idx[group[i]]]], [embs[kw2idx[group[j]]]])[0][0]
                )
        return np.mean(sims) if sims else None

    results = []
    for g in groups:
        base_sim = avg_similarity(g, emb_base)
        tuned_sim = avg_similarity(g, emb_tuned)
        results.append({"group": g, "base": base_sim, "tuned": tuned_sim})
    return results

# ---------------------------------------------------
# 3. Cluster Structure Preservation
# ---------------------------------------------------


def cluster_preservation(emb_base, emb_tuned, n_clusters=5):

    emb_base = [to_numpy(e) for e in emb_base]
    emb_tuned = [to_numpy(e) for e in emb_tuned]

    km_base = KMeans(n_clusters=n_clusters, random_state=0).fit(emb_base)
    km_tuned = KMeans(n_clusters=n_clusters, random_state=0).fit(emb_tuned)

    ari = adjusted_rand_score(km_base.labels_, km_tuned.labels_)
    nmi = normalized_mutual_info_score(km_base.labels_, km_tuned.labels_)
    return {"ARI": ari, "NMI": nmi}

# ---------------------------------------------------
# 4. Neighborhood Consistency
# ---------------------------------------------------


def neighborhood_consistency(keywords, emb_base, emb_tuned, k=5):

    emb_base = [to_numpy(e) for e in emb_base]
    emb_tuned = [to_numpy(e) for e in emb_tuned]

    sims_base = cosine_similarity(emb_base)
    sims_tuned = cosine_similarity(emb_tuned)

    overlaps = []
    for i in range(len(keywords)):
        nn_base = np.argsort(-sims_base[i])[1:k+1]  # exclude self
        nn_tuned = np.argsort(-sims_tuned[i])[1:k+1]
        overlap = len(set(nn_base) & set(nn_tuned)) / k
        overlaps.append(overlap)

    return {
        "mean_overlap": np.mean(overlaps),
        "per_keyword": dict(zip(keywords, overlaps))
    }

# ---------------------------------------------------
# 5. Dimensionality Reduction (PCA for numerical check)
# ---------------------------------------------------


def pca_shift(emb_base, emb_tuned, n_components=2):

    emb_base = [to_numpy(e) for e in emb_base]
    emb_tuned = [to_numpy(e) for e in emb_tuned]

    pca = PCA(n_components=n_components)
    base_2d = pca.fit_transform(emb_base)
    tuned_2d = pca.fit_transform(emb_tuned)
    return {"base_2d": base_2d, "tuned_2d": tuned_2d}

# ---------------------------------------------------
# 6. Embedding Alignment (Procrustes)
# ---------------------------------------------------


def procrustes_alignment(emb_base, emb_tuned):

    emb_base = [to_numpy(e) for e in emb_base]
    emb_tuned = [to_numpy(e) for e in emb_tuned]

    _, _, disparity = procrustes(emb_base, emb_tuned)
    return {"disparity": disparity}


# ----------------------------------------------------------- #
# Research Question 2                                         #
# ----------------------------------------------------------- #


def cooccurrence_count(texts, keywords):
    n = len(keywords)
    cooc_matrix = np.zeros((n,n), dtype=int)

    for text in texts:
        sentences = kss.split_sentences(text)
        for sent in sentences:
            present = [i for i, kw in enumerate(keywords) if kw in sent]
            for i,j in combinations(present,2):
                cooc_matrix[i,j] += 1
                cooc_matrix[j,i] += 1
    return cooc_matrix


def cooccurrence_jaccard(texts, keywords):
    n = len(keywords)
    sentence_sets = [set() for _ in range(n)]

    for text_idx, text in enumerate(texts):
        sentences = kss.split_sentences(text)
        for s_idx, sent in enumerate(sentences):
            for i, kw in enumerate(keywords):
                if kw in sent:
                    sentence_sets[i].add((text_idx,s_idx))

    jaccard_matrix = np.zeros((n,n))
    for i,j in combinations(range(n),2):
        intersection = len(sentence_sets[i].intersection(sentence_sets[j]))
        union = len(sentence_sets[i].union(sentence_sets[j]))
        jaccard_matrix[i,j] = intersection/union if union>0 else 0
        jaccard_matrix[j,i] = jaccard_matrix[i,j]
    np.fill_diagonal(jaccard_matrix, 1.0)
    return jaccard_matrix


def cooccurrence_pmi(texts, keywords):
    n = len(keywords)
    cooc_matrix = np.zeros((n,n))
    counts = np.zeros(n)
    total_sentences = 0

    for text in texts:
        sentences = kss.split_sentences(text)
        total_sentences += len(sentences)
        for sent in sentences:
            present = [i for i, kw in enumerate(keywords) if kw in sent]
            for i in present:
                counts[i] += 1
            for i,j in combinations(present,2):
                cooc_matrix[i,j] += 1
                cooc_matrix[j,i] += 1

    pmi_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if cooc_matrix[i,j] > 0:
                p_i = counts[i]/total_sentences
                p_j = counts[j]/total_sentences
                p_ij = cooc_matrix[i,j]/total_sentences
                pmi_matrix[i,j] = np.log(p_ij/(p_i*p_j))
    np.fill_diagonal(pmi_matrix, 0)
    return pmi_matrix


def conditional_cooccurrence(texts, keywords):
    n = len(keywords)
    counts = np.zeros(n)
    cooc_matrix = np.zeros((n,n))

    for text in texts:
        sentences = kss.split_sentences(text)
        for sent in sentences:
            present = [i for i, kw in enumerate(keywords) if kw in sent]
            for i in present:
                counts[i] += 1
            for i,j in combinations(present,2):
                cooc_matrix[i,j] += 1
    cond_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            cond_matrix[i,j] = cooc_matrix[i,j]/counts[i] if counts[i]>0 else 0
    return cond_matrix


def sentence_level_cooccurrence_distribution(texts, keywords):
    freq = []
    for text in texts:
        sentences = kss.split_sentences(text)
        for sent in sentences:
            count = sum(1 for kw in keywords if kw in sent)
            freq.append(count)
    return freq


def keyword_community_detection(texts, keywords, threshold=1):
    n = len(keywords)
    cooc_matrix = np.zeros((n,n), dtype=int)

    for text in texts:
        sentences = kss.split_sentences(text)
        for sent in sentences:
            present = [i for i, kw in enumerate(keywords) if kw in sent]
            for i,j in combinations(present,2):
                cooc_matrix[i,j] += 1
                cooc_matrix[j,i] += 1

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, label=keywords[i])
    for i in range(n):
        for j in range(i+1, n):
            if cooc_matrix[i,j] >= threshold:
                G.add_edge(i, j, weight=cooc_matrix[i,j])

    from networkx.algorithms import community
    communities = list(community.greedy_modularity_communities(G, weight='weight'))

    node_to_comm = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = idx

    return G, node_to_comm