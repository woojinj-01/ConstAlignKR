import matplotlib.pyplot as plt
import os, math
import numpy as np
import networkx as nx
from datetime import datetime


def plot_cossim_keywords(clustering_scores, show=False, store=True):
    # Scatter Plot of avg_sim_children vs avg_sim_parent

    labels = list(clustering_scores.keys())
    avg_children = [clustering_scores[l]['avg_sim_children'] for l in labels]
    avg_parent = [clustering_scores[l]['avg_sim_parent'] for l in labels]

    plt.figure(figsize=(10, 8))
    plt.scatter(avg_children, avg_parent, color='blue')

    # annotate each point with its label
    for i, label in enumerate(labels):
        plt.text(avg_children[i]+0.005, avg_parent[i]+0.005, label, fontsize=9)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Avg. Inter-children Cosine Similarity')
    plt.ylabel('Avg. Parent-Children Cosine Similarity')
    
    plt.grid(True)

    if store:
        os.makedirs('./plot', exist_ok=True)

        dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'./plot/rq1_cossim_{dt_str}.png'

        plt.savefig(filename, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
        
    plt.close()


def plot_prcdnt_by_year(case_dict, show=False, store=True):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10)) # 6 subplots
    axes = axes.flatten()

    def get_y_lim(max_val):
        if max_val <= 10:
            return 10
        elif max_val <= 100:
            return 100
        elif max_val <= 500:
            return 500
        else:
            return math.ceil(max_val / 1000) * 1000

    for idx, (case_type, year_counts) in enumerate(case_dict.items()):
        ax = axes[idx]
        years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in years]
        ax.bar([str(y) for y in years], counts, color='#4e79a7')
        ax.set_ylim(0, get_y_lim(max(counts))) # individual y-limit per subplot
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Cases')
        ax.tick_params(axis='x', rotation=45)
        # ax.set_title(case_type)

    plt.tight_layout()
    
    if store:
        os.makedirs('./plot', exist_ok=True)

        dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'./plot/prcdnt_by_year_{dt_str}.png'

        plt.savefig(filename, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
        
    plt.close()


# ----------------------------------------------------------- #
# Research Question 1                                         #
# ----------------------------------------------------------- #

def get_keyword_indices(keywords):
    return list(range(len(keywords)))

# ---------------------------------------------------
# 1. Visualize Direct Embedding Shift (cosine similarity)
# Input: output of embedding_shift
# ---------------------------------------------------


def plot_embedding_shift(shift_results, keywords):
    sims = [shift_results['per_keyword'][kw] for kw in keywords]
    indices = get_keyword_indices(keywords)

    plt.figure(figsize=(12,5))
    plt.bar(indices, sims)
    plt.xlabel('Keyword Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Embedding Shift per Keyword')
    plt.show()

    plt.figure(figsize=(6,4))
    plt.hist(sims, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title('Distribution of Embedding Shifts')
    plt.show()

# ---------------------------------------------------
# 2. Visualize Intra-keyword Semantic Cohesion
# Input: output of intra_group_cohesion
# ---------------------------------------------------


def plot_intra_group_cohesion(cohesion_results):
    groups = [f'Group {i}' for i in range(len(cohesion_results))]
    base_sims = [r['base'] for r in cohesion_results]
    tuned_sims = [r['tuned'] for r in cohesion_results]

    x = np.arange(len(groups))
    width = 0.35

    plt.figure(figsize=(12,5))
    plt.bar(x - width/2, base_sims, width, label='Base')
    plt.bar(x + width/2, tuned_sims, width, label='Tuned')
    plt.xticks(x, groups, rotation=45, ha='right')
    plt.ylabel('Average Cosine Similarity')
    plt.title('Intra-group Semantic Cohesion')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------
# 3. Visualize PCA / 2D embeddings
# Input: output of pca_shift
# Use keyword indices instead of names
# ---------------------------------------------------


def plot_pca_shift(pca_results, keywords, labels=None, title_suffix=''):
    base_2d = pca_results['base_2d']
    tuned_2d = pca_results['tuned_2d']

    indices = get_keyword_indices(keywords)

    plt.figure(figsize=(12,6))
    plt.scatter(base_2d[:,0], base_2d[:,1], c=labels, cmap='tab20', alpha=0.6, label='Base')
    plt.scatter(tuned_2d[:,0], tuned_2d[:,1], c=labels, cmap='tab20', marker='x', alpha=0.8, label='Tuned')

    # Draw arrows from base -> tuned
    for i in range(len(base_2d)):
        plt.arrow(base_2d[i,0], base_2d[i,1], tuned_2d[i,0]-base_2d[i,0], tuned_2d[i,1]-base_2d[i,1],
        color='gray', alpha=0.3, head_width=0.02, length_includes_head=True)

    plt.title('2D PCA Embedding Shift '+title_suffix)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

# ---------------------------------------------------
# 4. Visualize Neighborhood Consistency
# Input: output of neighborhood_consistency
# Use keyword indices
# ---------------------------------------------------


def plot_neighborhood_consistency(neigh_results, keywords):
    overlaps = [neigh_results['per_keyword'][kw] for kw in keywords]
    indices = get_keyword_indices(keywords)

    plt.figure(figsize=(12,5))
    plt.bar(indices, overlaps, color='lightgreen')
    plt.xlabel('Keyword Index')
    plt.ylabel('Neighborhood Overlap (Jaccard)')
    plt.title('Neighborhood Consistency per Keyword')
    plt.show()

    plt.figure(figsize=(6,4))
    plt.hist(overlaps, bins=10, color='lightgreen', edgecolor='black')
    plt.xlabel('Neighborhood Overlap')
    plt.ylabel('Count')
    plt.title('Distribution of Neighborhood Consistency')
    plt.show()

# ---------------------------------------------------
# 5. Visualize Cluster Structure Preservation
# Use indices for coloring or labeling points if needed
# ---------------------------------------------------


def plot_cluster_preservation(cluster_results, keywords, labels_base=None, labels_tuned=None, title_suffix=''):
    ari = cluster_results.get('ARI', None)
    nmi = cluster_results.get('NMI', None)

    print(f"Adjusted Rand Index (Base vs Tuned): {ari:.3f}")
    print(f"Normalized Mutual Information (Base vs Tuned): {nmi:.3f}")

    if labels_base is not None and labels_tuned is not None:
        from sklearn.decomposition import PCA
        indices = get_keyword_indices(keywords)

        plt.figure(figsize=(12,5))
        pca_base = PCA(n_components=2).fit_transform(labels_base)
        pca_tuned = PCA(n_components=2).fit_transform(labels_tuned)

        plt.scatter(pca_base[:,0], pca_base[:,1], c=indices, cmap='tab20', alpha=0.6, label='Base')
        plt.scatter(pca_tuned[:,0], pca_tuned[:,1], c=indices, cmap='tab20', marker='x', alpha=0.8, label='Tuned')
        plt.title('Cluster Structure Visualization '+title_suffix)
        plt.legend()
        plt.show()


# ----------------------------------------------------------- #
# Research Question 2                                         #
# ----------------------------------------------------------- #


def plot_cooccurrence_count(cooc_matrix):
    plt.figure(figsize=(8,6))
    plt.imshow(cooc_matrix, cmap='Reds')
    plt.colorbar(label='Co-occurrence Count')
    plt.title('Keyword Co-occurrence Count')
    plt.xlabel('Keyword Index')
    plt.ylabel('Keyword Index')
    plt.show()


def plot_cooccurrence_jaccard(jaccard_matrix):
    plt.figure(figsize=(8,6))
    plt.imshow(jaccard_matrix, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(label='Jaccard Index')
    plt.title('Keyword Co-occurrence Jaccard Index')
    plt.xlabel('Keyword Index')
    plt.ylabel('Keyword Index')
    plt.show()


def plot_cooccurrence_pmi(pmi_matrix):
    plt.figure(figsize=(8,6))
    plt.imshow(pmi_matrix, cmap='coolwarm')
    plt.colorbar(label='PMI')
    plt.title('Keyword Co-occurrence PMI')
    plt.xlabel('Keyword Index')
    plt.ylabel('Keyword Index')
    plt.show()


def plot_conditional_cooccurrence(cond_matrix):
    plt.figure(figsize=(8,6))
    plt.imshow(cond_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='P(B|A)')
    plt.title('Conditional Co-occurrence Probability')
    plt.xlabel('Keyword Index')
    plt.ylabel('Keyword Index')
    plt.show()


def plot_sentence_level_cooccurrence_distribution(freq_list):
    plt.figure(figsize=(8,5))
    plt.hist(freq_list, bins=range(max(freq_list)+2), align='left', color='orange', edgecolor='black')
    plt.xlabel('Number of Keywords in Sentence')
    plt.ylabel('Frequency')
    plt.title('Sentence-level Co-occurrence Frequency Distribution')
    plt.show()


def plot_keyword_community_detection(G, node_to_comm):
    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(G, seed=42)
    communities = set(node_to_comm.values())
    colors = plt.cm.tab20(np.linspace(0,1,len(communities)))
    for comm, color in zip(communities, colors):
        nodes = [n for n in G.nodes() if node_to_comm[n]==comm]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[color], label=f'Community {comm}', node_size=300)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=8)
    plt.title('Keyword Co-occurrence Community Detection')
    plt.legend()
    plt.show()




