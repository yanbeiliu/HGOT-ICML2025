import pickle
import os

import torch
import networkx as nx
from torch_geometric.data import HeteroData
import dgl
import numpy as np
import random
from dataloader import get_dblp_hg
from model import Encoder, GConv
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，这个后端适用于生成图像文件但不显示它们
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from aug.eval import get_split, LREvaluator
import seaborn as sns


def induce_subgraph_from_metapath(hetero_data: HeteroData, metapath: list):
    """
    根据给定的元路径在异质图数据中生成诱导子图。

    参数:
    - hetero_data: HeteroData, 包含异质图数据的对象
    - metapath: list, 表示元路径的节点类型和边类型

    返回:
    - nx.Graph, 根据元路径生成的诱导子图
    """
    # 初始化一个空图
    subgraph = nx.Graph()

    # 初始化当前节点集合
    current_nodes = torch.arange(hetero_data[metapath[0]].num_nodes)

    for i in range(0, len(metapath) - 2, 2):
        src_type = metapath[i]
        edge_type = (metapath[i], metapath[i + 1], metapath[i + 2])
        tgt_type = metapath[i + 2]

        # 获取对应类型的边
        edge_index = hetero_data[edge_type].edge_index

        # 保留与当前节点集合相关的目标节点
        mask = torch.isin(edge_index[0], current_nodes)
        filtered_edge_index = edge_index[:, mask]

        # 添加新的边到subgraph中
        for src, tgt in filtered_edge_index.T:
            subgraph.add_edge(src.item(), tgt.item())

        # 更新当前节点集合为目标节点
        current_nodes = filtered_edge_index[1]

    return subgraph


def merge_graphs(graph1, graph2):
    """
    合并两个 NetworkX 图成为一个合成图。

    参数:
    - graph1: NetworkX 图对象
    - graph2: NetworkX 图对象

    返回:
    - NetworkX 图对象, 合成后的图
    """
    # 创建一个新的图对象
    combined_graph = nx.Graph()

    # 添加第一个图的节点和边
    combined_graph.add_nodes_from(graph1.nodes(data=True))
    combined_graph.add_edges_from(graph1.edges(data=True))

    # 添加第二个图的节点和边
    combined_graph.add_nodes_from(graph2.nodes(data=True))
    combined_graph.add_edges_from(graph2.edges(data=True))

    return combined_graph


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_dimensionality(data, method='tsne', n_components=2):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=30, learning_rate=0.001)
    else:
        raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")
    return reducer.fit_transform(data)


# t-SNE Visualization
def draw_cluster_diagram(embeddings, labels, fig_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', s=10)
    plt.title('Visualization')
    plt.savefig(fig_name)


def draw_diagrams(features, embeddings, labels, prefix):
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings)

    acc = accuracy_score(labels, kmeans_labels)
    nmi = normalized_mutual_info_score(labels, kmeans_labels)
    ari = adjusted_rand_score(labels, kmeans_labels)
    print(f"Accuracy: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}")

    directory = os.path.join(os.getcwd(), "pictures")
    if not os.path.exists(directory):
        os.makedirs(directory)

    embeddings = reduce_dimensionality(embeddings)
    draw_cluster_diagram(embeddings, kmeans_labels, os.path.join(directory, prefix + '_kmeans.png'))

    features = reduce_dimensionality(features)
    draw_cluster_diagram(features, kmeans_labels, os.path.join(directory, prefix + '_original.png'))


def generate_random_colors(n):
    colors = np.random.rand(n, 3) * 0.7
    return colors


def draw_box_diagrams(data, names):
    plt.figure(figsize=(8,6))
    box = plt.boxplot(data, patch_artist=True, showfliers=False)

    colors = generate_random_colors(len(names))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)  # 设置箱子的填充颜色

    plt.title("Multiple Box Plot Example")
    plt.xticks([i for i in range(1, len(data) + 1)], names)
    plt.ylabel("Values")
    plt.ylim(0.8, 1.0)

    y_ticks = plt.yticks()[0]
    for median in y_ticks:
        plt.axhline(y=median, color='grey', linestyle='-', linewidth=1)

    plt.savefig('pictures/box.png')


if __name__ == '__main__':
    database = ['dblp','imdb','acm','yelp'] # 'yelp', 'dblp', 'imdb', 'acm', 'aminer', 'freebase'
    all_acc_data = []
    for db in database:
        with open('pkl/' + db + '.pkl', 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)
            all_acc_data.append(loaded_data)

    acc_array = [data["acc"] for data in all_acc_data]
    draw_box_diagrams(acc_array, database)
