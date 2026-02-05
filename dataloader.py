import os.path
import os.path as osp
import numpy as np
from torch_geometric.transforms import AddMetaPaths
import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets import DBLP, IMDB, OGB_MAG, RCDD, Taobao
from dgl.data.utils import _get_dgl_url, download, get_download_dir
import pickle
import scipy.io
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
from collections import defaultdict


def get_dblp_hg():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset/dblp')
    # We initialize conference node features with a single one-vector as feature:
    # dataset = DBLP(path, transform=T.Constant(node_types='conference'))
    dataset = DBLP(path)
    data = dataset[0]

    # 4 node types: "paper", "author", "conference", and "term"
    # 6 edge types: ("paper","author"), ("author", "paper"),
    #               ("paper, "term"), ("paper", "conference"),
    #               ("term, "paper"), ("conference", "paper")
    # Add two metapaths:
    # 1. From "author" to "author" through "paper"
    # 2. From "author" to "conference" through "paper"
    metapaths = [[("author", "paper"), ("paper", "term"), ("term", "paper"), ("paper", "author")],
                 [("author", "paper"), ("paper", "conference"), ("conference", "paper"), ("paper", "author")]]
    data = AddMetaPaths(metapaths)(data)
    return data


def get_imdb_hg():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset/IMDB')
    # metapaths = [[('movie', 'actor'), ('actor', 'movie')],
    #              [('movie', 'director'), ('director', 'movie')]]
    # transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
    #                            drop_unconnected_node_types=True)
    # dataset = IMDB(path, transform=transform)
    dataset = IMDB(path)
    return dataset[0]


def get_acm_hg():
    url = "dataset/ACM3025.pkl"
    data_path = get_download_dir() + "/ACM3025.pkl"
    if not os.path.exists(data_path):
        download(_get_dgl_url(url), path=data_path)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    hg_data = HeteroData()
    # keys: label, feature, PAP, PLP
    labels = torch.from_numpy(data["label"].todense()).float()
    hg_data.y = np.argmax(labels, axis=1)

    hg_data.x = torch.from_numpy(data["feature"].todense()).float()

    adj1 = torch.from_numpy(data["PAP"].todense()).float()
    edge_index1 = np.nonzero(adj1)
    hg_data.edge_index1 = edge_index1.t()

    adj2 = torch.from_numpy(data["PLP"].todense()).float()
    edge_index2 = np.nonzero(adj2)
    hg_data.edge_index2 = edge_index2.t()

    combined_edge_index = torch.cat([hg_data.edge_index1, hg_data.edge_index2], dim=1)
    combined_edge_index = torch.unique(combined_edge_index, dim=1)  # 去重
    hg_data.combined_edge_index = combined_edge_index

    return hg_data


def get_yelp_hg(path1='BUB', path2='BRB'):
    # BUB, BRB, BSB, BLB
    mat_data = scipy.io.loadmat('dataset/yelp2614.mat')

    hg_data = HeteroData()
    hg_data.x = torch.from_numpy(mat_data["features"]).float()
    hg_data.y = torch.tensor(np.argmax(mat_data['label'], axis=1))

    hg_data.edge_index1 = torch.tensor(np.vstack(np.nonzero(mat_data[path1])))

    hg_data.edge_index2 = torch.tensor(np.vstack(np.nonzero(mat_data[path2])))

    combined_edge_index = torch.cat([hg_data.edge_index1, hg_data.edge_index2], dim=1)
    combined_edge_index = torch.unique(combined_edge_index, dim=1)  # 去重
    hg_data.combined_edge_index = combined_edge_index

    return hg_data


def convert_sparse_to_tensor(sparse_matrix):
    sparse_matrix = sparse_matrix.tocoo()
    indices = torch.tensor(np.array([sparse_matrix.row, sparse_matrix.col]), dtype=torch.int64)

    return indices


def get_aminer_hg():
    # The order of node types: 0 p 1 a 2 r
    path = "dataset/aminer/"
    label = np.load(path + "labels.npy").astype('int64')

    hg_data = HeteroData()
    hg_data.x = torch.tensor(sp.eye(len(label)).toarray()).float()
    hg_data.y = torch.tensor(label)

    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")

    hg_data.edge_index1 = convert_sparse_to_tensor(pap).clone().detach()
    hg_data.edge_index2 = convert_sparse_to_tensor(prp).clone().detach()

    combined_edge_index = torch.cat([hg_data.edge_index1, hg_data.edge_index2], dim=1)
    combined_edge_index = torch.unique(combined_edge_index, dim=1)  # 去重
    hg_data.combined_edge_index = combined_edge_index

    return hg_data


def get_freebase_hg():
    # The order of node types: 0 p 1 a 2 r
    path = "dataset/freebase/"
    label = np.load(path + "labels.npy").astype('int64')

    hg_data = HeteroData()
    hg_data.x = torch.tensor(sp.eye(len(label)).toarray()).float()
    hg_data.y = torch.tensor(label)

    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")

    hg_data.edge_index1 = convert_sparse_to_tensor(mam).clone().detach()
    hg_data.edge_index2 = convert_sparse_to_tensor(mdm).clone().detach()

    combined_edge_index = torch.cat([hg_data.edge_index1, hg_data.edge_index2], dim=1)
    combined_edge_index = torch.unique(combined_edge_index, dim=1)  # 去重
    hg_data.combined_edge_index = combined_edge_index

    return hg_data


if __name__ == '__main__':
    data = get_dblp_hg()
    print(data)
    # print(data.y, data.y.dtype)
    # data = get_aminer_hg()
    # print(data.y, type(data.y[0]))

