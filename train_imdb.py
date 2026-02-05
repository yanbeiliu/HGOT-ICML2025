import torch_geometric

from model import *
from torch.optim import Adam
from tqdm import tqdm
from aug.eval import get_split, LREvaluator
from utils import *
from torch_geometric.transforms import AddMetaPaths
from dataloader import get_imdb_hg


def test_imdb(encoder_model, data, graph0, graph1, graph2):
    encoder_model.eval()
    z1, z2 = encoder_model(
        data['movie'].x, graph0, graph1, graph2, mode='test')
    z = z1 + z2
    split = get_split(num_samples=z.size()[0], train_ratio=0.6, test_ratio=0.1)
    result = LREvaluator()(z, data.y, split)
    return result, z


def print_model_parameters(model):
    # pass
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Parameter: {name}")
            print(f"Gradients: {param.grad.norm()}")  # 打印梯度的范数
        else:
            print(f"Parameter: {name} has no gradients (perhaps it's frozen or hasn't been used).")


def _train_imdb(encoder_model, data, graph0, graph1, graph2, optimizer, sigma=1.0, rho=1, torchviz=None):
    encoder_model.train()
    optimizer.zero_grad()

    assert not torch.isnan(data['movie'].x).any(), "data['movie'].x contains NaN or Inf"

    z1, z2, loss = encoder_model(
        data['movie'].x, graph0, graph1, graph2, mode='train', sigma=sigma, rho=rho)

    loss.backward()

    optimizer.step()

    return loss.item()


def z_score_normalize(x):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    x_normalized = (x - mean) / std
    return x_normalized


def min_max_normalize(x):
    min_val = x.min(dim=0, keepdim=True)[0]
    max_val = x.max(dim=0, keepdim=True)[0]
    x_normalized = (x - min_val) / (max_val - min_val)
    return x_normalized


def train_imdb(device):
    data = get_imdb_hg()
    data.train_mask, data.val_mask, data.test_mask = (data['movie'].train_mask,
                                                      data['movie'].val_mask, data['movie'].test_mask)
    data.y = data['movie'].y

    metapaths = [[("movie", "actor"), ("actor", "movie")],
                 [("movie", "director"), ("director", "movie")]]
    data = AddMetaPaths(metapaths)(data)

    # print(data.x_dict)
    # print(data.edge_index_dict)

    edge_index1 = data.edge_index_dict[('movie', 'metapath_0', 'movie')]
    edge_index2 = data.edge_index_dict[('movie', 'metapath_1', 'movie')]

    num_nodes = data['movie'].x.shape[0]

    print("Number of nodes:", num_nodes)

    combined_edge_index = torch.cat([edge_index1, edge_index2], dim=1)
    combined_edge_index = torch.unique(combined_edge_index, dim=1)  # 去重

    print("edge_index1:", edge_index1.min(), edge_index1.max())
    print("edge_index2:", edge_index2.min(), edge_index2.max())
    print("combined_edge_index:", combined_edge_index.min(), combined_edge_index.max())

    edge_index1, _ = torch_geometric.utils.add_self_loops(edge_index1)
    edge_index2, _ = torch_geometric.utils.add_self_loops(edge_index2)
    combined_edge_index, _ = torch_geometric.utils.add_self_loops(combined_edge_index)

    print(data['movie'].x.min(), data['movie'].x.max(), combined_edge_index.shape)

    data[('movie', 'metapath_2', 'movie')] = combined_edge_index

    data.to(device)

    input_dim = data['movie'].x.shape[1]
    print('input dimesion:==========================', input_dim)
    gconv0 = PSAGEConv(input_dim, 64, 2).to(device)
    gconv1 = PSAGEConv(input_dim, 64, 2).to(device)
    gconv2 = PSAGEConv(input_dim, 64, 2).to(device)

    encoder_model = Encoder(gconv0, gconv1, gconv2, 64, device).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=0.0001)

    best = {'acc': 0.0, 'micro_f1': 0.0, 'macro_f1': 0.0}
    epoch_num = 600
    best_embeddings = None

    with tqdm(total=epoch_num, desc='(T)') as pbar:
        for epoch in range(1, epoch_num+1):
            sigma = 1
            rho = 1
            loss = _train_imdb(encoder_model, data, combined_edge_index.to(device),
                               edge_index1.to(device), edge_index2.to(device), optimizer, sigma, rho)

            pbar.set_postfix({'loss': loss})
            pbar.update()

            if epoch % 2 == 0:
                test_result, z = test_imdb(encoder_model, data, combined_edge_index.to(device),
                                        edge_index1.to(device), edge_index2.to(device))
                print(f'Best test ACC={test_result["acc"]:.4f}, '
                      f'micro_f1={test_result["micro_f1"]:.4f}, '
                      f'macro_f1={test_result["macro_f1"]:.4f}')

                if test_result["acc"] > best["acc"]:
                    best["acc"] = test_result["acc"]
                    best["micro_f1"] = test_result["micro_f1"]
                    best["macro_f1"] = test_result["macro_f1"]
                    best_embeddings = z.detach().cpu().numpy()

    print(f'The final best acc is:{best["acc"]:.4f}, micro_f1:{best["micro_f1"]:.4f}, macro_f1: {best["macro_f1"]:.4f}')
    draw_diagrams(data['movie'].x.cpu(), best_embeddings, data.y.cpu(), 'imdb')
    return best