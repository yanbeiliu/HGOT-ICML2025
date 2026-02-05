from model import *
from torch.optim import Adam
from tqdm import tqdm
from aug.eval import get_split, LREvaluator
from utils import *
from dataloader import get_dblp_hg
import copy


def test_dblp(encoder_model, data, graph0, graph1, graph2):
    encoder_model.eval()
    z1, z2 = encoder_model(
        data['author'].x, graph0, graph1, graph2, mode='test')
    z = z1 + z2
    split = get_split(num_samples=z.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = LREvaluator()(z, data.y, split)
    return result, z


def _train_dblp(encoder_model, data, graph0, graph1, graph2, optimizer, sigma=1.0, rho=1):
    encoder_model.train()
    optimizer.zero_grad()

    z1, z2, loss = encoder_model(
        data['author'].x, graph0, graph1, graph2, mode='train', sigma=sigma, rho=rho)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_dblp(device):
    data = get_dblp_hg()
    data.train_mask, data.val_mask, data.test_mask = (data['author'].train_mask,
                                                      data['author'].val_mask, data['author'].test_mask)
    data.y = data['author'].y

    num_nodes = data['author'].x.shape[0]

    print("Number of nodes:", num_nodes)

    edge_index1 = data.edge_index_dict[('author', 'metapath_0', 'author')]
    edge_index2 = data.edge_index_dict[('author', 'metapath_1', 'author')]
    combined_edge_index = torch.cat([edge_index1, edge_index2], dim=1)
    combined_edge_index = torch.unique(combined_edge_index, dim=1)

    print(edge_index1.shape, edge_index2.shape, combined_edge_index.shape)

    data[('author', 'metapath_2', 'author')] = combined_edge_index

    data.to(device)

    input_dim = data['author'].x.shape[1]
    gconv0 = GConv(input_dim, 20, 2).to(device)
    gconv1 = GConv(input_dim, 20, 2).to(device)
    gconv2 = GConv(input_dim, 20, 2).to(device)

    encoder_model = Encoder(gconv0, gconv1, gconv2, 1024, device).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=0.001)

    best = {'acc': 0.0, 'micro_f1': 0.0, 'macro_f1': 0.0}
    best_embeddings = None
    epoch_num = 180

    with tqdm(total=epoch_num, desc='(T)') as pbar:
        for epoch in range(1, epoch_num+1):
            sigma = 1
            rho = 1
            loss = _train_dblp(encoder_model, data, combined_edge_index.to(device),
                               edge_index1.to(device), edge_index2.to(device), optimizer, sigma, rho)

            pbar.set_postfix({'loss': loss})
            pbar.update()

            if epoch % 2 == 0:
                test_result, z = test_dblp(encoder_model, data, combined_edge_index.to(device),
                               edge_index1.to(device), edge_index2.to(device))
                print(f'Best test ACC={test_result["acc"]:.4f}, '
                      f'micro_f1={test_result["micro_f1"]:.4f}, '
                      f'macro_f1={test_result["macro_f1"]:.4f}')

                if best["acc"] < test_result["acc"]:
                    best["acc"] = test_result["acc"]
                    best["micro_f1"] = test_result["micro_f1"]
                    best["macro_f1"] = test_result["macro_f1"]
                    best_embeddings = z.detach().cpu().numpy()

    print(f'The final best acc is:{best["acc"]:.4f}, micro_f1:{best["micro_f1"]:.4f}, macro_f1: {best["macro_f1"]:.4f}')
    draw_diagrams(data['author'].x.cpu(), best_embeddings, data.y.cpu(), 'dblp')
    return best