from torch_geometric.nn import HANConv, HeteroConv, GCNConv, HGTConv
import torch

from torch import nn

from torch_geometric.nn.inits import uniform
import ot
from ot.gromov import semirelaxed_gromov_wasserstein, semirelaxed_fused_gromov_wasserstein, \
    semirelaxed_fused_gromov_wasserstein2, gromov_wasserstein, fused_gromov_wasserstein
from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj
# from torchmetrics.functional import pairwise_cosine_similarity
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
import numpy as np
from ot.gromov._utils import init_matrix, gwloss, gwggrad, init_matrix_semirelaxed, tensor_product
from ot.backend import get_backend
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, Linear, SAGEConv


class HAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata):
        super(HAN, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_channels, metadata, heads=2)
        self.conv2 = HANConv(hidden_channels, out_channels, metadata, heads=2)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x = self.conv1(x_dict, edge_index_dict)
        x = self.conv2(x, edge_index_dict)
        x = x['author']
        return x


class PSAGEConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(PSAGEConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                layer = SAGEConv(input_dim, hidden_dim)
            else:
                layer = SAGEConv(hidden_dim, hidden_dim)

            # init.xavier_uniform_(layer.lin.weight)
            self.layers.append(layer)

    def forward(self, x, edge_index, edge_weight=None):
        z = x

        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)

        return z


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                layer = GCNConv(input_dim, hidden_dim)
            else:
                layer = GCNConv(hidden_dim, hidden_dim)

            # init.xavier_uniform_(layer.lin.weight)
            self.layers.append(layer)

    def forward(self, x, edge_index, edge_weight=None):
        z = x

        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)

        return z


class Encoder(torch.nn.Module):
    def __init__(self, aggregate_encoder, encoder1, encoder2, hidden_dim, device):
        super(Encoder, self).__init__()
        self.device = device
        self.aggregate_encoder = aggregate_encoder
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)
        # self.linear = torch.nn.ModuleList(
        #     [torch.nn.Linear(hidden_dim + 512, hidden_dim) for _ in range(1)])

    def _cal_forward(self, feature, z1, z2, edge_index1, edge_index2, mode, sigma, rho, edge_weight):
        # edge
        C1 = torch.squeeze(to_dense_adj(edge_index1, max_num_nodes=feature.shape[0]))
        # attr
        F1 = feature
        N1l = F1.shape[0]
        N1r = F1.shape[1]
        h1 = ot.unif(N1l, type_as=F1)

        # edge
        C2 = torch.squeeze(to_dense_adj(edge_index2, max_num_nodes=feature.shape[0]))
        # attr
        F2 = feature
        N2l = F2.shape[0]
        N2r = F2.shape[1]
        h2 = ot.unif(N2l, type_as=F2)

        # Mp = torch.cdist(F1, F2, p=2)
        Mp = ot.dist(F1, F2, metric='euclidean')
        # Mb = torch.cdist(z1, z2, p=2)
        Mb = ot.dist(z1, z2, metric='euclidean')

        if sigma < 1:

            # P = fused_gromov_wasserstein(Mp, C1, C2, h1, h2, symmetric=True, alpha=1, log=False)
            P = semirelaxed_fused_gromov_wasserstein(
                Mp, C1, C2, h1, symmetric=True, alpha=1 - sigma, log=False, G0=None)

            nx = get_backend(h1, C1, C2)
            constC, hC1, hC2, fC2t = init_matrix_semirelaxed(
                C1, C2, h1, loss_fun='square_loss', nx=nx)
            OM = torch.ones(N1l, N2l).to(self.device)
            OM = OM / (N1l * N2l)
            qOneM = nx.sum(OM, 0)
            ones_p = nx.ones(h1.shape[0], type_as=h1)
            marginal_product = nx.outer(ones_p, nx.dot(qOneM, fC2t))
            Mp2 = tensor_product(constC + marginal_product, hC1, hC2, P, nx=nx)
            Mp2 = F.normalize(Mp2)
            Mp = (sigma) * Mp + (1 - sigma) * Mp2

            B = ot.emd(h1, h2, Mb)

            gw0, logP = ot.gromov.gromov_wasserstein(C1, C2, h1, h2, 'square_loss', verbose=True, log=True)

            gw, logP = ot.gromov.entropic_gromov_wasserstein(C1, C2, h1, h2, 'square_loss', epsilon=5e-4, log=True, verbose=True)

            # B = ot.optim.cg(h1, h2, Mb, reg=reg, f=f, df=df)
            # B = ot.optim.semirelaxed_cg(h1, h2, Mb, reg=reg, f=f, df=df)

            # kl_loss = nn.KLDivLoss(reduction='batchmean')
            # loss = kl_loss(Mp, Mb)

            sloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            loss = sloss(Mp, Mb)

            # loss = torch.linalg.matrix_norm(Mp - Mb, ord='fro')

            loss = rho * loss + torch.linalg.matrix_norm(P - B, ord='fro')

        elif sigma == 1:
            # speed up
            sl = SamplesLoss(loss='sinkhorn', p=2, debias=True, blur=0.1 ** (1 / 2), backend='tensorized')
            m = 0 * Mb + 1 * Mp
            sl.potentials = True
            u, v = sl(F1, F2)
            P = torch.exp((u.t() + v - m) * 1 / 0.1)
            # P = self.comp(u, v, m)

            sl.potentials = True
            u, v = sl(z1, z2)
            # B = self.comp(u, v, m)
            B = torch.exp((u.t() + v - m) * 1 / 0.1)

            # large data
            # P = ot.emd(h1, h2, Mp)
            # B = ot.emd(h1, h2, Mb)

            # kl_loss = nn.KLDivLoss(reduction='batchmean')
            # loss = kl_loss(Mp, Mb)

            # faster convergence
            sloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            loss = sloss(Mp, Mb)

            # loss = torch.linalg.matrix_norm(Mp - Mb, ord='fro')

            loss = rho * loss + torch.linalg.matrix_norm(P - B, ord='fro')

        # P.requires_grad=True
        # B.requires_grad=True

        return loss

    def forward(self, data, edge_index0, edge_index1, edge_index2, mode, sigma=1, rho=1, edge_weight=None):
        z0 = self.aggregate_encoder(data, edge_index0)
        z1 = self.encoder1(data, edge_index1)
        z2 = self.encoder2(data, edge_index2)

        if mode == 'train':
            loss1 = self._cal_forward(data, z0, z1, edge_index0, edge_index1, mode, sigma, rho, edge_weight)
            loss2 = self._cal_forward(data, z0, z2, edge_index0, edge_index2, mode, sigma, rho, edge_weight)
            loss = loss1 + loss2
        else:
            return z1, z2

        return z1, z2, loss
