import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from torch_scatter import scatter, scatter_add
from torch_cluster import radius_graph


def get_device(tensor):
    int = tensor.get_device()
    if int == 0:
        return 'cuda'
    elif int == -1:
        return 'cpu'
    else:
        return None

class GaussianSmearing(nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, device='cpu'):
        super().__init__()
        self.device = device
        mu = torch.linspace(start, stop, num_gaussians).to(self.device)
        self.coeff = -0.5 / (mu[1] - mu[0]).item() ** 2
        self.register_buffer('mu', mu)

    def forward(self, dist):
        # right now mixed devices
        dist = dist.to(self.device)
        dist = dist.view(-1, 1) - self.mu.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class TensorProductConvLayer(nn.Module):

    def __init__(self, in_irreps, sh_irreps, out_irreps, edge_fdim, residual=True, dropout=0.0,
                 h_dim=None, relu_in_fc=True):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if h_dim is None:
            h_dim = edge_fdim

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        if relu_in_fc:
            self.fc_net = nn.Sequential(
                nn.Linear(edge_fdim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h_dim, tp.weight_numel)
            )
        else:
            self.fc_net = nn.Sequential(
                nn.Linear(edge_fdim, h_dim),
                nn.Dropout(dropout),
                nn.Linear(h_dim, tp.weight_numel)
            )

    def forward(self, x, edge_index, edge_attr, edge_sh, out_nodes=None, aggr='mean'):
        edge_src, edge_dst = edge_index
        tp_out = self.tp(x[edge_src], edge_sh, self.fc_net(edge_attr))

        out_nodes = out_nodes or x.shape[0]

        out = scatter(src=tp_out, index=edge_dst, dim=0, dim_size=out_nodes, reduce=aggr)
        # assert out.shape[0] == x.shape[0]

        if self.residual:
            padded = F.pad(x, (0, out.shape[-1] - x.shape[-1]))
            out = out + padded

        return out


class EquiReact(nn.Module):

    def __init__(self, node_fdim: int, edge_fdim: int, sh_lmax: int = 2,
                 n_s: int = 16, n_v: int = 16, n_conv_layers: int = 2,
                 max_radius: float = 10.0, max_neighbors: int = 20,
                 distance_emb_dim: int = 32, dropout_p: float = 0.1,
                 sum_mode='node', verbose=False, device='cpu', graph_mode='energy',
                 random_baseline=False, invariant=False,
                 classification = False, arch='normal',
                 **kwargs):

        super().__init__(**kwargs)

        if invariant:
            sh_lmax = 0

        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.n_s, self.n_v = n_s, n_v
        self.n_conv_layers = n_conv_layers
        self.sum_mode = sum_mode
        self.n_s_full = 2 * self.n_s if self.n_conv_layers >= 3 else self.n_s
        self.distance_emb_dim = distance_emb_dim
        self.max_radius = max_radius
        self.max_neighbors = max_neighbors
        self.verbose = verbose
        self.graph_mode = graph_mode
        self.device = device
        self.random_baseline = random_baseline
        self.arch = arch
        self.classification = classification
        if self.random_baseline:
            self.graph_mode = 'node'
            print("random baseline is on, i.e. features will be replaced with random numbers")

        if invariant:
            irrep_seq = [
                f"{n_s}x0e",
                f"{n_s}x0e",
                f"{n_s}x0e",
                f"{n_s}x0e"
            ]
        else:
            irrep_seq = [
                f"{n_s}x0e",
                f"{n_s}x0e + {n_v}x1o",
                f"{n_s}x0e + {n_v}x1o + {n_v}x1e",
                f"{n_s}x0e + {n_v}x1o + {n_v}x1e + {n_s}x0o"
            ]

        self.node_embedding = nn.Sequential(
            nn.Linear(node_fdim, n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
            nn.Linear(n_s, n_s)
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(distance_emb_dim, n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
            nn.Linear(n_s, n_s)
        )

        self.dist_expansion = GaussianSmearing(start=0.0, stop=max_radius, num_gaussians=distance_emb_dim, device=self.device)

        conv_layers = []
        for i in range(n_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]

            parameters = {
                "in_irreps": in_irreps,
                "sh_irreps": self.sh_irreps,
                "out_irreps": out_irreps,
                "edge_fdim": 3 * n_s,
                "h_dim": 3 * n_s,
                "residual": False,
                "dropout": dropout_p,
                "relu_in_fc": (self.arch!='no_relu_in_fc'),
            }

            layer = TensorProductConvLayer(**parameters)
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)

        self.score_predictor_edges = nn.Sequential(
            nn.Linear(2 * self.n_s_full + distance_emb_dim, self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, 1)
        )

        self.score_predictor_nodes = nn.Sequential(
            nn.Linear(self.n_s_full, 2 * self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * self.n_s, self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, 1)
        )

        self.score_predictor_nodes_half_with_relu = nn.Sequential(
            nn.Linear(self.n_s, 2 * self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * self.n_s, self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, 1)
        )

        self.score_predictor_nodes_half = nn.Sequential(
            nn.Linear(self.n_s, 2 * self.n_s, bias=False),
            nn.Dropout(dropout_p),
            nn.Linear(2 * self.n_s, self.n_s, bias=False),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, 1, bias=False)
        )

        self.score_predictor_nodes_with_edges = nn.Sequential(
            nn.Linear(self.n_s_full + distance_emb_dim, 2 * self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * self.n_s, self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, 1)
        )

        self.classif = nn.Sigmoid()


    def build_graph(self, data):

        radius_edges = radius_graph(data.pos, self.max_radius, data.batch)

        src, dst = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        edge_length_emb = self.dist_expansion(edge_vec.norm(dim=-1))

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return data.x.to(self.device), radius_edges.to(self.device), edge_length_emb.to(self.device), edge_sh.to(self.device)


    def forward_repr_mol(self, data):

        x, edge_index, edge_attr, edge_sh = self.build_graph(data)
        if self.verbose:
            print('dim of x', x.shape)
            print('dim of radius_graph (edges)', edge_index.shape)
            print('dim of edge length emb (gaussians)', edge_attr.shape)
            print('dim of edge sph harmonics', edge_sh.shape)

        x = self.node_embedding(x)
        edge_attr_emb = self.edge_embedding(edge_attr)
        if self.verbose:
            print('dim of x after node embedding', x.shape)
            print('dim of radius_graph (edges) after embedding', edge_attr_emb.shape)

        src, dst = edge_index

        def split_into_dict(x, irreps):
            x_dict = {}
            n0 = 0
            for j in str(irreps).split('+'):
                n, sym = j.split('x')
                l = int(sym[:-1])
                n = int(n) * (2*l+1)
                x_dict[j] = x[:,n0:n0+n]
                n0 += n
            return x_dict

        def update_dict(x_dict, x_update_dict):
            for key, val in x_update_dict.items():
                if key in x_dict:
                    x_dict[key] += val
                else:
                    x_dict[key] = val

        x_dict = {f'{self.n_s}x0e': x}
        for i in range(self.n_conv_layers):
            if i>0:
                x = torch.hstack([x_dict[j] for j in str(self.conv_layers[i-1].tp.irreps_out).split('+')])
            edge_attr_ = torch.cat([edge_attr_emb, x[dst, :self.n_s], x[src, :self.n_s]], dim=-1)
            x_update = self.conv_layers[i](x, edge_index, edge_attr_, edge_sh)
            x_update_dict = split_into_dict(x_update, self.conv_layers[i].tp.irreps_out)
            update_dict(x_dict, x_update_dict)

        x = torch.cat((x_dict[f'{self.n_s}x0e'], x_dict[f'{self.n_s}x0o']), dim=1) if f'{self.n_s}x0o' in x_dict else x_dict[f'{self.n_s}x0e']
        return x, edge_index, edge_attr


    def forward_energy_mode(self, data):

        if data.x.shape[0]==0:
            return torch.zeros((data.num_graphs, 1), device=self.device)

        x, (src, dst), edge_attr = self.forward_repr_mol(data)
        data.batch = data.batch.to(self.device)

        if self.random_baseline:
            x = torch.rand(x.shape)

        if self.sum_mode == 'both':
            score_inputs_nodes = x
            score_inputs_edges = torch.cat([edge_attr, x[src], x[dst]], dim=-1)

            edge_batch = data.batch[src]
            scores_nodes = self.score_predictor_nodes(score_inputs_nodes)
            scores_edges = self.score_predictor_edges(score_inputs_edges)

            score_node = scatter_add(scores_nodes, index=data.batch, dim=0)
            score_edge = scatter_add(scores_edges, index=edge_batch, dim=0)
            score_edge = F.pad(score_edge, (0, 0, 0, score_node.shape[0]-score_edge.shape[0]))
            score = score_node + score_edge
        elif self.sum_mode == 'node':
            score_inputs_nodes = x
            scores_nodes = self.score_predictor_nodes(score_inputs_nodes)
            score = scatter_add(scores_nodes, index=data.batch, dim=0)
        elif self.sum_mode == 'edge':
            score_inputs_edges = torch.cat([edge_attr, x[src], x[dst]], dim=-1)
            edge_batch = data.batch[src]
            scores_edges = self.score_predictor_edges(score_inputs_edges)
            score = scatter_add(scores_edges, index=edge_batch, dim=0)
        else:
            raise RuntimeError(f'sum mode {self.sum_mode} not defined')

        padsize = data.num_graphs-score.shape[0]
        if padsize>0:
            score = F.pad(score, (0, 0, 0, padsize))
        return score


    def forward_vector_mode(self, graph):
        x, (src, dst), edge_attr = self.forward_repr_mol(graph)
        if self.sum_mode == 'both':
            xedge = scatter_add(edge_attr, index=src, dim=0)
            xedge = F.pad(xedge, (0, 0, 0, x.shape[0]-xedge.shape[0]))
            x = torch.hstack((x, xedge))
        if self.graph_mode=='vector_masked':
            x *= graph.local_mask.to(self.device)[:,None]
        x = scatter_add(x, index=graph.batch.to(self.device), dim=0)
        if self.verbose:
            print('reaction x dims', x.shape)
        if self.sum_mode == 'node':

            if self.arch in ['normal', 'no_relu_in_fc']:
                score = self.score_predictor_nodes(x)
            if self.arch=='normal_scaled':
                x[:,self.n_s:]*=1e7
                score = self.score_predictor_nodes(x)
            elif self.arch=='both':
                score1 = self.score_predictor_nodes_half_with_relu(x[:,:self.n_s])
                score2 = self.score_predictor_nodes_half(x[:,self.n_s:]*1e7)
                score = score1 * score2
            elif self.arch=='both_nonscaled':
                score1 = self.score_predictor_nodes_half_with_relu(x[:,:self.n_s])
                score2 = self.score_predictor_nodes_half(x[:,self.n_s:])
                score = score1 * score2
            elif self.arch=='pseudo':
                score = self.score_predictor_nodes_half(x[:,self.n_s:]*1e7)

            if self.classification is True:
                score = self.classif(score) * 2 - 1

        elif self.sum_mode == 'both':
            score = self.score_predictor_nodes_with_edges(x)
        return score


    def forward(self, data, return_repr=False):
        if self.graph_mode in ['vector', 'vector_masked']:
            energy = self.forward_vector_mode(data)
            representations = None  # TODO
        elif self.graph_mode == 'energy':
            energy = self.forward_energy_mode(data)
            representations = None  # TODO

        return energy, representations
