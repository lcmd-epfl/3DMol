import math
import torch
from torch import nn
from e3nn import o3
from torch_scatter import scatter, scatter_add

from .graph import BuildGraph


class TensorProductConvLayer(nn.Module):

    def __init__(self, in_irreps, sh_irreps, out_irreps, edge_fdim, dropout=0.0,
                 h_dim=None, relu_in_fc=True, internal_weights=False, arch='normal'):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.internal_weights = internal_weights
        if h_dim is None:
            h_dim = edge_fdim

        if self.internal_weights is False:  # normal option
            self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)
        else:
            self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, internal_weights=True, shared_weights=True)

            if arch in {'normal_weights100', 'both_weights100', 'pseudo_weights100'}:
                if '0o' in self.tp.irreps_out:
                    out_irreps_list = str(self.tp.irreps_out).split('+')
                    out_irreps_pseudoscalar = [*map(lambda x: x.endswith('0o'), out_irreps_list)]
                    assert sum(out_irreps_pseudoscalar)==1
                    idx_irrep_pseudoscalar = out_irreps_pseudoscalar.index(True)

                    weights_pointer = 0
                    for i, instr in enumerate(tp.instructions):
                        weights_pointer += math.prod(instr.path_shape)
                        if i==idx_irrep_pseudoscalar:
                            weights_pointer_end = weights_pointer + math.prod(instr.path_shape)
                            break
                    with torch.no_grad():
                        self.tp.weight[weights_pointer:weights_pointer_end] *= 100.0

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

        if self.internal_weights is False:
            tp_out = self.tp(x[edge_src], edge_sh, self.fc_net(edge_attr))
        else:
            tp_out = self.tp(x[edge_src], edge_sh)

        out_nodes = out_nodes or x.shape[0]
        out = scatter(src=tp_out, index=edge_dst, dim=0, dim_size=out_nodes, reduce=aggr)
        return out


class EquiReact(nn.Module):

    def __init__(self, node_fdim: int, edge_fdim: int, sh_lmax: int = 2,
                 n_s: int = 16, n_v: int = 16, n_conv_layers: int = 2,
                 max_radius: float = 10.0,
                 distance_emb_dim: int = 32, dropout_p: float = 0.1,
                 verbose=False, device='cpu', graph_mode='vector',
                 invariant=False,
                 classification=False, arch='normal',
                 internal_weights=False,
                 **kwargs):

        super().__init__(**kwargs)

        if invariant:
            sh_lmax = 0

        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.n_s, self.n_v = n_s, n_v
        self.n_conv_layers = n_conv_layers
        self.n_s_full = 2 * self.n_s if (not invariant and self.n_conv_layers >= 3) else self.n_s
        self.distance_emb_dim = distance_emb_dim
        self.max_radius = max_radius
        self.verbose = verbose
        self.graph_mode = graph_mode
        self.device = device
        self.arch = arch
        self.classification = classification

        if invariant:
            irrep_seq = [
                f"{n_s}x0e",
                f"{n_s}x0e",
                f"{n_s}x0e",
                f"{n_s}x0e"
            ]
            irrep_last = f"{n_s}x0e"
        else:
            irrep_seq = [
                f"{n_s}x0e",
                f"{n_s}x0e + {n_v}x1o",
                f"{n_s}x0e + {n_v}x1o + {n_v}x1e",
                f"{n_s}x0e + {n_v}x1o + {n_v}x1e + {n_s}x0o"
            ]
            if self.n_conv_layers >=3:
                irrep_last = f"{n_s}x0e + {n_s}x0o"
            else:
                irrep_last = f"{n_s}x0e"

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

        conv_layers = []
        for i in range(n_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            if i<n_conv_layers-1:
                out_irreps = irrep_seq[min(i+1, len(irrep_seq)-1)]
            else:
                out_irreps = irrep_last

            parameters = {
                "in_irreps": in_irreps,
                "sh_irreps": self.sh_irreps,
                "out_irreps": out_irreps,
                "edge_fdim": 3 * n_s,
                "h_dim": 3 * n_s,
                "dropout": dropout_p,
                "relu_in_fc": (self.arch!='no_relu_in_fc'),
                "internal_weights": internal_weights,
                "arch": self.arch,
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

        self.build_graph = BuildGraph(sh_irreps=self.sh_irreps, max_radius=self.max_radius, distance_emb_dim=self.distance_emb_dim, device=self.device)

    def forward_repr_mol(self, data, _extra):

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

        def update_dict(x_dict, x_update, irreps):
            n0 = 0
            keys = str(irreps).split('+')
            # delete keys not in irreps (anything not going to updated)
            for key in set(x_dict.keys())-set(keys):
                del x_dict[key]
            # update
            for key in keys:
                n, sym = key.split('x')
                l = int(sym[:-1])
                n = int(n) * (2*l+1)
                update = x_update[:,n0:n0+n]
                if key in x_dict:
                    x_dict[key] += update
                else:
                    x_dict[key] = update
                n0 += n

        # TODO precompute all the strig operations ???
        scalar_key = f'{self.n_s}x0e'
        pseudoscalar_key = f'{self.n_s}x0o'
        x_dict = {scalar_key: x}
        for i in range(self.n_conv_layers):
            edge_attr_ = torch.cat([edge_attr_emb, x_dict[scalar_key][dst], x_dict[scalar_key][src]], dim=-1)
            if i>0:
                x = torch.hstack([x_dict[j] for j in str(self.conv_layers[i-1].tp.irreps_out).split('+')])
            x_update = self.conv_layers[i](x, edge_index, edge_attr_, edge_sh)
            update_dict(x_dict, x_update, self.conv_layers[i].tp.irreps_out)

        x = torch.cat((x_dict[scalar_key], x_dict[pseudoscalar_key]), dim=1) if pseudoscalar_key in x_dict else x_dict[scalar_key]
        return x, edge_index, edge_attr

    def forward_mol(self, graph, extra):
        x, _, _ = self.forward_repr_mol(graph, extra)
        if self.graph_mode=='vector_masked':
            x *= graph.local_mask[:,None]
        x = scatter_add(x, index=graph.batch.to(self.device), dim=0)
        if self.verbose:
            print('reaction x dims', x.shape)

        if self.arch in {'normal', 'no_relu_in_fc', 'normal_weights100'}:
            score = self.score_predictor_nodes(x)
        elif self.arch=='normal_scaled':
            x[:,self.n_s:]*=1e7
            score = self.score_predictor_nodes(x)
        elif self.arch=='both':
            score1 = self.score_predictor_nodes_half_with_relu(x[:,:self.n_s])
            score2 = self.score_predictor_nodes_half(x[:,self.n_s:]*1e7)
            score = score1 * score2
        elif self.arch in {'both_nonscaled', 'both_weights100'}:
            score1 = self.score_predictor_nodes_half_with_relu(x[:,:self.n_s])
            score2 = self.score_predictor_nodes_half(x[:,self.n_s:])
            score = score1 * score2
        elif self.arch=='pseudo':
            score = self.score_predictor_nodes_half(x[:,self.n_s:]*1e7)
        elif self.arch in {'pseudo_nonscaled', 'pseudo_weights100'}:
            score = self.score_predictor_nodes_half(x[:,self.n_s:])

        if self.classification is True:
            score = self.classif(score) * 2 - 1

        return score

    def forward(self, graph, extra, return_repr=False):
        predictions = self.forward_mol(graph, extra)
        if return_repr:
            representations = 0  # TODO
        else:
            representations = None  # TODO
        return predictions, representations
