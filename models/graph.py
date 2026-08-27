import numpy as np
import torch
import torch.nn as nn
from e3nn import o3
from torch_cluster import radius_graph


class GaussianSmearing(nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, device='cpu'):
        super().__init__()
        self.device = device
        mu = torch.linspace(start, stop, num_gaussians).to(self.device)
        self.coeff = -0.5 / (mu[1] - mu[0]).item() ** 2
        self.register_buffer('mu', mu)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.mu.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))



class BuildGraph(nn.Module):

    def __init__(self, sh_irreps, max_radius=10.0, distance_emb_dim=32, device='cpu'):
        super().__init__()
        self.device = device
        self.max_radius = max_radius
        self.distance_emb_dim = distance_emb_dim
        self.sh_irreps = sh_irreps
        self.dist_expansion = GaussianSmearing(start=0.0, stop=max_radius, num_gaussians=distance_emb_dim, device=self.device)

    def forward(self, data):
        radius_edges = radius_graph(data.pos, self.max_radius, data.batch)
        src, dst = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        edge_length_emb = self.dist_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return data.x, radius_edges, edge_length_emb, edge_sh



