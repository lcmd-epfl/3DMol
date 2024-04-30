import torch
from torch_geometric.data import Batch


class CustomCollator(object):
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, batch):
        data = list(map(list, zip(*batch)))
        targets, idx, graphs = data
        targets = torch.tensor(targets).float().reshape(-1, 1).to(self.device)
        graphs = Batch.from_data_list(graphs)
        return graphs, targets, idx
