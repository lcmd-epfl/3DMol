import torch
from torch_geometric.data import Batch


class CustomCollator:
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, batch):
        data = list(map(list, zip(*batch, strict=True)))  # on cpu
        targets, idx, graphs, extra = data
        targets = torch.tensor(targets).float().reshape(-1, 1).to(self.device)
        graphs = Batch.from_data_list(graphs).to(self.device)
        if extra[0] is not None:
            extra = {key: torch.tensor([d[key] for d in extra], device=self.device) for key in extra[0]}
        else:
            extra = None
        return graphs, extra, targets, idx  # on device except for idx
