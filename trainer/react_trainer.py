from torch.optim import Adam
from trainer.trainer import Trainer


class ReactTrainer(Trainer):
    def __init__(self, **kwargs):
        super(ReactTrainer, self).__init__(optim=Adam, **kwargs)
        print(f"In trainer, metrics is {kwargs['metrics']} and std is {kwargs['std']}")

    def forward_pass(self, batch, return_repr=False):
        graphs, targets = tuple(batch)
        y_pred, representations = self.model(graphs, return_repr=return_repr)
        loss = self.loss_func(y_pred, targets)
        return loss, y_pred, targets, representations
