import torch

class DQN(torch.nn.Module):
    def __init__(self, n_hidden=64, rows=6, cols=7):
        super(DQN, self).__init__()
        self.dense1 = torch.nn.Linear(rows*cols*2, n_hidden)
        self.dense2 = torch.nn.Linear(n_hidden, cols)

    def forward(self, x):
    # assumes input x already formatted as a torch array
        h = self.dense1(x).relu()
        out = self.dense2(h)
        return out