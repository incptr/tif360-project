import torch

class DQN(torch.nn.Module):
    def __init__(self, n_hidden=128, rows=6, cols=7):
        super(DQN,self).__init__()
        self.dense1 = torch.nn.Linear(rows*cols*2, n_hidden)
        self.dense2 = torch.nn.Linear(n_hidden, n_hidden//2)
        self.dense3 = torch.nn.Linear(n_hidden//2, cols)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
    # assumes input x already formatted as a torch array
        x = self.dense1(x).relu()
        x = self.dropout(x)
        x = self.dense2(x).relu()
        out = self.dense3(x)
        return out