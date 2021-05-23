import torch
import torch.nn as nn

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

class DQN_CNN(nn.Module):
    def __init__(self, num_classes=7,n_hidden=128,rows=7,cols=7):
        super(DQN_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_hidden, kernel_size=4, stride=1)
        self.relu1 = nn.ReLU()


        self.lin1 = nn.Linear(in_features=12 * n_hidden, out_features=n_hidden)
        self.lin2 = nn.Linear(in_features= n_hidden, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)

        output = torch.flatten(output)

        output = self.lin1(output)
        output = self.lin2(output)

        return output