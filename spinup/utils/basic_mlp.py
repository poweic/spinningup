import torch.nn as nn


class BasicMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, act=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.act = act(inplace=True)

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            y = fc(x)
            if x.shape == y.shape:
                x = x + y
            else:
                x = y
            if i < len(self.fcs) - 1:
                x = self.act(x)
        return x
