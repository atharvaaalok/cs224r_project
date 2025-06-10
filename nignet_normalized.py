import torch
import torch.nn as nn
import torch.nn.functional as F


class NIGnetNorm(nn.Module):
    def __init__(self, layer_count, act_fn, skip_connections = True):
        super(NIGnetNorm, self).__init__()
        
        self.layer_count = layer_count
        self.skip_connections = skip_connections

        self.closed_transform = lambda t: torch.hstack([
            torch.cos(2 * torch.pi * t),
            torch.sin(2 * torch.pi * t)
        ])

        Linear_class = nn.Linear

        self.linear_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()

        self.alphas = nn.ParameterList()

        for i in range(layer_count):
            self.linear_layers.append(Linear_class(2, 2))
            self.act_layers.append(act_fn())

            self.alphas.append(nn.Parameter(torch.tensor(1.0)))
        
        self.final_linear = Linear_class(2, 2)
    

    def forward(self, T):
        t = T
        X = self.closed_transform(t)

        for i, (linear_layer, act_layer) in enumerate(zip(self.linear_layers, self.act_layers)):
            X = linear_layer(X)

            if self.skip_connections:
                residual = X
            
            X = act_layer(X)

            if self.skip_connections:
                alpha_sq = self.alphas[i] ** 2
                X = (X + alpha_sq * residual) / 2.0
        
        X = self.final_linear(X)

        
        # Center and scale
        centroid = torch.mean(X, axis = 0)
        X = X - centroid
        max_abs = torch.max(torch.abs(X))
        X = X / max_abs

        return X