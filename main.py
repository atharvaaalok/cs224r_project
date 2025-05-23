import torch
from torch import nn
from NIGnets import NIGnet

from shape_assets.shapes import airfoil as target_shape
from compute_L_by_D import compute_L_by_D


# Import the NIGnet model that we trained to fit the airfoil
airfoil_file_name = 'NACA0012'
nig_net = NIGnet(layer_count = 4, act_fn = nn.Tanh)
nig_net.load_state_dict(torch.load(f'assets/nignet_fit_to_{airfoil_file_name}.pth', weights_only = True))


# Sample points on the curve
num_pts = 250
t = torch.linspace(0, 1, num_pts).reshape(-1, 1)
X = nig_net(t)
X = X.detach().cpu().numpy()



L_by_D = compute_L_by_D(X)
print(f'L_by_D_X: {L_by_D}')