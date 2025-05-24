# Basic imports
import numpy as np
import torch
from torch import nn
import geosimilarity as gs
from NIGnets import NIGnet


from shape_assets.utils import automate_training, plot_curves


# Get points on target airfoil curve
airfoil_file_name = 'NACA0012'
Xt = np.loadtxt(f'assets/{airfoil_file_name}.dat')
Xt = torch.from_numpy(Xt).to(torch.float32)

# Discretize the input interval t = [0, 1] that will map to the shape
num_pts = Xt.shape[0]
t = torch.linspace(0, 1, num_pts).reshape(-1, 1)


# Create a NIGnet object
nig_net = NIGnet(layer_count = 2, act_fn = nn.Tanh)


# Fit the NIGnet to the target airfoil using the provided automate training function
automate_training(
    model = nig_net, loss_fn = gs.MSELoss(), X_train = t, Y_train = Xt,
    learning_rate = 0.1, epochs = 100000, print_cost_every = 2000
)


# Visualize the fit
Xc = nig_net(t)
plot_curves(Xc, Xt)


# Save the model
torch.save(nig_net.state_dict(), f'assets/nignet_fit_to_{airfoil_file_name}.pth')