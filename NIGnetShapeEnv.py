import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np
from NIGnets import NIGnet
import gymnasium as gym
from gymnasium import spaces

from compute_L_by_D import compute_L_by_D



class NIGnetShapeEnv(gym.Env):

    metadata = {'render_modes': []}

    def __init__(self, nig_net, action_sigma, max_episode_steps):
        super().__init__()

        self.nig_net = nig_net
        self.action_sigma = action_sigma
        self.max_episode_steps = max_episode_steps

        # Flatten parameter vector once to fix observation and action dimensions
        with torch.no_grad():
            self._param_template = parameters_to_vector(self.nig_net.parameters()).detach().clone()
        self.num_params = self._param_template.numel()

        # Define observation space
        low = -np.inf * np.ones(self.num_params, dtype = np.float32)
        high = np.inf * np.ones(self.num_params, dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # Define action space to be between +/-1, they will be scaled by action_sigma during step()
        self.action_space = spaces.Box(low = -1.0, high = 1.0,
                                       shape = (self.num_params,), dtype = np.float32)
        
        # Internal state
        self._step_count = 0
        self.state = self._param_template.cpu().numpy().astype(np.float32)
    

    def _get_obs(self):
        return self.state.copy()


    def _get_info(self, reward = None):
        return {'steps': self._step_count, 'reward': reward}
    

    def reset(self, seed = None):
        super().reset(seed = seed)

        # Reinitialize nig_net to the network parameters that represent starting airfoil
        vector_to_parameters(self._param_template, self.nig_net.parameters())

        self.state = self._param_template.cpu().numpy().copy().astype(np.float32)
        self._step_count = 0

        observation = self._get_obs()
        info = self._get_info(reward = None)
        return observation, info


    def step(self, action):
        self._step_count += 1

        action = np.clip(action, -1, 1).astype(np.float32)
        perturb = torch.from_numpy(action) * self.action_sigma
        new_params = torch.from_numpy(self.state) + perturb
        # Set nig_net parameters to the new parameter state
        vector_to_parameters(new_params, self.nig_net.parameters())

        self.state = new_params.cpu().numpy().astype(np.float32)

        # Calculate reward
        num_pts = 250
        t = torch.linspace(0, 1, num_pts).reshape(-1, 1)
        X = self.nig_net(t).detach().cpu().numpy()
        reward = compute_L_by_D(X)

        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        observation = self._get_obs()
        info = self._get_info(reward)

        return observation, reward, terminated, truncated, info
    

    def render(self):
        pass


    def close(self):
        pass