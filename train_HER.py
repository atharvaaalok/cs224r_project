import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from NIGnets import NIGnet
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt
import numpy as np

from NIGnetShapeEnvGoal import NIGnetShapeEnvGoal
from compute_L_by_D import compute_L_by_D
# from shape_assets.utils import plot_curves


def plot_curves_normalized(Xc: torch.Tensor, Xt: torch.Tensor, filename = None) -> None:
    # Center and scale
    centroid = torch.mean(Xt, axis = 0)
    Xt = Xt - centroid
    max_abs = torch.max(torch.abs(Xt))
    Xt = Xt / max_abs

    # Get torch tensor to cpu and disable gradient tracking to plot using matplotlib
    Xc = Xc.detach().cpu()
    Xt = Xt.detach().cpu()
    
    plt.fill(Xt[:, 0], Xt[:, 1], color = "#C9C9F5", alpha = 0.46, label = "Optimized Shape")
    plt.fill(Xc[:, 0], Xc[:, 1], color = "#F69E5E", alpha = 0.36, label = "Initial Shape")

    plt.plot(Xt[:, 0], Xt[:, 1], color = "#000000", linewidth = 2)
    plt.plot(Xc[:, 0], Xc[:, 1], color = "#000000", linewidth = 2, linestyle = "--")

    plt.axis('equal')
    plt.tight_layout()
    plt.legend()

    plt.savefig(filename + '_HER_news' + '.svg')

    if filename is not None:
        plt.savefig(filename + '.png', dpi = 600)
    plt.show()




if __name__ == '__main__':
    goal_sampler = lambda: np.random.uniform(10.0, 50.0)

    # Import the NIGnet model that we trained to fit the airfoil
    airfoil_file_name = 'NACA0012'
    nig_net = NIGnet(layer_count = 2, act_fn = nn.Tanh)
    nig_net.load_state_dict(torch.load(f'assets/nignet_fit_to_normalized_{airfoil_file_name}.pth', weights_only = True))


    # Create an environment
    action_sigma = 0.01
    max_episode_steps = 5
    non_convergence_reward = -50
    def make_env():
        env = NIGnetShapeEnvGoal(nig_net = nig_net, action_sigma = action_sigma,
                        max_episode_steps = max_episode_steps,
                        non_convergence_reward = non_convergence_reward,
                        goal_sampler = goal_sampler)
        return env

    # Use vectorized environments
    num_env = 16
    env = SubprocVecEnv([make_env for _ in range(num_env)])
    env = VecMonitor(env)


    # Create RL model
    tensorboard_log_file_location = 'training_logs'
    # Set up logger
    new_logger = configure(tensorboard_log_file_location, ['stdout', 'tensorboard'])

    print_stats_every = 100 // num_env
    model = SAC(policy = 'MultiInputPolicy', env = env,
                replay_buffer_class = HerReplayBuffer,
                replay_buffer_kwargs = dict(
                    n_sampled_goal = 4,
                    goal_selection_strategy = 'future',
                ),
                verbose = 1,
                tensorboard_log = tensorboard_log_file_location,
                learning_starts = max_episode_steps * num_env * 2,
                train_freq = (1, 'step'), batch_size = 512)


    model.set_logger(new_logger)

    # Start training
    total_timesteps = 1_00_000
    model.learn(total_timesteps = total_timesteps, progress_bar = True)


    # Policy evaluation
    test_goal_sampler = lambda: np.array([30.0])

    def make_test_env():
        env = NIGnetShapeEnvGoal(nig_net = nig_net, action_sigma = action_sigma,
                        max_episode_steps = max_episode_steps,
                        non_convergence_reward = non_convergence_reward,
                        goal_sampler = test_goal_sampler)
        return env
    num_env = 16
    test_env = SubprocVecEnv([make_test_env for _ in range(num_env)])
    test_env = VecMonitor(test_env)
    observation = test_env.reset()
    for _ in range(max_episode_steps - 1):
        action, _states = model.predict(observation, deterministic = True)
        observation, reward, done, info = test_env.step(action)
        print(f'reward: {reward[0]}')
    
    obs = observation['observation'][0]
    achieved_goal = observation['achieved_goal'][0]
    desired_goal = observation['desired_goal'][0]

    print(obs)
    print(f'achieved_goal: {achieved_goal}')
    print(f'desired_goal: {desired_goal}')
    
    # Convert observation to network parameters
    test_nig_net = NIGnet(layer_count = 2, act_fn = nn.Tanh)
    vector_to_parameters(torch.from_numpy(obs), test_nig_net.parameters())

    # Calculate L by D and plot the airfoil produced
    num_pts = 250
    t = torch.linspace(0, 1, num_pts).reshape(-1, 1)
    X = test_nig_net(t)
    L_by_D = compute_L_by_D(X.detach().cpu().numpy())

    X_original = nig_net(t)
    fig_filename = f'figures/generated_airfoil_total_timesteps_{total_timesteps}_max_episode_len_{max_episode_steps}_non_convergence_reward{-non_convergence_reward}'
    plot_curves_normalized(X_original, X, filename = fig_filename)

    print(f'\n\nFinal L_by_D of trained policy: {L_by_D:7.3f}')
    print(f'Last reward: {reward[0]}')