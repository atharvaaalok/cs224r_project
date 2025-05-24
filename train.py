import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from NIGnets import NIGnet
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.logger import configure

from NIGnetShapeEnv import NIGnetShapeEnv
from compute_L_by_D import compute_L_by_D
from shape_assets.utils import plot_curves



if __name__ == '__main__':

    # Import the NIGnet model that we trained to fit the airfoil
    airfoil_file_name = 'NACA0012'
    nig_net = NIGnet(layer_count = 2, act_fn = nn.Tanh)
    nig_net.load_state_dict(torch.load(f'assets/nignet_fit_to_{airfoil_file_name}.pth', weights_only = True))


    # Create an environment
    action_sigma = 0.01
    max_episode_steps = 5
    non_convergence_reward = -5
    def make_env():
        env = NIGnetShapeEnv(nig_net = nig_net, action_sigma = action_sigma,
                        max_episode_steps = max_episode_steps,
                        non_convergence_reward = non_convergence_reward)
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
    model = PPO(policy = 'MlpPolicy', env = env, verbose = 1,
                tensorboard_log = tensorboard_log_file_location,
                n_steps = print_stats_every, batch_size = 64)


    model.set_logger(new_logger)

    # Start training
    model.learn(total_timesteps = 100000, progress_bar = True)


    # Policy evaluation
    def make_test_env():
        env = NIGnetShapeEnv(nig_net = nig_net, action_sigma = action_sigma,
                        max_episode_steps = max_episode_steps + 1,
                        non_convergence_reward = non_convergence_reward)
        return env
    num_env = 16
    test_env = SubprocVecEnv([make_test_env for _ in range(num_env)])
    test_env = VecMonitor(test_env)
    observation = test_env.reset()
    for _ in range(max_episode_steps):
        action, _states = model.predict(observation, deterministic = True)
        observation, reward, done, info = test_env.step(action)
    
    observation = observation[0]
    
    # Convert observation to network parameters
    test_nig_net = NIGnet(layer_count = 2, act_fn = nn.Tanh)
    vector_to_parameters(torch.from_numpy(observation), test_nig_net.parameters())

    # Calculate L by D and plot the airfoil produced
    num_pts = 250
    t = torch.linspace(0, 1, num_pts).reshape(-1, 1)
    X = test_nig_net(t)
    L_by_D = compute_L_by_D(X.detach().cpu().numpy())

    X_original = nig_net(t)
    fig_filename = 'generated_airfoil.png'
    plot_curves(X_original, X, filename = fig_filename)

    print(f'\n\nFinal L_by_D of trained policy: {L_by_D:7.3f}')
    print(f'Last reward: {reward[0]}')