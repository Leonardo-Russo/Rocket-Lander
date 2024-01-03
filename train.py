import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from rocket import Rocket

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    task = 'landing'  # 'hover' or 'landing'
    max_m_episode = 800000
    max_steps = 800

    # Create a vectorized environment
    env_fn = lambda: Rocket(task=task, max_steps=max_steps)
    env = make_vec_env(env_fn, n_envs=4, seed=1)

    # Load the model or create a new one
    model_path = os.path.join('models', task + '_ppo')
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env)  # Set the environment here
    else:
        model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=1000000)  # Adjust the number of timesteps as needed

    # Save the model
    model_path = os.path.join('models', task + '_ppo')
    model.save(model_path)
