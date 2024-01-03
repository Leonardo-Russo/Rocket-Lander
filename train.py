import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
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

    # Initialize the PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=100)  # Adjust the number of timesteps as needed

    # Save the model
    model_path = os.path.join('models', task + '_ppo')
    model.save(model_path)

    # Load the model (if needed)
    # model = PPO.load("ppo_rocket")

    # Evaluate the model
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        # env.render(render_mode='human')
        env.envs[0].render()

