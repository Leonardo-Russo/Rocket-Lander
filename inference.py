import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from rocket import Rocket

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    task = 'landing'
    max_m_episode = 800000
    max_steps = 800

    # Create a vectorized environment
    env_fn = lambda: Rocket(task=task, max_steps=max_steps)
    env = make_vec_env(env_fn, n_envs=4, seed=1)

    # Load the model
    model_path = os.path.join('Models', task + '_ppo')
    model = PPO.load(model_path)

    # Evaluate the model
    max_eval_steps = 10000
    obs = env.reset()
    ep_counter = 0
    ep_eval = 5
    for step_id in range(max_eval_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        # Render the first environment (assuming single environment for visualization)
        env.envs[0].render()

        # Stop if any of the environments is done
        if dones[0]:
            ep_counter += 1
        
        if ep_counter >= ep_eval:
            break
