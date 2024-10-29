# Try to find the best possible high score for the level and setup the children are playing in the school.
# Allow overfitting here by only playing the one level.

import os

import gymnasium as gym

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import torch as th

from planted_env import PlantEdEnv
from custom_subproc_vec_env import SubprocVecEnv

env = PlantEdEnv("Environment 1")

s_size = env.observation_space.shape
a_size = env.action_space

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action

def make_env(level_name, port):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = PlantEdEnv(f"human-contd-{level_name}", port, level_name)
        env.reset()
        return env
    return _init

if __name__ == "__main__":
  envs = SubprocVecEnv([
    make_env('summer_low_nitrate', 8764),
  ], start_method='fork')
  envs = VecNormalize(envs, norm_obs_keys = [
    "temperature",
    "sun_intensity",
    "humidity",
    "accessible_water",
    "accessible_nitrate",
    "green_thumbs",
    # Plant
    "biomasses", #leaf, stem, root, seed
    "n_organs", #leaf, stem, root, seed
    "open_spots",
    "starch_pool",
    "max_starch_pool"])
  #envs = VecNormalize.load("models/human.pkl", envs)

  model = PPO(policy = "MultiInputPolicy", env = envs, verbose=2, n_steps = 96, batch_size = 48, ent_coef = 0.05, device="cpu", learning_rate=0.0001, policy_kwargs = dict(
    activation_fn=th.nn.Tanh,
    net_arch=dict(
      pi=[64,32,32],
      vf=[64,32,32]),
    optimizer_kwargs = dict(
      betas = (0.99, 0.99))
  ))
  #model = PPO.load("models/human", env=envs)

  model.learn(600*600*1, log_interval=10)
  envs.close()

  model.save("models/human")
  envs.save("models/human.pkl")

