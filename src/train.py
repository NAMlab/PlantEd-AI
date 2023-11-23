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
        env = PlantEdEnv(f"Level4_try1_{level_name}", port, level_name)
        env.reset()
        return env
    return _init

if __name__ == "__main__":
  envs = SubprocVecEnv([
    make_env('spring_high_nitrate', 8771),
    make_env('spring_low_nitrate', 8772),
    make_env('summer_high_nitrate', 8773),
    make_env('summer_low_nitrate', 8774),
    make_env('fall_high_nitrate', 8775),
    make_env('fall_low_nitrate', 8776)
  ], start_method='fork')
  #envs = VecNormalize(envs, norm_obs_keys = [
  #  "temperature",
  #  "sun_intensity",
  #  "humidity",
  #  "accessible_water",
  #  "accessible_nitrate",
  #  "green_thumbs",
  #  # Plant
  #  "biomasses", #leaf, stem, root, seed
  #  "n_organs", #leaf, stem, root, seed
  #  "open_spots",
  #  "starch_pool",
  #  "max_starch_pool"])
  envs = VecNormalize.load("models/Level3_try1_vec_normalize.pkl", envs)

  #model = PPO(policy = "MultiInputPolicy", env = envs, verbose=2, n_steps = 96, batch_size = 48, ent_coef = 0.05, device="cpu", policy_kwargs = dict(
  #    activation_fn=th.nn.Tanh,
  #    net_arch=dict(
  #      pi=[32,16],
  #      vf=[32,16])
  #  ))
  model = PPO.load("models/Level3_try1", env=envs)

  model.learn(50*600*6, log_interval=10)
  envs.close()

  print(model)
  model.save("models/Level4_try1")
  envs.save("models/Level4_try1_vec_normalize.pkl")

