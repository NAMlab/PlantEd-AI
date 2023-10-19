import os

import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from planted_env import PlantEdEnv

env = PlantEdEnv("Environment 1")

s_size = env.observation_space.shape
a_size = env.action_space

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action

def make_env():
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = PlantEdEnv("ep1")
        env.reset()
        return env
    return _init

envs = DummyVecEnv([make_env()])

model = A2C(policy = "MultiInputPolicy", env = envs, verbose=1)

model.learn(5000, log_interval=10)
envs.close()

print(model)

