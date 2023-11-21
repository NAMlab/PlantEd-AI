import os
import sys
import math

import gymnasium as gym

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import torch as th

from planted_env import PlantEdEnv

import optuna

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

def make_env(episode_name, reset_callback):
  """
  Utility function for multiprocessed env.

  :param env_id: the environment ID
  :param num_env: the number of environments you wish to have in subprocesses
  :param seed: the inital seed for RNG
  :param rank: index of the subprocess
  """
  def _init():
    env = PlantEdEnv(episode_name, port)
    env.reset()
    env.reset_callback = reset_callback
    return env
  return _init

study = optuna.create_study(study_name="Single Environment Study", storage="sqlite:///optuna-studies.db", load_if_exists=True, direction="maximize")

def objective(trial):

    episode_rewards = []
    episode_number = 1

    # This is called before the env is reset so that we have access to the total rewards
    # at the end of each episode.
    def reset_callback(env):
      nonlocal episode_rewards
      nonlocal episode_number
      episode_rewards.append(env.total_rewards)
      trial.report(env.total_rewards, episode_number)
      episode_number += 1
      print(f"Episode {episode_number} finished with total reward of {env.total_rewards}")

    envs = DummyVecEnv([make_env(f"Trial_{trial.number}", reset_callback)])
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
    batch_size = trial.suggest_int("batch_size", 2, 96)
    n_steps = 2 * batch_size

    layer_1_size = trial.suggest_int("layer_1_size", 4, 32)
    layer_2_size = trial.suggest_int("layer_2_size", 4, 32)

    learning_rate = trial.suggest_float("learning_rate", 5e-6, 0.003)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1)
    gamma = trial.suggest_float("gamma", 0.8, 0.9997)

    activation_fns = dict(
      Tanh = th.nn.Tanh,
      ReLU = th.nn.ReLU,
      LeakyReLU = th.nn.LeakyReLU,
      SELU = th.nn.SELU
    )

    activation_fn = activation_fns[trial.suggest_categorical("activation_fn", list(activation_fns.keys()))]

    model = PPO(policy = "MultiInputPolicy", env = envs, verbose=2, n_steps = n_steps, batch_size = batch_size, learning_rate = learning_rate,
      clip_range = clip_range, ent_coef = ent_coef, gamma = gamma,
      policy_kwargs = dict(
        activation_fn=activation_fn,
        net_arch=dict(
          pi=[layer_1_size, layer_2_size],
          vf=[layer_1_size, layer_2_size])
      ))
    model.learn(18060, log_interval=10)
    print(episode_rewards)
    envs.close()
    return sum(episode_rewards) / len(episode_rewards)

study.optimize(objective, n_trials=50)
print(f"Best value: {study.best_value} (params: {study.best_params})")
