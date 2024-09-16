import os
import sys
import math
import statistics

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

def make_env(level_name, port, episode_name, reset_callback):
  def _init():
    env = PlantEdEnv(episode_name + "_" + level_name, port, level_name)
    env.reset()
    env.reset_callback = reset_callback
    return env
  return _init

study = optuna.create_study(study_name="Paperv1 Study", storage="sqlite:///level-studies.db", load_if_exists=True, direction="maximize")

def objective(trial):

    scenario_rewards = []
    episode_rewards = []
    episode_number = 1

    # This is called before the env is reset so that we have access to the total rewards
    # at the end of each episode.
    def reset_callback(env):
      nonlocal episode_rewards
      nonlocal scenario_rewards
      nonlocal episode_number
      scenario_rewards.append(env.total_rewards)
      print(f"Appending Scenario reward {env.total_rewards} for scenario {env.level_name}")
      if len(scenario_rewards) > 5: #this was the last scenario
        print("All Scenarios, finished!")
        episode_mean_reward = sum(scenario_rewards) / len(scenario_rewards)
        scenario_rewards = []
        episode_rewards.append(episode_mean_reward)
        trial.report(episode_mean_reward, episode_number)
        episode_number += 1
        print(f"Episode {episode_number} finished with total reward of {episode_mean_reward}")

    # Having multiple in parallel could be messy because I wouldn't be sure all their scores
    # can return to this process in a clean way to be reported to Optuna. So instead we're doing
    # one environment at a time but simply start more trials in parallel.
    envs = DummyVecEnv([
      make_env('spring_high_nitrate', port, f"Trial_{trial.number}", reset_callback),
      make_env('spring_low_nitrate', port+1, f"Trial_{trial.number}", reset_callback),
      make_env('summer_high_nitrate', port+2, f"Trial_{trial.number}", reset_callback),
      make_env('summer_low_nitrate', port+3, f"Trial_{trial.number}", reset_callback),
      make_env('fall_high_nitrate', port+4, f"Trial_{trial.number}", reset_callback),
      make_env('fall_low_nitrate', port+5, f"Trial_{trial.number}", reset_callback)
    ])
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
    betas = trial.suggest_float("betas", .99, .999)

    activation_fns = dict(
      Tanh = th.nn.Tanh,
      ReLU = th.nn.ReLU,
      LeakyReLU = th.nn.LeakyReLU,
      SELU = th.nn.SELU
    )

    activation_fn = activation_fns[trial.suggest_categorical("activation_fn", list(activation_fns.keys()))]

    model = PPO(policy = "MultiInputPolicy", env = envs, verbose=2, n_steps = n_steps, batch_size = batch_size, learning_rate = learning_rate,
      clip_range = clip_range, ent_coef = ent_coef, gamma = gamma, device="cpu",
      policy_kwargs = dict(
        activation_fn=activation_fn,
        net_arch=dict(
          pi=[layer_1_size, layer_2_size],
          vf=[layer_1_size, layer_2_size]),
        optimizer_kwargs = dict(
          betas = (betas, betas))
      ))
    model.learn(200*600*6, log_interval=10)
    print(episode_rewards)
    envs.close()
    return statistics.mean(episode_rewards)

# Test our current parameters (only in first instance of this script though!)
# study.enqueue_trial(params={
#   "batch_size": 48,
#   "layer_1_size": 32,
#   "layer_2_size": 16,
#   "learning_rate": 0.0001,
#   "clip_range": 0.2,
#   "ent_coef": 0.05,
#   "gamma": 0.99,
#   "activation_fn": "Tanh",
#   "betas": 0.99
# })
study.optimize(objective, n_trials=50, gc_after_trial=True)
print(f"Best value: {study.best_value} (params: {study.best_params})")
