import os
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

def make_env():
  """
  Utility function for multiprocessed env.

  :param env_id: the environment ID
  :param num_env: the number of environments you wish to have in subprocesses
  :param seed: the inital seed for RNG
  :param rank: index of the subprocess
  """
  def _init():
    env = PlantEdEnv("World1")
    env.reset()
    return env
  return _init

# Taken and adapted from https://stackoverflow.com/questions/77043606/stop-stable-baselines-learn-method-when-execution-is-terminated
class StopOnSuccessCallback(BaseCallback):
    def __init__(self, episode_rewards, trial, verbose=0):
        self.episode_rewards = episode_rewards
        self.latest_reward_state = 0
        self.episode_number = 1
        self.trial = trial
        super(StopOnSuccessCallback, self).__init__(verbose)

    def _on_step(self):
        env =  self.model.env.envs[0]
        if env.total_rewards == 0: # this indicates that a new environment has been started
            self.episode_rewards.append(self.latest_reward_state) # the total rewards from the last step (this misses the last step but fine)
            self.trial.report(self.latest_reward_state, self.episode_number)
            self.episode_number += 1
        else: # update the total rewards so we can save it when the environment closes
            self.latest_reward_state = env.total_rewards
        return True


study = optuna.create_study(study_name="Single Environment Study", storage="sqlite:///optuna-studies.db", load_if_exists=True, direction="maximize")

def objective(trial):

    envs = DummyVecEnv([make_env()])
    batch_size = trial.suggest_int("batch_size", 2, 96)
    n_steps = 2 * batch_size

    layer_1_size = trial.suggest_int("layer_1_size", 4, 32)
    layer_2_size = trial.suggest_int("layer_2_size", 4, 32)

    learning_rate = trial.suggest_float("learning_rate", 5e-6, 0.003)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    activation_fns = dict(
      Tanh = th.nn.Tanh,
      ReLU = th.nn.ReLU,
      LeakyReLU = th.nn.LeakyReLU,
      SELU = th.nn.SELU
    )

    activation_fn = activation_fns[trial.suggest_categorical("activation_fn", list(activation_fns.keys()))]

    model = PPO(policy = "MultiInputPolicy", env = envs, verbose=2, n_steps = n_steps, batch_size = batch_size, learning_rate = learning_rate,
      clip_range = clip_range,
      policy_kwargs = dict(
        activation_fn=activation_fn,
        net_arch=dict(
          pi=[layer_1_size, layer_2_size],
          vf=[layer_1_size, layer_2_size])
      ))
    episode_rewards = []
    callback = StopOnSuccessCallback(episode_rewards, trial)
    model.learn(480000, log_interval=10, callback=callback)
    print(episode_rewards)
    envs.close()
    return sum(episode_rewards) / len(episode_rewards)

study.optimize(objective, n_trials=50)
print(f"Best value: {study.best_value} (params: {study.best_params})")
