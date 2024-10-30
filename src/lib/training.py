# Common functions used for training the models.

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import torch as th

from lib.planted_env import PlantEdEnv

def create_model(envs):
  return PPO(policy = "MultiInputPolicy", env = envs, verbose=2, n_steps = 96, batch_size = 48, ent_coef = 0.05, device="cpu", learning_rate=0.0001, policy_kwargs = dict(
    activation_fn=th.nn.Tanh,
    net_arch=dict(
      pi=[64,32,32],
      vf=[64,32,32]),
    optimizer_kwargs = dict(
      betas = (0.99, 0.99))
  ))

def make_env(level_prefix, level_name, port):
    def _init():
        env = PlantEdEnv(f"{level_prefix}_{level_name}", port, level_name, False)
        env.reset()
        return env
    return _init

def normalize_envs(envs):
  return VecNormalize(envs, norm_obs_keys = [
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