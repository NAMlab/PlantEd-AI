from stable_baselines3.common.vec_env import VecNormalize
from lib.custom_subproc_vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from lib.training import create_model, make_env, normalize_envs

# Set to True if you want to continue training from a previous model
continue_training = False

envs = SubprocVecEnv([
  make_env('spring_high_nitrate', 8771),
  make_env('spring_low_nitrate', 8772),
  make_env('summer_high_nitrate', 8773),
  make_env('summer_low_nitrate', 8774),
  make_env('summer_low_nitrate_old', 8775),
  make_env('fall_high_nitrate', 8776),
  make_env('fall_low_nitrate', 8777)
], start_method='fork')

if continue_training:
  envs = VecNormalize.load("models/2_multiEnvironment.pkl", envs)
else:
  envs = normalize_envs(envs)

if continue_training:
  model = PPO.load("models/2_multiEnvironment", env=envs)
else:
  model = create_model(envs)

model.learn(600*600*7, log_interval=10)
envs.close()

model.save("models/2_multiEnvironment")
envs.save("models/2_multiEnvironment.pkl")

