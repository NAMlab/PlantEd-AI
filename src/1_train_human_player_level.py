# Try to find the best possible high score for the level the human player plays (re Figure 1 in the manuscript).
from lib.custom_subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO
from lib.training import create_model, make_env, normalize_envs

# Set to True if you want to continue training from a previous model
continue_training = False

envs = SubprocVecEnv([
  make_env('1_human_level', 'summer_low_nitrate', 8764),
], start_method='fork')
if continue_training:
  envs = VecNormalize.load("models/1_human_level.pkl", envs)
else:
  envs = normalize_envs(envs)

if continue_training:
  model = PPO.load("models/1_human_level", env=envs)
else:
  model = create_model(envs) 

model.learn(600*600*1, log_interval=10)
envs.close()

model.save("models/1_human_level")
envs.save("models/1_human_level.pkl")