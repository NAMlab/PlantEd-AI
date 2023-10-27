# Important! Comment out the server start in planted_env, start it manually and observe in a different window
# if you want.
import os

import gymnasium as gym

from planted_env import PlantEdEnv

env = PlantEdEnv("Environment 1")
env.reset()
while True:
  print("Step")
  observation, reward, terminated, truncated, info = env.step(2)
  if terminated:
    print("Re-setting")
    env.reset()


