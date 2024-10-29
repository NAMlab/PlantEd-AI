# This script is used to test the environment and server behaviour with a sequence of manually
# selected actions. It was useful during debugging.
from lib.planted_env import PlantEdEnv

env = PlantEdEnv("Manual-TestEnv", 8778, 'summer_low_nitrate')

s_size = env.observation_space.shape
a_size = env.action_space

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action

env.reset()

observation, reward, terminated, truncated, info = env.step(2)
print(observation["open_spots"][0])
env.step(0)
env.step(1)
env.step(2)
env.step(3)
env.step(4)
env.step(9)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
env.step(3)
# for i in range(0, 240):
#   env.step(2)
#for i in range(0, 240):
#  observation, reward, terminated, truncated, info = env.step(2)
#  print(observation)
#  if terminated:
#    break
#while True:
#  observation, reward, terminated, truncated, info = env.step(0)
#  if terminated:
#    break
#  observation, reward, terminated, truncated, info = env.step(1)
#  if terminated:
#    break

env.close()
