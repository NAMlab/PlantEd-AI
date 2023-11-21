import csv
import sys
import os
import glob
import asyncio
import json
import websockets
import multiprocessing
import collections
import time
import random
from enum import Enum

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from PlantEd.server import server

Biomass = collections.namedtuple("Biomass", "leaf stem root seed")

# Action Space
Action = Enum('Action', [ 
  'GROW_LEAVES',
  'GROW_STEM',
  'GROW_ROOTS',
  'PRODUCE_STARCH',
  'OPEN_STOMATA',
  'CLOSE_STOMATA',
  'BUY_LEAF',
  'BUY_STEM',
  'BUY_ROOT',
  'BUY_SEED'
  ], start=0) # start at 0 because the gym action space starts at 0
# @TODO add these next:
# 10 - watering can
# 11 - add fertilizer

class PlantEdEnv(gym.Env):
  metadata = {"render_modes": ["ansi"], "render_fps": 30}

  def __init__(self, instance_name="PlantEd_instance", port=8765):
    self.port = port
    self.instance_name = instance_name
    self.csv_file = None
    self.server_process = None
    self.running = False
    self.game_counter = 0 # add 1 at each reset --> to save all the logs
    # Remove previous game logs
    # for f in glob.glob('game_logs/*.csv'):
    #   os.remove(f)
    self.reset_callback = None

    # See Action enum above for action space.
    self.action_space = spaces.Discrete(len(Action))
    self.observation_space = spaces.Dict({
      # Environment
      "temperature": spaces.Box(-200, 200),
      "sun_intensity": spaces.Box(-1, 1),
      "humidity": spaces.Box(0, 100),
      "accessible_water": spaces.Box(0, 6000000),
      "accessible_nitrate": spaces.Box(0, 1200),
      "green_thumbs": spaces.Box(0, 25),
      # Plant
      "biomasses": spaces.Box(0, 100, shape=(4,)), #leaf, stem, root, seed
      "n_organs": spaces.Box(0, 25, shape=(4,)), #leaf, stem, root, seed
      "open_spots": spaces.Box(0, 100),
      "starch_pool": spaces.Box(0, 100),
      "max_starch_pool": spaces.Box(0, 100),
      "stomata_state": spaces.MultiBinary(1)
      })

  def __del__(self):
    self.close()

  def close(self):
    if(self.csv_file):
      self.csv_file.close()
    if(self.server_process):
      self.server_process.terminate()
      print("Terminated server process, waiting 5 sec")
      time.sleep(5)
    self.running = False

  def reset(self, seed=None, options=None):
    if self.reset_callback:
      self.reset_callback(self)
    print("RESET CALLED")
    if self.running:
      self.close()

    self.game_counter += 1
    self.init_csv_logger()

    self.server_process = multiprocessing.Process(target=self.start_server)
    self.server_process.start()
    print("Server process started, waiting 10 sec...")
    time.sleep(10)
    asyncio.run(self.load_level())

    self.running = True
    self.last_step_score = -1
    self.total_rewards = 0
    self.stomata = True
    self.last_observation = None

    observation, reward, terminated, truncated, info = self.step(2)
    return(observation, info)

  async def load_level(self):
    async with websockets.connect("ws://localhost:" + str(self.port)) as websocket:
      message = {
        "type": "load_level",
        "message": {
          "player_name": "Planted-AI",
          "level_name": "LEVEL_NAME",
        }
      }
      await websocket.send(json.dumps(message))
      response = await websocket.recv()
      print(response)

  def render(self):
    pass

  def start_server(self):
    sys.stdout = open("server.log", 'w')
    sys.stderr = open("server.err", 'w')
    server.start(self.port)

  def init_csv_logger(self):
    self.csv_file = open('game_logs/' + self.instance_name + '_run' + str(self.game_counter) + '.csv', 'w', newline='')
    self.csv_writer = csv.writer(self.csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    self.csv_writer.writerow(["time","temperature","sun_intensity", "humidity","precipitation","accessible_water","accessible_nitrate",
        "leaf_biomass", "stem_biomass", "root_biomass", "seed_biomass", "starch_pool", "max_starch_pool", "water_pool", "max_water_pool",
        "leaf_percent", "stem_percent", "root_percent", "seed_percent", "starch_percent", 
        "n_leaves", "n_stems", "n_roots", "n_seeds", "green_thumbs", "open_spots",
        "action", "reward"])

  def get_biomasses(self, res):
    leaf = sum([x[1] for x in res["plant"]["leafs_biomass"]])
    stem = sum([x[1] for x in res["plant"]["stems_biomass"]])
    root = res["plant"]["roots_biomass"]
    seed = sum([x[1] for x in res["plant"]["seeds_biomass"]])
    return(Biomass(leaf, stem, root, seed))

  def get_n_organs(self, res):
    return([
     len(res["plant"]["leafs_biomass"]),
     len(res["plant"]["stems_biomass"]),
     len(res["plant"]["root"]["first_letters"]),
     len(res["plant"]["seeds_biomass"]),
    ])

  async def execute_step(self, message):
    terminated = False
    truncated = False
    async with websockets.connect("ws://localhost:" + str(self.port)) as websocket:
      try:
        await websocket.send(json.dumps(message))
        response = await websocket.recv()
        res = json.loads(response)
        terminated = not res["running"]
      except websockets.exceptions.ConnectionClosedError:
        print("SERVER CRASHED")
        res = None
        truncated = True

    return(res, terminated, truncated)

  @staticmethod
  def build_observation(res, biomasses, n_organs, stomata):
    root_grid = np.array(res["plant"]["root"]["root_grid"])
    nitrate_grid = np.array(res["environment"]["nitrate_grid"])
    water_grid = np.array(res["environment"]["water_grid"])

    res["environment"]["accessible_water"] = (root_grid * water_grid).sum()
    res["environment"]["accessible_nitrate"] = (root_grid * nitrate_grid).sum()

    observation = {
        # Environment
        "temperature": np.array([res["environment"]["temperature"]]).astype(np.float32),
        "sun_intensity": np.array([res["environment"]["sun_intensity"]]).astype(np.float32),
        "humidity": np.array([res["environment"]["humidity"]]).astype(np.float32),
        "accessible_water": np.array([res["environment"]["accessible_water"]]).astype(np.float32),
        "accessible_nitrate": np.array([res["environment"]["accessible_nitrate"]]).astype(np.float32),
        "green_thumbs": np.array([res["green_thumbs"]]).astype(np.float32),

        # Plant
        "biomasses": np.array([
          biomasses.leaf,
          biomasses.stem,
          biomasses.root,
          biomasses.seed,
          ]).astype(np.float32),
        "n_organs": np.array(n_organs).astype(np.float32),
        "open_spots": np.array([res["plant"]["open_spots"]]).astype(np.float32),
        "starch_pool": np.array([res["plant"]["starch_pool"]]).astype(np.float32),
        "max_starch_pool": np.array([res["plant"]["max_starch_pool"]]).astype(np.float32),
        "stomata_state": np.array([stomata])
      }
    return(observation)

  def custom_step(self, message):
    print("step. " + "CUSTOM")
    res, terminated, truncated = asyncio.run(self.execute_step(message))
    if truncated:
      observation = self.last_observation # @TODO This is not optimal because it pretends that what the actor did had no influence at all.
      reward = 0.0
    else:
      biomasses = self.get_biomasses(res)
      n_organs = self.get_n_organs(res)
      observation = self.build_observation(res, biomasses, n_organs, self.stomata)
      self.last_observation = observation

      reward = self.calc_reward(biomasses, "CUSTOM")
      self.total_rewards += reward
      self.write_log_row(res, message["message"], biomasses, n_organs, "CUSTOM", reward)

    return(observation, reward, terminated, truncated, {})

  def step(self, a):
    action = Action(a)
    print("step. " + action.name)
    if action == Action.OPEN_STOMATA:
      self.stomata = True
    if action == Action.CLOSE_STOMATA:
      self.stomata = False
    message = {
      "type": "simulate",
      "message": {
        "delta_t": 6 * 60 * 10,
        "growth_percentages": {
          "leaf_percent": 100 if action == Action.GROW_LEAVES else 0,
          "stem_percent": 100 if action == Action.GROW_STEM else 0,
          "root_percent": 100 if action == Action.GROW_ROOTS else 0,
          "seed_percent": 0,
          "starch_percent": 100 if action not in [Action.GROW_LEAVES, Action.GROW_STEM, Action.GROW_ROOTS] else -100, # make starch when manipulating stomata or new roots
          "stomata": self.stomata,
        },
        "shop_actions":{
          "buy_watering_can": None,
          "buy_nitrate": None,
          "buy_leaf": 1 if action == Action.BUY_LEAF else None,
          "buy_branch": 1 if action == Action.BUY_STEM else None,
          "buy_root": {'directions': [[random.gauss(0, 0.4), random.gauss(1, 0.3)]]} if action == Action.BUY_ROOT else None,
          "buy_seed": 1 if action == Action.BUY_SEED else None
        }
      }
    }

    res, terminated, truncated = asyncio.run(self.execute_step(message))
    if truncated:
      observation = self.last_observation # @TODO This is not optimal because it pretends that what the actor did had no influence at all.
      reward = 0.0
    else:
      biomasses = self.get_biomasses(res)
      n_organs = self.get_n_organs(res)
      observation = self.build_observation(res, biomasses, n_organs, self.stomata)
      self.last_observation = observation

      reward = self.calc_reward(biomasses, action)
      self.total_rewards += reward
      self.write_log_row(res, message["message"], biomasses, n_organs, action.name, reward)

    return(observation, reward, terminated, truncated, {})

  def calc_reward(self, biomasses, action):
    current_score = biomasses.seed + (biomasses.leaf + biomasses.stem + biomasses.root) * 0.01
    reward = 0 if self.last_step_score == -1 else current_score - self.last_step_score
    self.last_step_score = current_score
    # Punish opening and closing of stomata as well as buying organs to push AI to use "produce starch" if it wants to do that
    if action in [Action.OPEN_STOMATA, Action.CLOSE_STOMATA, Action.BUY_LEAF, Action.BUY_STEM, Action.BUY_ROOT, Action.BUY_SEED]:
      reward = reward * 0.9 - 1e-4
    return(reward)

  def write_log_row(self, res, game_state, biomasses, n_organs, action, reward):
    self.csv_writer.writerow([
      res["environment"]["time"],
      res["environment"]["temperature"],
      res["environment"]["sun_intensity"],
      res["environment"]["humidity"],
      res["environment"]["precipitation"],
      res["environment"]["accessible_water"],
      res["environment"]["accessible_nitrate"],

      biomasses.leaf,
      biomasses.stem,
      biomasses.root,
      biomasses.seed,
      res["plant"]["starch_pool"],
      res["plant"]["max_starch_pool"],
      res["plant"]["water_pool"],
      res["plant"]["max_water_pool"],

      game_state["growth_percentages"]["leaf_percent"],
      game_state["growth_percentages"]["stem_percent"],
      game_state["growth_percentages"]["root_percent"],
      game_state["growth_percentages"]["seed_percent"],
      game_state["growth_percentages"]["starch_percent"],

      n_organs[0], # leaf
      n_organs[1], # stems
      n_organs[2], # root
      n_organs[3], # seed
      res["green_thumbs"],
      res["plant"]["open_spots"],

      action,
      reward
      ])


if __name__ == "__main__":
  from stable_baselines3.common.env_checker import check_env

  env = PlantEdEnv("Test-Env")
  # It will check your custom environment and output additional warnings if needed
  check_env(env)
  env.close()
