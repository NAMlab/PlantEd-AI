import csv
import sys
import os
import asyncio
import json
import websockets
import multiprocessing
import time

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from PlantEd_Server.server import server

class PlantEdEnv(gym.Env):
  metadata = {"render_modes": ["ansi"], "render_fps": 30}
  max_steps = 200

  def __init__(self, instance_name="PlantEd_instance", port=8765):
    self.port = port
    self.last_step_biomass = 0.0
    self.instance_name = instance_name
    self.running = False

    self.action_space = spaces.Discrete(7)
    self.observation_space = spaces.Dict({
      # Environment
      "temperature": spaces.Box(-200, 200),
      "sun_intensity": spaces.Box(-1, 1),
      "humidity": spaces.Box(0, 100),
      # Plant
      "biomasses": spaces.Box(0, 1000, shape=(4,)), #leaf, stem, root, seed
      "starch_pool": spaces.Box(0, 1000),
      "max_starch_pool": spaces.Box(0, 1000),
      "stomata_state": spaces.MultiBinary(1)
      })

  def __del__(self):
    self.close()

  def close(self):
    self.csv_file.close()
    self.server_process.terminate()
    self.running = False

  def reset(self, seed=None, options=None):
    if self.running:
      self.close()
    self.stomata = True
    self.init_csv_logger()
    self.server_process = multiprocessing.Process(target=self.start_server)
    self.server_process.start()
    self.running = True
    time.sleep(20)
    self.current_step = 0
    observation, reward, terminated, truncated, info = self.step(8)
    return(observation,info)


  def render(self):
    pass

  def start_server(self):
    sys.stdout = open("server.log", 'w')
    sys.stderr = open("server.err", 'w')
    server.start(self.port)

  def init_csv_logger(self):
    self.csv_file = open(self.instance_name + '.csv', 'w', newline='')
    self.csv_writer = csv.writer(self.csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    self.csv_writer.writerow(["time","temperature","sun_intensity", "humidity","precipitation",
        "leaf_biomass", "stem_biomass", "root_biomass", "seed_biomass", "starch_pool", "max_starch_pool", "water_pool", "max_water_pool",
        "leaf_percent", "stem_percent", "root_percent", "seed_percent", "starch_percent", "stomata"])

  def step(self, action):
    return(asyncio.run(self._async_step(action)))

  # Action Space:
  # 1 - grow leaves
  # 2 - grow stem
  # 3 - grow roots
  # 4 - grow seeds
  # 5 - accumulate starch
  # 6 - open stomata
  # 7 - close stomata
  # 8 - grow new root
  # Observation Space:
  # [
  #  # Environment
  #  temperature,
  #  sun_intensity,
  #  humidity,
  #
  #  # Plant
  #  leaf_biomass,
  #  stem_biomass,
  #  root_biomass,
  #  seed_biomass,
  #  starch_pool,
  #  max_starch_pool,
  #  stomata_state
  # ]
  async def _async_step(self,action):
    terminated = False
    truncated = False
    # @TODO if anything goes wrong here with the server etc. we should output a "truncated" = true
    # except websockets.exceptions.ConnectionClosedError
    async with websockets.connect("ws://localhost:" + str(self.port)) as websocket:
      if action == 6:
        self.stomata = True
      if action == 7:
        self.stomata = False
      game_state = {
          "delta_t": 60 * 10,
          "growth_percentages": {
              "leaf_percent": 100 if action == 1 else 0,
              "stem_percent": 100 if action == 2 else 0,
              "root_percent": 100 if action == 3 else 0,
              "seed_percent": 100 if action == 4 else 0,
              "starch_percent": 100 if action in [5,6,7,8] else -100, # make starch when manipulating stomata or new roots
              "stomata": self.stomata,
          },
          "increase_water_grid": None,
          "increase_nitrate_grid": None,
          "buy_new_root": {'directions': [(686.0, 60.0)]} if action == 8 else {'directions': []}
      }
      await websocket.send(json.dumps(game_state))
      response = await websocket.recv()
      res = json.loads(response)
      self.write_log_row(res, game_state)
      observation = {
          "temperature": np.array([res["environment"]["temperature"]]).astype(np.float32),
          "sun_intensity": np.array([res["environment"]["sun_intensity"]]).astype(np.float32),
          "humidity": np.array([res["environment"]["humidity"]]).astype(np.float32),

          "biomasses": np.array([
            res["plant"]["leaf_biomass"],
            res["plant"]["stem_biomass"],
            res["plant"]["root_biomass"],
            res["plant"]["seed_biomass"],
            ]).astype(np.float32),
          "starch_pool": np.array([res["plant"]["starch_pool"]]).astype(np.float32),
          "max_starch_pool": np.array([res["plant"]["max_starch_pool"]]).astype(np.float32),

          "stomata_state": np.array([game_state["growth_percentages"]["stomata"]])
        }

      reward = self.calc_reward(res)
      self.current_step += 1
      if self.current_step > self.max_steps:
        terminated = True
      return(observation, reward, terminated, truncated, {})

  def calc_reward(self, res):
    total_biomass = res["plant"]["leaf_biomass"] + res["plant"]["stem_biomass"] + res["plant"]["root_biomass"] + res["plant"]["seed_biomass"]
    reward = total_biomass - self.last_step_biomass
    self.last_step_biomass = total_biomass
    return(reward)

  def write_log_row(self, res, game_state):
    self.csv_writer.writerow([
      res["environment"]["time"],
      res["environment"]["temperature"],
      res["environment"]["sun_intensity"],
      res["environment"]["humidity"],
      res["environment"]["precipitation"],

      res["plant"]["leaf_biomass"],
      res["plant"]["stem_biomass"],
      res["plant"]["root_biomass"],
      res["plant"]["seed_biomass"],
      res["plant"]["starch_pool"],
      res["plant"]["max_starch_pool"],
      res["plant"]["water_pool"],
      res["plant"]["max_water_pool"],

      game_state["growth_percentages"]["leaf_percent"],
      game_state["growth_percentages"]["stem_percent"],
      game_state["growth_percentages"]["root_percent"],
      game_state["growth_percentages"]["seed_percent"],
      game_state["growth_percentages"]["starch_percent"],
      game_state["growth_percentages"]["stomata"]
      ])


if __name__ == "__main__":
  from stable_baselines3.common.env_checker import check_env

  env = PlantEdEnv("Test-Env")
  # It will check your custom environment and output additional warnings if needed
  check_env(env)
  env.close()