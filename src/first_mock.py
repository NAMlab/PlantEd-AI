import websockets
import json
import csv
import asyncio

async def step(
    grow_new_root,
    leaf_percent,
    stem_percent,
    root_percent,
    seed_percent,
    starch_percent,
    log_csv_writer):
  async with websockets.connect("ws://localhost:8765") as websocket:
    game_state = {
        "delta_t": 60 * 10,
        "growth_percentages": {
            "leaf_percent": leaf_percent,
            "stem_percent": stem_percent,
            "root_percent": root_percent,
            "seed_percent": seed_percent,
            "starch_percent": starch_percent,
            "stomata": True,
        },
        "increase_water_grid": None,
        "increase_nitrate_grid": None,
        "buy_new_root": {'directions': [(686.0, 60.0)]} if grow_new_root else {'directions': []}
    }
    print(json.dumps(game_state))
    await websocket.send(json.dumps(game_state))
    response = await websocket.recv()
    res = json.loads(response)
    res["plant"]["root"] = None
    res["environment"]["nitrate_grid"] = None
    res["environment"]["water_grid"] = None
    print(json.dumps(res, indent=2))
    log_csv_writer.writerow([
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
    return(res)

with open('log.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  writer.writerow(["time","temperature","sun_intensity", "humidity","precipitation",
      "leaf_biomass", "stem_biomass", "root_biomass", "seed_biomass", "starch_pool", "max_starch_pool", "water_pool", "max_water_pool",
      "leaf_percent", "stem_percent", "root_percent", "seed_percent", "starch_percent", "stomata"])

  asyncio.run(step(True, 0, 0, 100, 0, -20, writer))
  for i in range(150):
    print("root")
    asyncio.run(step(False, 0, 0, 100, 0, -20, writer))
  for i in range(150):
    print("leaf")
    asyncio.run(step(False, 100, 0, 0, 0, -20, writer))
