"""Render recorded MuJoCo generalized coordinates; no animated invented gait."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from fly_sud.physical import make_sim

if __name__=="__main__":
    cfg=json.loads((ROOT/"config/physical.json").read_text())
    for condition in (sys.argv[1:] or ["untrained","paired","unpaired"]):
        source=ROOT/"results/physical_pilot"/f"{condition}_000"
        replay=np.load(source/"replay.npz");state=pd.read_parquet(source/"state_history.parquet")
        output=ROOT/"output/frames"/condition;output.mkdir(parents=True,exist_ok=True)
        fly,sim,_=make_sim(cfg);renderer=mujoco.Renderer(sim.mj_model,height=352,width=608)
        cam=mujoco.MjvCamera();cam.distance=7.;cam.azimuth=105;cam.elevation=-35
        for i,(q,idx) in enumerate(zip(replay["qpos"],replay["state_index"])):
            sim.mj_data.qpos[:]=q;mujoco.mj_forward(sim.mj_model,sim.mj_data)
            cam.lookat[:]=sim.mj_data.xpos[sim._internal_bodyids_by_fly[fly.name][0]]
            renderer.update_scene(sim.mj_data,camera=cam)
            Image.fromarray(renderer.render()).save(output/f"{i:04d}.png")
        renderer.close();sim.close();print(condition,len(replay["qpos"]),flush=True)
