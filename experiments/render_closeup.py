"""Close overhead replay of supplemental observations; no simulation stepping."""
import sys,json
from pathlib import Path
import numpy as np
from PIL import Image
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_sud.physical import make_sim
for condition in ['untrained','paired','unpaired','dan_silenced']:
    source=ROOT/'results/continuation_pilot'/f'{condition}_000'
    replay=np.load(source/'replay.npz');cfg=json.loads((ROOT/'config/physical.json').read_text())
    out=ROOT/'output/closeup_frames'/condition;out.mkdir(parents=True,exist_ok=True)
    fly,sim,_=make_sim(cfg);renderer=mujoco.Renderer(sim.mj_model,height=352,width=608)
    camera=mujoco.MjvCamera();camera.azimuth=0;camera.elevation=-90;camera.distance=11
    body=sim._internal_bodyids_by_fly[fly.name][0]
    for i,q in enumerate(replay['qpos']):
        sim.mj_data.qpos[:]=q;mujoco.mj_forward(sim.mj_model,sim.mj_data)
        pos=sim.mj_data.xpos[body];camera.lookat[:]=(pos[0],pos[1]-2.0,pos[2])
        renderer.update_scene(sim.mj_data,camera=camera)
        Image.fromarray(renderer.render()).save(out/f'{i:04d}.png')
    renderer.close();sim.close();print(condition,len(replay['qpos']),'close-up frames',flush=True)
