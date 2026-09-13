"""Render saved physical body poses from above; never re-run an experiment."""
import sys,json
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_sud.physical import make_sim
def project(points,cam):
    p=np.asarray(points,float);v=p-np.asarray(cam['pos']);f=np.asarray(cam['forward']);u=np.asarray(cam['up']);r=np.cross(f,u)
    depth=v@f;vertical=v@u;horizontal=v@r
    scale=cam['near']/depth
    return np.column_stack((304+horizontal*scale/cam['halfheight']*176,176-vertical*scale/cam['halfheight']*176))
def render(condition,probe=False):
    cfg=json.loads((ROOT/'config/physical.json').read_text());src=ROOT/'results/physical_pilot'/f'{condition}_000'
    replay=np.load(src/'replay.npz');states=pd.read_parquet(src/'state_history.parquet');summary=json.loads((src/'summary.json').read_text())
    output=ROOT/'output/overhead_frames'/condition;output.mkdir(parents=True,exist_ok=True)
    fly,sim,_=make_sim(cfg);renderer=mujoco.Renderer(sim.mj_model,height=352,width=608)
    camera=mujoco.MjvCamera();camera.azimuth=0;camera.elevation=-90
    # Color source markers only for replay presentation; physics is not stepped.
    for gid in range(sim.mj_model.ngeom):
        name=mujoco.mj_id2name(sim.mj_model,mujoco.mjtObj.mjOBJ_GEOM,gid) or ''
        if 'odor_marker_' in name:
            positive=not name.endswith('-1');is_b=positive==(summary['odor_b_side']=='left')
            sim.mj_model.geom_rgba[gid]=(.9,.52,.98,1) if is_b else (.3,.85,.8,1)
    cameras={}
    indices=range(len(replay['qpos']))
    if probe:indices=[0,next(i for i,s in enumerate(replay['state_index']) if states.iloc[s].phase=='test')]
    for i in indices:
        sim.mj_data.qpos[:]=replay['qpos'][i];mujoco.mj_forward(sim.mj_model,sim.mj_data)
        phase=states.iloc[replay['state_index'][i]].phase
        camera.lookat[:]=(20,0,0) if phase=='training' else (12,0,0)
        camera.distance=68 if phase=='training' else 48
        renderer.update_scene(sim.mj_data,camera=camera)
        gc=renderer.scene.camera[0];other=renderer.scene.camera[1]
        cameras[phase]={'pos':((gc.pos+other.pos)/2).tolist(),'forward':gc.forward.tolist(),'up':gc.up.tolist(),'near':float(gc.frustum_near),'halfheight':float((gc.frustum_top-gc.frustum_bottom)/2)}
        Image.fromarray(renderer.render()).save(output/f'{i:04d}.png')
    (output/'camera.json').write_text(json.dumps(cameras,indent=2));renderer.close();sim.close()
    print(condition,'overhead poses',len(indices),flush=True)
if __name__=='__main__':
    for c in ['untrained','paired','unpaired','dan_silenced']:render(c,'--probe' in sys.argv)
