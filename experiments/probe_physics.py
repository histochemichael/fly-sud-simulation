from pathlib import Path
import time
import numpy as np
import mujoco
from PIL import Image
from flygym import Simulation
from flygym.compose import FlatGroundWorld, ActuatorType
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain.common import make_locomotion_fly, apply_locomotion_action
from flygym_demo.complex_terrain.cpg_controller import CPGController, make_tripod_cpg_network
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps

if __name__ == '__main__':
    fly=make_locomotion_fly()
    world=FlatGroundWorld(half_size=100)
    world.add_fly(fly,(0,0,0.3),Rotation3D('quat',(1,0,0,0)))
    sim=Simulation(world)
    ctl=CPGController(make_tripod_cpg_network(sim.timestep),PreprogrammedSteps(),fly.get_actuated_jointdofs_order(ActuatorType.POSITION))
    sim.reset()
    start=time.perf_counter()
    for i in range(15000):
        ctl.cpg_network.intrinsic_amps=np.repeat([1.0,1.0],3)
        apply_locomotion_action(sim,fly.name,ctl.step())
        sim.step()
        if i%5000==0: print(i,sim.get_body_positions(fly.name)[0],flush=True)
    print('elapsed',time.perf_counter()-start, 'final',sim.get_body_positions(fly.name)[0],flush=True)
    cam=mujoco.MjvCamera(); cam.lookat[:]=sim.get_body_positions(fly.name)[0]; cam.distance=7; cam.azimuth=110; cam.elevation=-35
    render=mujoco.Renderer(sim.mj_model,height=480,width=640)
    render.update_scene(sim.mj_data,camera=cam)
    Path('output').mkdir(exist_ok=True)
    Image.fromarray(render.render()).save('output/physical_probe.png')
    render.close();sim.close()
