"""Actual FlyGym/MuJoCo trials and complete control/physics audit trails."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mujoco
from flygym import Simulation
from flygym.compose import FlatGroundWorld, ActuatorType
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain.common import make_locomotion_fly, apply_locomotion_action
from flygym_demo.complex_terrain.cpg_controller import CPGController, make_tripod_cpg_network
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps
from .circuit import RewardCircuit, training_schedule


def make_sim(config, heading=0.):
    fly=make_locomotion_fly()
    fly.colorize()
    world=FlatGroundWorld(half_size=100)
    # Physical outer walls define a two-choice arena, an allowed Y-maze equivalent.
    for name,pos,size in [
        ("left_wall",(11,17,1),(15,.25,1)),
        ("right_wall",(11,-17,1),(15,.25,1)),
        ("back_wall",(-4,0,1),(.25,17,1)),
        ("front_wall",(26,0,1),(.25,17,1))]:
        world.mjcf_root.worldbody.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,pos=pos,size=size,rgba=(.18,.23,.29,1))
    for side in [-1,1]:
        world.mjcf_root.worldbody.add_geom(name=f"odor_marker_{side}",type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=(config["source_x_mm"],side*config["source_y_mm"],.06),size=(1.3,.05,0),rgba=(.2,.6,.7,1),contype=0,conaffinity=0)
    world.add_fly(fly,(0,0,.3),Rotation3D("quat",(np.cos(heading/2),0,0,np.sin(heading/2))))
    sim=Simulation(world,timestep=config["physics_dt_s"])
    dofs=fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    ctl=CPGController(make_tripod_cpg_network(sim.timestep),PreprogrammedSteps(),dofs)
    sim.reset()
    return fly,sim,ctl


def run_physical_trial(args):
    config,condition,fly_id,outdir,record_training,save_replay=args
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    cond_id=["untrained","paired","unpaired","dan_silenced"].index(condition)
    # Perturbation uses matched paired seeds; no model receives the condition label.
    seed_seq=np.random.SeedSequence([config["master_seed"],1 if cond_id==3 else cond_id,fly_id])
    seeds=seed_seq.generate_state(3)
    rng=np.random.default_rng(int(seeds[0]))
    circuit=RewardCircuit(config,int(seeds[1]))
    initial=circuit.weights.copy()
    fly,sim,ctl=make_sim(config,rng.normal(0,.12))
    ctl.cpg_network.reset()
    names=[x.name for x in fly.get_bodysegs_order()]
    ant_idx=[names.index("l_funiculus"),names.index("r_funiculus")]
    thorax=sim._internal_bodyids_by_fly[fly.name][0]
    control_stride=round(config["control_dt_s"]/sim.timestep)
    frame_stride=round(1/config["video_fps"]/sim.timestep)
    states=[]; physics=[]; qpos=[]; replay_rows=[]
    t_global=0.; phase_step=0; held={}; current_drive=np.ones(2); turn_noise=0.
    b_left=bool(rng.integers(2))
    sources=np.array([[config["source_x_mm"],-config["source_y_mm"]],[config["source_x_mm"],config["source_y_mm"]]])
    if not b_left: sources=sources[::-1].copy()
    source_order=sources.copy()
    def control(odor,reward,train,session,trial,label,t):
        nonlocal current_drive,turn_noise,held
        contrast,activity,kc=circuit.step(odor,reward,train,config["control_dt_s"],condition=="dan_silenced" and not train)
        turn_noise=.9*turn_noise+rng.normal(0,config["motor_noise_sd"])*np.sqrt(.19)
        turn=float(np.clip(config["turn_gain"]*contrast+turn_noise,-config["max_turn"],config["max_turn"]))
        current_drive=config["base_drive"]*np.array([1-turn,1+turn])
        pos=sim.get_body_positions(fly.name)[0]
        mat=sim.mj_data.xmat[thorax].reshape(3,3)
        held=dict(condition=condition,fly_id=fly_id,phase="training" if train else "test",
                  conditioning_session=session,conditioning_trial=trial,phase_time_s=t,
                  simulation_time_s=t_global,x=float(pos[0]),y=float(pos[1]),z=float(pos[2]),
                  heading_rad=float(np.arctan2(mat[1,0],mat[0,0])),
                  odor_a_left=float(odor[0,0]),odor_a_right=float(odor[0,1]),
                  odor_b_left=float(odor[1,0]),odor_b_right=float(odor[1,1]),
                  ethanol_state=float(reward>0),internal_reward_state=reward,teaching_signal=reward,
                  action_left=float(current_drive[0]),action_right=float(current_drive[1]),
                  random_seed=int(seeds[0]),circuit_seed=int(seeds[1]),odor_b_side="left" if b_left else "right",
                  event=label,**activity)
        for k,w in enumerate(circuit.weights): held[f"kc_weight_{k:02d}"]=float(w)
        for k in range(len(kc)):
            held[f"kc_activity_l_{k:02d}"]=float(kc[k,0]); held[f"kc_activity_r_{k:02d}"]=float(kc[k,1])
        states.append(held.copy())
    def tick(i):
        nonlocal t_global
        ctl.cpg_network.intrinsic_amps=np.repeat(current_drive,3)
        apply_locomotion_action(sim,fly.name,ctl.step());sim.step()
        t_global+=sim.timestep
        pos=sim.mj_data.xpos[thorax]
        rot=sim.mj_data.xmat[thorax].reshape(3,3)
        # Every MuJoCo tick is retained; state_index joins the held neural/sensor/action values.
        physics.append((t_global,float(pos[0]),float(pos[1]),float(pos[2]),float(np.arctan2(rot[1,0],rot[0,0])),len(states)-1))
        if save_replay and i%frame_stride==0:
            qpos.append(sim.mj_data.qpos.copy()); replay_rows.append(len(states)-1)
    # Full circuit training uses all control updates for every fly. Physical
    # training footage is recorded for representatives; batch training is stationary.
    for session,trial,odor,ethanol,label in training_schedule(condition,config):
        duration=config["training_exposure_s"]
        for j in range(round(duration/config["control_dt_s"])):
            control(odor,ethanol*config["reward_strength"],True,session,trial,label,(trial-1)*duration+j*config["control_dt_s"])
            if record_training:
                for sub in range(control_stride): tick(j*control_stride+sub)
            else: t_global+=config["control_dt_s"]
    learned=circuit.weights.copy()
    # Transfer to test arena; resetting body/gait does not alter learned weights.
    sim.reset();ctl.cpg_network.reset();turn_noise=0.
    choice="none";latency=None;distance=0.;prev=sim.get_body_positions(fly.name)[0].copy()
    continue_after_choice=bool(config.get("continue_after_choice",False))
    first_choice_distance=None;termination_reason="test_window_complete"
    test_start=t_global
    for i in range(round(config["test_duration_s"]/sim.timestep)):
        if i%control_stride==0:
            positions=sim.get_body_positions(fly.name)[ant_idx,:2]
            d=np.linalg.norm(positions[:,None,:]-sources[None,:,:],axis=2)
            odor=(np.exp(-d/config["odor_length_mm"]).T+rng.normal(0,config["odor_noise_sd"],(2,2))).clip(0)
            control(odor,0.,False,0,1,"reward-free choice",i*sim.timestep)
        tick(i)
        pos=sim.mj_data.xpos[thorax].copy()
        distance+=float(np.linalg.norm(pos-prev));prev=pos
        if choice=="none" and pos[0]>=config["choice_x_mm"] and abs(pos[1])>=config["choice_abs_y_mm"]:
            choice="B" if (pos[1]>0)==b_left else "A";latency=(i+1)*sim.timestep
            first_choice_distance=distance
            if not continue_after_choice:
                termination_reason="first_choice";break
        if not np.all(np.isfinite(pos)) or pos[2]<.15:
            if choice=="none":choice="physics_failure"
            termination_reason="physics_failure";break
    summary=dict(condition=condition,fly_id=fly_id,choice=choice,completed=choice in ["A","B"],
                 latency_s=latency,distance_mm=first_choice_distance if first_choice_distance is not None else distance,odor_b_side="left" if b_left else "right",
                 random_seed=int(seeds[0]),circuit_seed=int(seeds[1]),last_x_mm=float(prev[0]),last_y_mm=float(prev[1]),
                 weight_a=float(learned[::2].mean()),weight_b=float(learned[1::2].mean()),
                 physical_training=record_training,physics_ticks=len(physics),control_updates=len(states),
                 continued_after_choice=continue_after_choice,observation_distance_mm=distance,
                 observation_end_s=(i+1)*sim.timestep,termination_reason=termination_reason)
    df=pd.DataFrame(states);df["state_index"]=np.arange(len(df));df["final_choice"]=choice;df["choice_latency_s"]=latency
    df["post_choice"]=(df.phase=="test") & (df.phase_time_s>=latency) if latency is not None else False
    df.to_parquet(outdir/"state_history.parquet",index=False,compression="zstd")
    pdf=pd.DataFrame(physics,columns=["simulation_time_s","x","y","z","heading_rad","state_index"])
    pdf["post_choice"]=pdf.simulation_time_s>test_start+latency if latency is not None else False
    pdf.to_parquet(outdir/"physics_history.parquet",index=False,compression="zstd")
    np.savez_compressed(outdir/"learning_states.npz",naive=initial,learned=learned,final=circuit.weights,projection=circuit.projection)
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2))
    if save_replay: np.savez_compressed(outdir/"replay.npz",qpos=np.asarray(qpos),state_index=np.asarray(replay_rows),sources=source_order)
    sim.close()
    return summary
