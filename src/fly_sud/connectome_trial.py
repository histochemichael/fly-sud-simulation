"""Independent-delay FlyGym trials driven by measured FlyWire connectivity."""
from pathlib import Path
from functools import lru_cache
import json,numpy as np,pandas as pd
from .connectome import Connectome,TemporalMemory,train_and_delay
@lru_cache(maxsize=4)
def get_graph(path,cfg):return Connectome(path,json.loads(cfg))
def run_trial(task):
    cfg,gpath,model,condition,delay,batch,individual,out,physical=task
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    seed=int(np.random.SeedSequence([cfg["master_seed"],batch,individual,int(delay*3600)]).generate_state(1)[0])
    rng=np.random.default_rng(seed);paired=(batch+individual)%2
    g=get_graph(gpath,json.dumps(cfg,sort_keys=True))
    m=TemporalMemory(len(g.ids["KC"]),cfg,model,float(rng.lognormal(0,.1)))
    states=[];physics=[];qposes=[];qstates=[]
    position=np.full(3,np.nan);heading=np.nan;drive=np.full(2,np.nan)
    def emit(mem,phase,session,trial,event,odor,orn,pn,kc,teaching,dt):
        on=not ((condition=="dan_acquisition" and phase=="training") or (condition=="dan_consolidation" and phase=="delay") or (condition=="dan_silenced" and phase=="test"))
        dan=np.full(len(g.ids["DAN"]),float(on));v,app,av,gate=g.decode(kc,mem,dan)
        states.append(dict(state_index=len(states),fly_id=f"{batch:02d}_{individual:03d}",batch=batch,model=model,condition=condition,delay_h=delay,phase=phase,conditioning_session=session,conditioning_trial=trial,
          biological_time_s=mem.time_s,interval_duration_s=dt,x=float(position[0]),y=float(position[1]),z=float(position[2]),heading_rad=float(heading),
          odor_a_left=float(odor[0,0]),odor_a_right=float(odor[0,1]),odor_b_left=float(odor[1,0]),odor_b_right=float(odor[1,1]),
          ethanol_state=mem.ethanol,instantaneous_teaching_signal=float(mem.ethanol if phase=="training" else 0),interval_mean_teaching_signal=float(teaching if phase=="training" else 0),
          action_left=float(drive[0]),action_right=float(drive[1]),decoded_value_left=float(v[0]),decoded_value_right=float(v[1]),random_seed=seed,paired_odor="AB"[paired],event=event,
          orn_rates=orn.astype("float32"),pn_rates=pn.astype("float32"),kc_rates=kc.astype("float32"),dan_rates=dan.astype("float32"),
          mbon_appetitive=app.astype("float32"),mbon_aversive=av.astype("float32"),
          aversive_memory=mem.aversive.astype("float32"),short_memory=mem.short.astype("float32"),long_memory=mem.long.astype("float32")))
    naive,learned,retained=train_and_delay(g,m,condition,paired,delay,emit)
    start=m.time_s;probe=[]
    for identity in range(2):
        odor=np.zeros((2,2));odor[identity]=1;_,_,kc=g.response(odor)
        dan=np.full(len(g.ids["DAN"]),0. if condition=="dan_silenced" else 1.)
        probe.append(float(g.decode(kc,m,dan)[0].mean()))
    choice="not_tested";latency=None;termination="neural_probe_only";distance=0.;b_left=bool(rng.integers(2))
    if physical:
        from .physical import make_sim
        from flygym_demo.complex_terrain.common import apply_locomotion_action
        fly,sim,ctl=make_sim(cfg,rng.normal(0,.12));ctl.cpg_network.reset()
        names=[x.name for x in fly.get_bodysegs_order()];ant=[names.index("l_funiculus"),names.index("r_funiculus")]
        thorax=sim._internal_bodyids_by_fly[fly.name][0]
        sources=np.array([[cfg["source_x_mm"],-cfg["source_y_mm"]],[cfg["source_x_mm"],cfg["source_y_mm"]]])
        if not b_left:sources=sources[::-1].copy()
        stride=round(cfg["control_dt_s"]/sim.timestep);frame_stride=round(1/cfg["video_fps"]/sim.timestep)
        choice="none";termination="test_window_complete";noise=0.;prev=sim.mj_data.xpos[thorax].copy()
        try:
            for i in range(round(cfg["test_duration_s"]/sim.timestep)):
                if i%stride==0:
                    position=sim.mj_data.xpos[thorax].copy();rot=sim.mj_data.xmat[thorax].reshape(3,3);heading=float(np.arctan2(rot[1,0],rot[0,0]))
                    ants=sim.get_body_positions(fly.name)[ant,:2];d=np.linalg.norm(ants[:,None,:]-sources[None,:,:],axis=2)
                    odor=np.maximum(np.exp(-d/cfg["odor_length_mm"]).T+rng.normal(0,cfg["odor_noise_sd"],(2,2)),0)
                    orn,pn,kc=g.response(odor);dan=np.full(len(g.ids["DAN"]),0. if condition=="dan_silenced" else 1.)
                    v,_,_,_=g.decode(kc,m,dan)
                    if cfg.get("center_motor_readout",False):
                        from .connectome_protocol import centered_motor_value
                        v,reference=centered_motor_value(g,m,odor,kc,dan)
                    total=cfg["innate_value"]*odor.sum(axis=0)+cfg["motor_value_gain"]*v
                    contrast=(total[0]-total[1])/(np.abs(total).sum()+.05)
                    noise=.9*noise+rng.normal(0,cfg["motor_noise_sd"])*np.sqrt(.19)
                    turn=float(np.clip(cfg["turn_gain"]*contrast+noise,-cfg["max_turn"],cfg["max_turn"]))
                    drive=cfg["base_drive"]*np.array([1-turn,1+turn])
                    emit(m,"test",0,1,"reward-free physical test",odor,orn,pn,kc,0.,cfg["control_dt_s"])
                    states[-1]["motor_value_left"]=float(v[0]);states[-1]["motor_value_right"]=float(v[1])
                    m.advance(cfg["control_dt_s"],np.zeros(len(kc)),learning=False)
                ctl.cpg_network.intrinsic_amps=np.repeat(drive,3)
                apply_locomotion_action(sim,fly.name,ctl.step());sim.step()
                pos=sim.mj_data.xpos[thorax].copy();rot=sim.mj_data.xmat[thorax].reshape(3,3)
                t=(i+1)*sim.timestep;distance+=float(np.linalg.norm(pos-prev));prev=pos
                physics.append((start+t,t,float(pos[0]),float(pos[1]),float(pos[2]),float(np.arctan2(rot[1,0],rot[0,0])),len(states)-1))
                if i%frame_stride==0:qposes.append(sim.mj_data.qpos.copy());qstates.append(len(states)-1)
                if choice=="none" and pos[0]>=cfg["choice_x_mm"] and abs(pos[1])>=cfg["choice_abs_y_mm"]:
                    choice="B" if (pos[1]>0)==b_left else "A";latency=t
                if not np.isfinite(pos).all() or pos[2]<.15:
                    termination="physics_failure"
                    if choice=="none":choice="physics_failure"
                    break
            np.savez_compressed(out/"replay.npz",qpos=np.asarray(qposes),state_index=np.asarray(qstates),sources=sources)
        finally:sim.close()
    result=dict(model=model,condition=condition,delay_h=delay,batch=batch,individual=individual,fly_id=f"{batch:02d}_{individual:03d}",seed=seed,paired_odor="AB"[paired],b_left=b_left,choice=choice,choice_latency_s=latency,
        paired_choice=choice=="AB"[paired],completed=choice in ("A","B"),termination_reason=termination,distance_mm=distance,
        neural_paired_minus_unpaired_value=probe[paired]-probe[1-paired],physical=physical,learning_gain=m.gain,
        test_start_biological_s=start,control_rows=len(states),physics_ticks=len(physics))
    df=pd.DataFrame(states);df["final_choice"]=choice;df["choice_latency_s"]=latency
    df.to_parquet(out/"state_history.parquet",index=False,compression="zstd")
    if physical:pd.DataFrame(physics,columns=["biological_time_s","test_time_s","x","y","z","heading_rad","state_index"]).to_parquet(out/"physics_history.parquet",index=False,compression="zstd")
    snapshots={}
    for name,snap in [("naive",naive),("post_training",learned),("pre_test",retained),("final",m.snapshot())]:
        for k,v in snap.items():snapshots[name+"_"+k]=v
    np.savez_compressed(out/"learning_states.npz",kc_root_ids=g.ids["KC"],**snapshots)
    (out/"summary.json").write_text(json.dumps(result,indent=2))
    return result
