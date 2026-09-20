"""Physical Y-maze trials. Collection vials are explicitly absorbing states."""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np,pandas as pd
from .connectome import Connectome,train_and_delay
from .panel_b import FunctionalReadout,memory_factory,make_y_sim,collection_side,turn_command,wall_steering,upwind_steering

@lru_cache(maxsize=8)
def resources(path,cfgtext):
    cfg=json.loads(cfgtext);g=Connectome(path,cfg)
    return g,FunctionalReadout(g)

def run_panel_trial(task):
    cfg,gpath,model,condition,delay,replicate,paired,b_left,out=task
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    seed=int(np.random.SeedSequence([cfg["master_seed"],replicate,paired]).generate_state(1)[0])
    individual=cfg.get("individual_controller_v4",False)
    if individual:
        seed=int(np.random.SeedSequence([cfg["master_seed"],replicate,paired,int(b_left),
            int(round(delay*3600)),["consolidating","cascade"].index(model),
            ["paired","unpaired","untrained","dan_silenced"].index(condition)]).generate_state(1)[0])
    rng=np.random.default_rng(seed);gain=float(rng.lognormal(-.5*.3**2,.3) if individual else rng.lognormal(0,.1))
    phenotype={}
    if individual:
        child=np.random.SeedSequence(seed).spawn(4)
        phenotype=dict(gait_seed=int(child[0].generate_state(1)[0]),
            sensory_seed=int(child[1].generate_state(1)[0]),motor_seed=int(child[2].generate_state(1)[0]),
            heading_seed=int(child[3].generate_state(1)[0]),
            motor_noise_sd=float(cfg["motor_noise_sd"]*rng.lognormal(-.5*.2**2,.2)))
        sensory_rng=np.random.default_rng(phenotype["sensory_seed"])
        motor_rng=np.random.default_rng(phenotype["motor_seed"])
        cfg={**cfg,"gait_seed":phenotype["gait_seed"]}
        (out/"individual_configuration.json").write_text(json.dumps(dict(config=cfg,phenotype=phenotype,learning_gain=gain,seed=seed),indent=2))
    g,readout=resources(gpath,json.dumps({k:v for k,v in cfg.items() if k!="gait_seed"},sort_keys=True))
    m=memory_factory(len(g.ids["KC"]),cfg,model,gain)
    learning_history=[]
    def emit(mem,phase,session,trial,event,odor,orn,pn,kc,teaching,dt):
        learning_history.append(dict(phase=phase,session=session,trial=trial,event=event,biological_time_s=mem.time_s,dt_s=dt,
            odor_a_left=odor[0,0],odor_a_right=odor[0,1],odor_b_left=odor[1,0],odor_b_right=odor[1,1],
            ethanol=mem.ethanol,teaching_signal=teaching,aversive=mem.aversive.copy(),short=mem.short.copy(),
            long=mem.long.copy(),intermediate=getattr(mem,"intermediate",np.zeros_like(mem.long)).copy()))
    diagnostic=condition.startswith("clamp_")
    naive,learned,retained=train_and_delay(g,m,"untrained" if diagnostic else condition,paired,delay,emit)
    test_start=m.time_s;dan_on=condition!="dan_silenced"
    probe=[readout.probe(m,i,dan_on) for i in range(2)]
    heading_rng=np.random.default_rng(phenotype["heading_seed"]) if individual else rng
    fly,sim,ctl,polygon=make_y_sim(cfg,float(heading_rng.normal(0,.15 if individual else .08)))
    from flygym_demo.complex_terrain.common import apply_locomotion_action
    names=[s.name for s in fly.get_bodysegs_order()];ant=[names.index("l_funiculus"),names.index("r_funiculus")]
    thorax=sim._internal_bodyids_by_fly[fly.name][0]
    d=cfg["arm_mm"]/np.sqrt(2)
    sources=np.array([[cfg["stem_mm"]+d,-d],[cfg["stem_mm"]+d,d]])
    if not b_left:sources=sources[::-1].copy()
    steps=round(cfg["test_duration_s"]/sim.timestep);stride=round(cfg["control_dt_s"]/sim.timestep)
    physics=np.empty((steps,6),dtype=np.float64);nphysics=0
    states=[];qposes=[];qstates=[];noise=0.;choice="none";latency=None;status="test_window_complete";captured=False
    current_pos=sim.mj_data.xpos[thorax].copy();heading=0.;drive=np.ones(2);odor=np.zeros((2,2));v=np.zeros(2);turn=0.
    try:
        # Actual physics ends at collection; logical assay and memory continue to 120 s.
        for j in range(round(cfg["test_duration_s"]/cfg["control_dt_s"])):
            t=j*cfg["control_dt_s"]
            if not captured:
                current_pos=sim.mj_data.xpos[thorax].copy();rot=sim.mj_data.xmat[thorax].reshape(3,3);heading=float(np.arctan2(rot[1,0],rot[0,0]))
                ants=sim.get_body_positions(fly.name)[ant,:2];dist=np.linalg.norm(ants[:,None,:]-sources[None,:,:],axis=2)
                odor=np.maximum(np.exp(-dist/cfg["odor_length_mm"]).T+(sensory_rng if individual else rng).normal(0,cfg["odor_noise_sd"],(2,2)),0.)
                feat=readout.features(odor);v=readout.values(feat,m,dan_on)
                if diagnostic:
                    sign={"clamp_approach":1.,"clamp_avoid":-1.,"clamp_neutral":0.}[condition]
                    v=sign*odor[paired]
                noise=.9*noise+(motor_rng if individual else rng).normal(0,phenotype.get("motor_noise_sd",cfg["motor_noise_sd"]))*np.sqrt(.19)
                drive,turn=turn_command(cfg,odor,v,noise)
                wall_turn=wall_steering(ants,polygon)
                wind_turn=upwind_steering(current_pos,heading,cfg)
                turn=float(np.clip(turn+wall_turn+wind_turn,-cfg['max_turn'],cfg['max_turn']))
                drive=cfg['base_drive']*np.array([1-turn,1+turn])
            else:
                drive=np.zeros(2);turn=0.;wall_turn=0.;wind_turn=0.;v=np.zeros(2);odor=np.zeros((2,2))
            states.append(dict(state_index=j,fly_id=f"{model}_{condition}_{delay:g}h_{replicate}_{paired}_{int(b_left)}",model=model,condition=condition,
                phase="collection_hold" if captured else "test",conditioning_session=0,conditioning_trial=0,
                simulation_time_s=t,biological_time_s=m.time_s,x=float(current_pos[0]),y=float(current_pos[1]),z=float(current_pos[2]),heading_rad=heading,
                odor_a_left=odor[0,0],odor_a_right=odor[0,1],odor_b_left=odor[1,0],odor_b_right=odor[1,1],
                ethanol_state=m.ethanol,instantaneous_teaching_signal=0.,memory_reference="pre_test plus deterministic reward-free evolution",
                memory_elapsed_s=t,aversive_mean=float(m.aversive.mean()),short_mean=float(m.short.mean()),long_mean=float(m.long.mean()),
                intermediate_mean=float(getattr(m,"intermediate",np.zeros(1)).mean()),
                value_left=float(v[0]),value_right=float(v[1]),turn=turn,wall_turn=wall_turn,wind_turn=wind_turn,turn_noise=noise,action_left=drive[0],action_right=drive[1],
                random_seed=seed,paired_odor="AB"[paired],b_left=b_left,physics_active=not captured))
            m.advance(cfg["control_dt_s"],np.zeros(len(g.ids["KC"])),learning=False)
            if captured:continue
            for sub in range(stride):
                ctl.cpg_network.intrinsic_amps=np.repeat(drive,3)
                apply_locomotion_action(sim,fly.name,ctl.step());sim.step()
                pos=sim.mj_data.xpos[thorax].copy();rot=sim.mj_data.xmat[thorax].reshape(3,3)
                tm=t+(sub+1)*sim.timestep;angle=float(np.arctan2(rot[1,0],rot[0,0]))
                physics[nphysics]=(tm,*pos,angle,j);nphysics+=1
                if sub==0 and j%max(1,round(1/cfg["video_fps"]/cfg["control_dt_s"]))==0:
                    qposes.append(sim.mj_data.qpos.copy());qstates.append(j)
                if not np.isfinite(pos).all() or pos[2]<.15 or pos[2]>9:
                    status="physics_failure";break
                side=collection_side(pos,cfg)
                if side:
                    choice="B" if (side>0)==b_left else "A";latency=tm;captured=True
                    current_pos=pos;heading=angle;break
            if status=="physics_failure":break
        df=pd.DataFrame(states);df["final_choice"]=choice;df["choice_latency_s"]=latency
        df.to_parquet(out/"state_history.parquet",index=False,compression="zstd")
        pd.DataFrame(physics[:nphysics],columns=["simulation_time_s","x","y","z","heading_rad","state_index"]).to_parquet(out/"physics_history.parquet",index=False,compression="zstd")
        pd.DataFrame(learning_history).to_parquet(out/"learning_history.parquet",index=False,compression="zstd")
        np.savez_compressed(out/"replay.npz",qpos=qposes,state_index=qstates,sources=sources,arena_polygon=polygon)
        np.savez_compressed(out/"learning_states.npz",kc_root_ids=g.ids["KC"],
            **{prefix+"_"+k:v for prefix,snap in [("naive",naive),("post_training",learned),("pre_test",retained),("final",m.snapshot())] for k,v in snap.items()})
        result=dict(model=model,condition=condition,delay_h=delay,replicate=replicate,paired_odor="AB"[paired],b_left=b_left,
            fly_id=f"{model}_{condition}_{delay:g}h_{replicate}_{paired}_{int(b_left)}",**phenotype,
            seed=seed,learning_gain=gain,choice=choice,choice_latency_s=latency,termination=status,
            score=0 if choice=="none" else (1 if choice=="AB"[paired] else -1),
            neural_value_difference=probe[paired]-probe[1-paired],physics_ticks=nphysics,control_rows=len(states),
            assay_duration_s=cfg["test_duration_s"],test_start_biological_s=test_start,
            captured=captured,physical_time_s=nphysics*sim.timestep,
            capture_policy=cfg["capture_policy"],diagnostic=diagnostic)
        (out/"summary.json").write_text(json.dumps(result,indent=2))
        return result
    finally:sim.close()
