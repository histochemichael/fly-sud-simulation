"""Read-only model diagnostics plus explicitly clamped, learning-disabled motor trials."""
from pathlib import Path
import sys,json,argparse,hashlib,importlib.metadata,platform,zipfile
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from fly_sud.connectome import Connectome,TemporalMemory,train_and_delay

def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den else None

def motor_command(cfg,odor,valence,noise,direct=None):
    injected=valence*odor[1]
    total=cfg["innate_value"]*odor.sum(axis=0)+cfg["motor_value_gain"]*injected
    contrast=float((total[0]-total[1])/(np.abs(total).sum()+.05))
    turn=float(np.clip(cfg["turn_gain"]*contrast+noise,-cfg["max_turn"],cfg["max_turn"])) if direct is None else direct
    return cfg["base_drive"]*np.array([1-turn,1+turn]),injected,contrast,turn

def motor_trial(task):
    cfg,seed,side,mode,folder=task
    from fly_sud.physical import make_sim
    from flygym_demo.complex_terrain.common import apply_locomotion_action
    out=Path(folder);out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(seed);initial_heading=float(rng.normal(0,.12))
    fly,sim,ctl=make_sim(cfg,initial_heading);ctl.cpg_network.reset()
    names=[x.name for x in fly.get_bodysegs_order()];ant=[names.index("l_funiculus"),names.index("r_funiculus")]
    thorax=sim._internal_bodyids_by_fly[fly.name][0]
    sources=np.array([[cfg["source_x_mm"],-side*cfg["source_y_mm"]],[cfg["source_x_mm"],side*cfg["source_y_mm"]]])
    stride=round(cfg["control_dt_s"]/sim.timestep);frames=round(1/cfg["video_fps"]/sim.timestep)
    states=[];physics=[];qpos=[];qindex=[];noise=0.;choice="none";latency=None;termination="test_window_complete"
    valence={"approach":1.,"neutral":0.,"avoid":-1.,"direct_positive":0.,"direct_negative":0.}[mode]
    direct={"direct_positive":.3,"direct_negative":-.3}.get(mode)
    try:
        for i in range(round(cfg["test_duration_s"]/sim.timestep)):
            if i%stride==0:
                ants=sim.get_body_positions(fly.name)[ant,:2]
                d=np.linalg.norm(ants[:,None,:]-sources[None,:,:],axis=2)
                odor=np.maximum(np.exp(-d/cfg["odor_length_mm"]).T+rng.normal(0,cfg["odor_noise_sd"],(2,2)),0)
                noise=.9*noise+rng.normal(0,cfg["motor_noise_sd"])*np.sqrt(.19)
                drive,injected,contrast,turn=motor_command(cfg,odor,valence,noise,direct)
                p=sim.mj_data.xpos[thorax];r=sim.mj_data.xmat[thorax].reshape(3,3)
                states.append(dict(state_index=len(states),fly_id=f"{seed}_{side}_{mode}",condition=mode,phase="diagnostic_test",conditioning_session=0,conditioning_trial=0,
                    simulation_time_s=i*sim.timestep,x=float(p[0]),y=float(p[1]),z=float(p[2]),heading_rad=float(np.arctan2(r[1,0],r[0,0])),
                    odor_a_left=odor[0,0],odor_a_right=odor[0,1],odor_b_left=odor[1,0],odor_b_right=odor[1,1],
                    ethanol_state=0.,teaching_signal=0.,memory_state="disabled; unchanged zero state",injected_left=injected[0],injected_right=injected[1],
                    contrast=contrast,turn_noise=noise,turn=turn,action_left=drive[0],action_right=drive[1],random_seed=seed,target_side=side))
            ctl.cpg_network.intrinsic_amps=np.repeat(drive,3)
            apply_locomotion_action(sim,fly.name,ctl.step());sim.step()
            p=sim.mj_data.xpos[thorax].copy();r=sim.mj_data.xmat[thorax].reshape(3,3);t=(i+1)*sim.timestep
            physics.append((t,*p,float(np.arctan2(r[1,0],r[0,0])),len(states)-1))
            if i%frames==0:qpos.append(sim.mj_data.qpos.copy());qindex.append(len(states)-1)
            if choice=="none" and p[0]>=cfg["choice_x_mm"] and abs(p[1])>=cfg["choice_abs_y_mm"]:
                choice="target" if p[1]*side>0 else "other";latency=t
            if not np.isfinite(p).all() or p[2]<.15:termination="physics_failure";break
        result=dict(seed=seed,target_side=side,mode=mode,choice=choice,choice_latency_s=latency,termination=termination,
                    final_x=float(p[0]),final_y=float(p[1]),target_signed_y=float(p[1]*side),initial_heading=initial_heading,
                    physics_ticks=len(physics),control_rows=len(states),learning_enabled=False)
        df=pd.DataFrame(states);df["final_choice"]=choice;df["choice_latency_s"]=latency
        df.to_parquet(out/"state_history.parquet",index=False,compression="zstd")
        pd.DataFrame(physics,columns=["simulation_time_s","x","y","z","heading_rad","state_index"]).to_parquet(out/"physics_history.parquet",index=False,compression="zstd")
        np.savez_compressed(out/"replay.npz",qpos=qpos,state_index=qindex,sources=sources)
        np.savez_compressed(out/"learning_states.npz",initial=np.zeros(5177),final=np.zeros(5177),learning_enabled=False)
        (out/"summary.json").write_text(json.dumps(result,indent=2))
        return result
    finally:sim.close()

def perception(source,out,cfg):
    graph=Connectome(ROOT/"data/connectome_v783",cfg);rows=[];swaps=[];probes=[]
    prototypes=[]
    for identity in range(2):
        o=np.zeros((2,2));o[identity]=1;prototypes.append(graph.response(o)[2])
    for concentration in [.05,.1,.15,.2,.25,.3,.4,.5,.75,1.]:
        patterns=[]
        for identity in range(2):
            o=np.zeros((2,2));o[identity]=concentration;orn,pn,kc=graph.response(o);patterns.append(kc)
            similarities=[cosine(kc,p) for p in prototypes]
            probes.append(dict(concentration=concentration,identity="AB"[identity],active_orn=int((orn>0).sum()),active_pn=int((pn>0).sum()),active_kc=int((kc>0).sum()),
                cosine_training_A=similarities[0],cosine_training_B=similarities[1],kc_norm=float(np.linalg.norm(kc)),
                identifiable_by_training_prototypes=bool(np.linalg.norm(kc)>0 and np.argmax([s or 0 for s in similarities])==identity)))
    for path in sorted(source.glob("*/*/*/batch*/summary.json")):
        s=json.loads(path.read_text());cols=["state_index","phase","biological_time_s","kc_rates","pn_rates","aversive_memory","long_memory","motor_value_left","motor_value_right","odor_a_left","odor_a_right","odor_b_left","odor_b_right"]
        d=pd.read_parquet(path.parent/"state_history.parquet",columns=cols);d=d[d.phase=="test"]
        if s["choice_latency_s"] is not None:d=d[d.biological_time_s-s["test_start_biological_s"]<=s["choice_latency_s"]]
        for n,r in enumerate(d.itertuples()):
            kc=np.asarray(r.kc_rates);pn=np.asarray(r.pn_rates)
            weight=np.asarray(r.aversive_memory)+np.asarray(r.long_memory)
            odor=np.array([[r.odor_a_left,r.odor_a_right],[r.odor_b_left,r.odor_b_right]])
            v=np.array([r.motor_value_left,r.motor_value_right])
            innate=cfg["innate_value"]*odor.sum(axis=0)
            total=innate+cfg["motor_value_gain"]*v
            contrast=(total[0]-total[1])/(np.abs(total).sum()+.05)
            baseline=(innate[0]-innate[1])/(np.abs(innate).sum()+.05)
            rows.append(dict(trial=str(path.parent.relative_to(source)),model=s["model"],condition=s["condition"],delay_h=s["delay_h"],
                state_index=r.state_index,active_kc=int((kc>0).sum()),active_pn=int((pn>0).sum()),
                accessible_memory_fraction=float(weight[kc>0].sum()/weight.sum()) if weight.sum() else None,
                learned_motor_absmax=float(np.max(np.abs(v))),learned_turn_delta=float(cfg["turn_gain"]*(contrast-baseline))))
            if n%20==0:
                swapped=graph.response(odor[::-1])[2]
                swaps.append(dict(trial=str(path.parent.relative_to(source)),state_index=r.state_index,
                    original_swapped_cosine=cosine(kc,swapped),relative_identity_difference=float(np.linalg.norm(kc-swapped)/max(np.linalg.norm(kc),1e-15))))
    pd.DataFrame(rows).to_parquet(out/"perception_prechoice.parquet",index=False,compression="zstd")
    pd.DataFrame(swaps).to_csv(out/"perception_identity_swaps.csv",index=False)
    pd.DataFrame(probes).to_csv(out/"perception_concentration_probes.csv",index=False)
    d=pd.DataFrame(rows)
    d.groupby(["model","condition"]).agg(samples=("active_kc","size"),min_active_kc=("active_kc","min"),max_motor_signal=("learned_motor_absmax","max"),
        mean_accessible_memory=("accessible_memory_fraction","mean"),max_learned_turn_delta=("learned_turn_delta",lambda x:abs(x).max())).to_csv(out/"perception_summary.csv")

def memory_test(out,configs):
    from scipy.integrate import solve_ivp
    rows=[];checks=[];histories=[]
    for model,cfg in configs.items():
        g=Connectome(ROOT/"data/connectome_v783",cfg);m=TemporalMemory(len(g.ids["KC"]),cfg,model)
        def emit(mem,phase,session,trial,label,odor,orn,pn,kc,teaching,dt):
            histories.append(dict(model=model,phase=phase,biological_time_s=mem.time_s,session=session,trial=trial,dt_s=dt,
                ethanol=mem.ethanol,teaching=teaching,aversive=mem.aversive.copy(),short=mem.short.copy(),long=mem.long.copy()))
        naive,learned,_=train_and_delay(g,m,"paired",1,0,emit)
        y0=[m.aversive.mean(),m.short.mean(),m.long.mean()]
        def ode(t,y):
            av,short,long=y;k=1/(cfg["consolidation_tau_h"]*3600)
            return [-av/(cfg["aversive_tau_h"]*3600),-k*short if model=="consolidating" else 0,
                (k*short if model=="consolidating" else 0)-long/(cfg["long_memory_tau_h"]*3600)]
        ts=np.arange(0,24.00001,.25)
        sol=solve_ivp(ode,(0,24*3600),y0,t_eval=ts*3600,rtol=1e-10,atol=1e-13)
        errors=[];first=None
        for i,t in enumerate(ts):
            if i:m.advance(900,np.zeros(len(g.ids["KC"])),learning=False)
            values=[]
            for identity in range(2):
                odor=np.zeros((2,2));odor[identity]=1
                values.append(float(g.decode(g.response(odor)[2],m,np.ones(len(g.ids["DAN"])))[0].mean()))
            v=values[1]-values[0]
            if first is None and v>0:first=float(t)
            errors.append(float(np.max(np.abs(np.array([m.aversive.mean(),m.short.mean(),m.long.mean()])-sol.y[:,i]))))
            rows.append(dict(model=model,delay_h=t,internal_value_difference=v,aversive_mean=m.aversive.mean(),short_mean=m.short.mean(),long_mean=m.long.mean()))
            histories.append(dict(model=model,phase="delay_probe",biological_time_s=m.time_s,session=0,trial=0,dt_s=900 if i else 0,ethanol=m.ethanol,teaching=0.,
                aversive=m.aversive.copy(),short=m.short.copy(),long=m.long.copy()))
        np.savez_compressed(out/(model+"_memory_states.npz"),kc_root_ids=g.ids["KC"],
            **{prefix+"_"+k:v for prefix,snap in [("naive",naive),("post_training",learned),("final",m.snapshot())] for k,v in snap.items()})
        checks.append(dict(model=model,first_positive_sample_h=first,crossing_resolution_h=.25,max_retention_ode_error=max(errors),
            held_out_paper_transition_h=[12,15],matches_held_out_timing=12<=first<=15 if first is not None else False,
            endpoint_signs_previously_calibrated=True,refitted=False))
    pd.DataFrame(rows).to_csv(out/"memory_timecourse.csv",index=False)
    pd.DataFrame(histories).to_parquet(out/"memory_history.parquet",index=False,compression="zstd")
    (out/"memory_checks.json").write_text(json.dumps(checks,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--workers",type=int,default=8);a=p.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False)
    source=ROOT/"results/connectome_time_paper_v2";configs=json.loads((source/"config_snapshot.json").read_text());cfg=configs["consolidating"]
    seeds=[int(np.random.SeedSequence([19431,i]).generate_state(1)[0]) for i in range(8)]
    files=list((ROOT/"src").rglob("*.py"))+[Path(__file__)]
    metadata=dict(configs=configs,master_seed=19431,seeds=seeds,workers=a.workers,platform=platform.platform(),processor=platform.processor(),
        packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        prior_run=str(source.relative_to(ROOT)),prior_config_sha256=hashlib.sha256((source/"config_snapshot.json").read_bytes()).hexdigest(),
        intervention="Motor tests bypass connectome decoder; +/- odor B concentration is a diagnostic value clamp, not reward or learning.",
        notes=["No parameter fitting or production-model changes.","Eight matched seeds, two target placements, three valences; plus eight direct-steer trials.",
               "Within-seed repeats are not independent biological samples.","Perception reuses all 240 prior trials before first choice.",
               "Memory uses fixed unit-strength odor probes, no motion; only retention ODE is independently checked.",
               "CPU MuJoCo; no GPU acceleration requested. No publication."])
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    with zipfile.ZipFile(out/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,path.relative_to(ROOT))
    tasks=[(cfg,s,side,mode,str(out/"motor"/f"{s}_{side}_{mode}")) for s in seeds for side in [-1,1] for mode in ["approach","neutral","avoid"]]
    tasks += [(cfg,s,1,mode,str(out/"motor"/f"{s}_1_{mode}")) for s in seeds[:4] for mode in ["direct_positive","direct_negative"]]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(motor_trial,t) for t in tasks]
        perception(source,out,cfg);print("Perception complete",flush=True)
        memory_test(out,configs);print("Memory complete",flush=True)
        results=[]
        for f in as_completed(futures):
            results.append(f.result());print(f"Motor {len(results)}/{len(tasks)}",flush=True)
    pd.DataFrame(results).sort_values(["seed","target_side","mode"]).to_csv(out/"motor_choices.csv",index=False)
    print("COMPLETE",out,flush=True)
if __name__=="__main__":main()

