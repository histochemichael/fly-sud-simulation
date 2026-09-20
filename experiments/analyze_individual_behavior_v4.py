"""Analyze individual V4 results without changing completed experiments."""
from pathlib import Path
import sys,json,itertools,hashlib,zipfile
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
ROOT=Path(__file__).resolve().parents[1]
COLORS={"paired":"#e69f00","unpaired":"#009e73","untrained":"#56b4e9","dan_silenced":"#aa66cc"}

def aggregates(df):
 group=df.groupby(["condition","delay_h","replicate","paired_odor"]).score.mean()
 pairs=group.groupby(["condition","delay_h","replicate"]).mean().rename("cpi").reset_index()
 summary=pairs.groupby(["condition","delay_h"]).cpi.agg(["mean","std","count"]).reset_index()
 summary["sem"]=summary["std"]/np.sqrt(summary["count"])
 counts=df.groupby(["condition","delay_h"]).agg(flies=("score","size"),responders=("captured","sum"),target=("score",lambda s:(s==1).sum()),other=("score",lambda s:(s==-1).sum()),
   learning_gain_mean=("learning_gain","mean"),learning_gain_sd=("learning_gain","std"),
   pretest_value_mean=("neural_value_difference","mean"),pretest_value_sd=("neural_value_difference","std"),
   responder_latency_mean_s=("choice_latency_s","mean"),responder_latency_sd_s=("choice_latency_s","std"))
 return pairs,summary.merge(counts.reset_index(),on=["condition","delay_h"])

def exact_signflip(diff):
 d=np.asarray(diff,float)
 values=np.array([abs(np.mean(d*np.array(signs))) for signs in itertools.product([-1,1],repeat=len(d))])
 return float(np.mean(values>=abs(d.mean())-1e-12))

def analyze(run,output_name="analysis"):
 out=run/output_name;out.mkdir(exist_ok=False)
 policy=json.loads((run/"preregistered_policy.json").read_text())["policy"]
 selections=json.loads((run/"calibration_selection.json").read_text())
 validation=(run/"validation/choices.csv").exists()
 sets=[(p.parent.name,p) for p in sorted(run.glob("calibration_*/choices.csv"))]
 if validation:sets.append(("validation",run/"validation/choices.csv"))
 allsummary=[];audit=[];allseeds={};individual_manifest=[]
 for label,path in sets:
  df=pd.read_csv(path);pairs,summary=aggregates(df)
  pairs.to_csv(out/f"{label}_reciprocal_cpi.csv",index=False)
  summary.to_csv(out/f"{label}_summary.csv",index=False)
  summary.insert(0,"dataset",label);allsummary.append(summary)
  allseeds[label]=set(df.seed.astype(int))
  fig,ax=plt.subplots(figsize=(9,5))
  conditions=list(df.condition.unique());offsets=np.linspace(-.2,.2,len(conditions))
  for offset,c in zip(offsets,conditions):
   for i,delay in enumerate([.5,24.]):
    v=pairs.loc[(pairs.condition==c)&(pairs.delay_h==delay),"cpi"].to_numpy()
    if not len(v):continue
    ax.scatter(i+offset+np.linspace(-.025,.025,len(v)),v,s=25,alpha=.6,color=COLORS[c])
    ax.errorbar(i+offset,v.mean(),yerr=v.std(ddof=1)/np.sqrt(len(v)),fmt="D",capsize=5,color=COLORS[c],label=c if i==0 else None)
  ax.axhline(0,color=".6",lw=1);ax.set(xticks=[0,1],xticklabels=["30 min","24 h"],ylim=(-1.12,1.12),ylabel="Actual-choice CPI",xlabel="Retention delay")
  n_pairs=int(summary['count'].iloc[0]);n_flies=int(summary.flies.iloc[0])
  ax.set_title(f"{label}: mean ± SEM; dots = reciprocal pairs\nN={n_pairs} pairs, n={n_flies} flies per condition/delay")
  caption="Calibration only: descriptive outcomes, not confirmatory p-values."
  if label=="validation":
   raw=[]
   for delay in [.5,24.]:
    t=pairs[pairs.delay_h==delay].pivot(index="replicate",columns="condition",values="cpi")
    raw.append(exact_signflip(t.paired-t.unpaired))
   order=np.argsort(raw);adjusted=np.zeros(2);previous=0
   for rank,index in enumerate(order):
    previous=max(previous,min(1.,(2-rank)*raw[index]));adjusted[index]=previous
   caption=f"Paired vs unpaired: exact paired sign-flip, Holm p = {adjusted[0]:.4g} (30 min), {adjusted[1]:.4g} (24 h)"
  ax.legend();fig.text(.5,.01,caption,ha="center",fontsize=8)
  fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/f"{label}_cpi.png",dpi=180);plt.close(fig)
  conditions=[c for c in ["untrained","paired","unpaired","dan_silenced"] if c in df.condition.unique()]
  fig,axes=plt.subplots(2,len(conditions),figsize=(4*len(conditions)+2,7),sharey=True,squeeze=False)
  trajfig,taxes=plt.subplots(2,len(conditions),figsize=(4*len(conditions)+2,8),sharex=True,sharey=True,squeeze=False)
  for col,c in enumerate(conditions):
   for row,delay in enumerate([.5,24.]):
    subset=df[(df.condition==c)&(df.delay_h==delay)].sort_values(["replicate","paired_odor","b_left"])
    ax=axes[row,col];ta=taxes[row,col]
    ax.scatter(np.arange(len(subset)),subset.score,c=[("#e69f00" if s==1 else "#0072b2" if s==-1 else "#777777") for s in subset.score])
    ax.set(title=f"{c} | {delay:g} h",yticks=[-1,0,1],yticklabels=["Other","No response","Target"],xlabel="Individual fly",ylim=(-1.3,1.3))
    for r in subset.itertuples():
     trial=path.parent/c/f"{delay:g}h"/f"rep{r.replicate:02d}_odor{'AB'.index(r.paired_odor)}_side{int(r.b_left)}"
     individual_manifest.append(dict(dataset=label,individual_key=label+"/"+r.fly_id,
       relative_trial_directory=trial.relative_to(run).as_posix(),**r._asdict()))
     h=pd.read_parquet(trial/"physics_history.parquet")
     states=pd.read_parquet(trial/"state_history.parquet")
     replay=np.load(trial/"replay.npz");poly=replay["arena_polygon"]
     inside=MplPath(poly).contains_points(h[["x","y"]].to_numpy(),radius=1.)
     snap=np.load(trial/"learning_states.npz")
     cfg=json.loads((trial/"individual_configuration.json").read_text())["config"]
     batch_cfg=json.loads((path.parent/"config.json").read_text())
     config_matches={k:v for k,v in cfg.items() if k!="gait_seed"}==batch_cfg
     last=h.iloc[-1];along=(last.x-cfg["stem_mm"]+abs(last.y))/np.sqrt(2)
     across=abs(last.x-cfg["stem_mm"]-abs(last.y))/np.sqrt(2)
     in_collection=along>=cfg["arm_mm"]-cfg["collection_depth_mm"] and across<=cfg["corridor_width_mm"]/2
     physical_choice="B" if (last.y>0)==bool(r.b_left) else "A"
     collection_consistent=bool(in_collection and physical_choice==r.choice and abs(last.simulation_time_s-r.choice_latency_s)<1e-8) if r.captured else r.choice=="none"
     tail=h[h.simulation_time_s>=last.simulation_time_s-10]
     stationary_nonresponse=bool(not r.captured and np.linalg.norm(np.ptp(tail[["x","y"]].to_numpy(),axis=0))<1.)
     from scipy.linalg import expm
     k=1/(cfg["consolidation_tau_h"]*3600);b=1/(cfg["long_memory_tau_h"]*3600)
     initial=np.array([snap["pre_test_short"].mean(),snap["pre_test_long"].mean()])
     memory_error=0.
     for ix in [0,len(states)//2,len(states)-1]:
      elapsed=float(states.iloc[ix].memory_elapsed_s)
      expected=expm(np.array([[-k,0],[k,-b]])*elapsed)@initial
      expected_av=float(snap["pre_test_aversive"].mean())*np.exp(-elapsed/(cfg["aversive_tau_h"]*3600))
      memory_error=max(memory_error,abs(expected[0]-states.iloc[ix].short_mean),abs(expected[1]-states.iloc[ix].long_mean),abs(expected_av-states.iloc[ix].aversive_mean))
     moving=states[states.physics_active]
     lateral=cfg["behavioral_gain"]*(moving.value_left-moving.value_right)
     innate=cfg["innate_value"]*(moving.odor_a_left+moving.odor_b_left-moving.odor_a_right-moving.odor_b_right)
     replay_turn=np.clip(np.clip(np.tanh(lateral+innate)+moving.turn_noise,-cfg["max_turn"],cfg["max_turn"])+moving.wall_turn+moving.wind_turn,-cfg["max_turn"],cfg["max_turn"])
     turn_error=float(np.max(np.abs(replay_turn-moving.turn)))
     naive_zero=all(np.all(snap[k]==0) for k in snap.files if k.startswith("naive_") and k!="naive_time_s")
     audit.append(dict(dataset=label,fly_id=r.fly_id,seed=int(r.seed),physics_rows=len(h),finite=bool(np.isfinite(h.to_numpy()).all()),
       outside_tolerance=int((~inside).sum()),naive_zero=bool(naive_zero),
       final_state_present="final_long" in snap.files,stationary_nonresponse=stationary_nonresponse,config_matches=bool(config_matches),collection_consistent=bool(collection_consistent),memory_reconstruction_error=float(memory_error),turn_reconstruction_error=turn_error,
       unique_state_ids=bool(states.fly_id.nunique()==1 and states.fly_id.iloc[0]==r.fly_id),
       state_choice_consistent=bool((states.final_choice==r.choice).all()),
       learned_value_sd=float(states.loc[states.physics_active,"value_left"].std()),
       saturation_fraction=float((states.loc[states.physics_active,"turn"].abs()>=.59999).mean())))
     target_positive=(r.paired_odor=="B")==bool(r.b_left)
     sign=1 if target_positive else -1
     sampled=h.iloc[::100]
     ta.plot(sampled.x,sampled.y*sign,color="#e69f00" if r.score==1 else "#0072b2" if r.score==-1 else "#777777",alpha=.35,lw=.7)
     ta.scatter(h.x.iloc[-1],h.y.iloc[-1]*sign,s=5,color="black")
    closed=np.vstack([poly,poly[0]])
    ta.plot(closed[:,0],closed[:,1],color=".45",lw=1)
    from matplotlib.patches import Polygon
    for sign in [-1,1]:
     corners=np.array([[30+(along-cross)/np.sqrt(2),sign*(along+cross)/np.sqrt(2)] for along,cross in [(52,-5),(60,-5),(60,5),(52,5)]])
     ta.add_patch(Polygon(corners,facecolor="#e69f00",alpha=.15,edgecolor="none"))
    ta.set(title=f"{c} | {delay:g} h",xlabel="x (mm)",ylabel="Target-aligned y (mm)")
    ta.set_aspect("equal");ta.set_xlim(-10,90);ta.set_ylim(-55,55)
    ta.text(57,48,"Assigned target",fontsize=7);ta.text(63,-49,"Other",fontsize=8)
  fig.suptitle(f"{label}: one dot per simulated fly")
  fig.tight_layout();fig.savefig(out/f"{label}_individual_choices.png",dpi=160);plt.close(fig)
  trajfig.suptitle(f"{label}: orange target / blue other / gray nonresponse\nShaded tips = absorbing collection")
  trajfig.tight_layout();trajfig.savefig(out/f"{label}_individual_trajectories.png",dpi=160);plt.close(trajfig)
  # Latencies show censoring separately; nonresponse is not a measured latency.
  fig,ax=plt.subplots(figsize=(10,5))
  latency_labels=[]
  for i,((c,d),g) in enumerate(df.groupby(["condition","delay_h"])):
   responding=g[g.captured]
   if len(responding)>1 and responding.choice_latency_s.std()>0:
    violin=ax.violinplot([responding.choice_latency_s.to_numpy()],positions=[i],widths=.7,showextrema=False)
    for body in violin["bodies"]:body.set_facecolor(COLORS[c]);body.set_alpha(.2)
   ax.scatter(i+np.linspace(-.12,.12,len(responding)),responding.choice_latency_s,color=COLORS[c],s=20)
   latency_labels.append(f"{c}\n{d:g}h\nn={len(responding)}, NR={len(g)-len(responding)}")
  ax.set(xticks=range(len(latency_labels)),xticklabels=latency_labels,ylabel="Observed choice latency (s)",title=f"{label}: responder latencies, violin + individual dots\nNR = censored nonresponses at 120 s, not fabricated latency values")
  fig.tight_layout();fig.savefig(out/f"{label}_individual_latencies.png",dpi=180);plt.close(fig)
 summaries=pd.concat(allsummary,ignore_index=True);summaries.to_csv(out/"all_summaries.csv",index=False)
 pd.DataFrame(audit).to_csv(out/"individual_audit.csv",index=False)
 pd.DataFrame(individual_manifest).to_csv(out/"individual_manifest.csv",index=False)
 stats=[];criteria={}
 if validation:
  df=pd.read_csv(run/"validation/choices.csv");pairs,summary=aggregates(df)
  for delay in [.5,24.]:
   table=pairs[pairs.delay_h==delay].pivot(index="replicate",columns="condition",values="cpi")
   diff=(table.paired-table.unpaired).to_numpy()
   stats.append(dict(delay_h=delay,n_reciprocal_pairs=len(diff),contrast="paired minus unpaired",difference=float(diff.mean()),p_raw=exact_signflip(diff),test="exact two-sided paired sign flip"))
  order=np.argsort([r["p_raw"] for r in stats]);prev=0
  for rank,index in enumerate(order):
   prev=max(prev,min(1.,(len(stats)-rank)*stats[index]["p_raw"]));stats[index]["p_holm"]=prev
  paired=summary[summary.condition=="paired"]
  criteria=dict(correct_direction=bool(all(r.mean*(-1 if r.delay_h==.5 else 1)>0 for r in paired.itertuples())),
   nonsaturated_magnitude=bool(paired["mean"].abs().between(.05,.8).all()),
   mixed_paired_choices=bool(((paired.target>0)&(paired.other>0)).all()),
   controls_near_zero=bool(summary.loc[summary.condition.isin(["untrained","unpaired"]),"mean"].abs().le(.3).all()),
   response_at_least_80_percent=bool((summary.responders/summary.flies>=.8).all()),
   no_physics_failures=bool((df.termination!="physics_failure").all()),
   primary_contrasts_pass=bool(all(r["p_holm"]<.05 and r["difference"]*(-1 if r["delay_h"]==.5 else 1)>0 for r in stats)),
   seed_independence=bool(len(allseeds["validation"])==len(df) and all(not(allseeds["validation"]&s) for k,s in allseeds.items() if k!="validation")),
   physical_audit=bool(all(r["finite"] and not r["outside_tolerance"] and r["naive_zero"] and r["final_state_present"] and r["unique_state_ids"] and r["state_choice_consistent"] and r["config_matches"] and r["collection_consistent"] and r["memory_reconstruction_error"]<1e-9 and r["turn_reconstruction_error"]<1e-9 for r in audit if r["dataset"]=="validation")))
 (out/"statistics.json").write_text(json.dumps(dict(primary_tests=stats,criteria=criteria,all_pass=bool(criteria) and all(criteria.values())),indent=2))
 lines=[f"# {policy['version']} results","",policy.get("prior_information","First prospective individual-behavior calibration."),"",f"Status: {'validation completed' if validation else 'calibration gate failed; validation not run'}.",
  "","V3 is unchanged. New variability is explicitly modeled, not evidence of measured biological heterogeneity. No quantitative reproduction claim.",
  "",("Overall declared validation: "+("PASSED" if all(criteria.values()) else "FAILED")+". Failed criteria: "+(", ".join(k for k,v in criteria.items() if not v) or "none")+".") if validation else "No confirmatory validation was launched.",
  "",("Individual choice variation and the expected early/late direction were observed, but the entire predeclared statistical gate must pass before claiming validation. No post-validation retuning or exclusions were performed.") if validation else "Calibration outcomes must not be presented as independent validation.",
  "","## Actual CPI results","","Dataset | Condition | Delay h | Flies | Reciprocal N | Target / other / none | CPI ± SEM","---|---|---:|---:|---:|---|---:"]
 for r in summaries.itertuples():
  lines.append(f"{r.dataset} | {r.condition} | {r.delay_h:g} | {r.flies} | {r.count} | {r.target}/{r.other}/{r.flies-r.responders} | {r.mean:.3f} ± {r.sem:.3f}")
 lines+=["","## Declared validation criteria","",json.dumps(criteria,indent=2),"","## Primary statistics","",json.dumps(stats,indent=2),
  "","## Interpretation","",
  "CPI error bars are SEM across reciprocal replicate pairs, not individual fly movement variance. They quantify Monte Carlo replicate uncertainty under assumed individual distributions, not empirically calibrated biological variability. Each individual contributes one observed target/other/nonresponse score. Both individual choices and aggregate CPI are reported.",
  "","The paper reports early aversion and late preference with N=8 and mean ± SEM. Our early/late timing points were used in calibration and are not held-out timing evidence. Independent seeds test model repeatability, not biological validity.",
  "","The time-course mechanism was NOT fixed in this behavioral experiment: the V3 consolidating model retains its too-early sign crossing. Comparing memory mechanisms against separately specified time-course data is still required.",
  "",f"Stationary validation nonresponses: {sum(r['stationary_nonresponse'] for r in audit if r['dataset']=='validation')} (less than 1 mm spatial extent in the last 10 physical seconds). These remain in CPI denominators and are not interpreted as evidence of aversion.",
  "","The active V4 steering gain is behavioral_gain. Legacy turn_gain, motor_value_gain and motor_denominator_floor fields remain in inherited configurations but are not used when individual_controller_v4 is true.",
  "","All flies share one reference FlyWire connectome subset, with distinct seeded experiences and parameter profiles; these are not individually measured connectomes. Its anatomy does not specify the assumed noise, kinetics or motor equations. Dopamine is a simplified retrieval gate, not mechanistic DAN-to-KC learning.",
  "","The earlier open-arena experiment allowed a wider spread of paths; these physical Y-maze walls and obstacle/upwind reflexes constrain movement into corridors. Similar-looking paths within an arm are not the same thing as unanimous choices. Trajectory spread alone is not a target to fit. All assigned targets in control conditions are bookkeeping labels, not reward delivery sites.",
  "","Compute: local Intel Core i7-9700F, eight CPU workers, MuJoCo CPU dynamics. No GPU acceleration used for these trials.",
  "","## Data access","",
  "individual_manifest.csv indexes each fly, seed, outcome and trial directory. Each directory contains individual_configuration.json; state_history.parquet; physics_history.parquet; learning_history.parquet; learning_states.npz with naive/post-training/pre-test/final per-KC arrays and exact FlyWire root IDs; replay.npz; and summary.json. Use dataset plus fly_id as the unique key across gain candidates. Calibration candidates deliberately reuse matched individual seeds; validation does not.",
  "","Source: [Kaun et al. (2011), Figure 1 and Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/)",
  "","## Figures",""]
 for p in sorted(out.glob("*.png")):lines.append(f"![{p.stem}]({p.name})")
 (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
 with zipfile.ZipFile(run/"source_snapshot.zip") as z:
  manifest=json.loads((run/"preregistered_policy.json").read_text())
  for name,digest in manifest["source_sha256"].items():
   assert hashlib.sha256(z.read(Path(name).as_posix())).hexdigest()==digest
 with zipfile.ZipFile(out/"analysis_snapshot.zip","w",zipfile.ZIP_DEFLATED) as z:
  z.write(Path(__file__),Path(__file__).name)
  z.write(out/"RESULTS.md","RESULTS.md")
  z.write(ROOT/"tests/test_individual_behavior_v4.py","test_individual_behavior_v4.py")
 (out/"analysis_provenance.json").write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),policy_sha256=hashlib.sha256((run/"preregistered_policy.json").read_bytes()).hexdigest()),indent=2))
 print(out/"RESULTS.md")
if __name__=="__main__":analyze(Path(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else "analysis")
