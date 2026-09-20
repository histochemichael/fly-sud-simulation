"""Score actual choices; never substitute internal values for CPI."""
from pathlib import Path
import sys,json,itertools,hashlib,zipfile
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/sys.argv[1];d=pd.read_csv(out/"choices.csv");meta=json.loads((out/"metadata.json").read_text())
    fig=out/"figures";fig.mkdir(exist_ok=True)
    # Each reciprocal group has two counterbalanced placements, including nonresponders.
    groups=d.groupby(["model","condition","delay_h","replicate","paired_odor"]).score.mean().rename("PI").reset_index()
    cpi=groups.groupby(["model","condition","delay_h","replicate"]).PI.mean().rename("CPI").reset_index()
    groups.to_csv(out/"reciprocal_group_pi.csv",index=False);cpi.to_csv(out/"replicate_cpi.csv",index=False)
    summary=cpi.groupby(["model","condition","delay_h"]).CPI.agg(["mean","std","count"]).reset_index()
    summary["sem"]=summary["std"]/np.sqrt(summary["count"])
    counts=d.groupby(["model","condition","delay_h"]).agg(trials=("score","size"),positive_choices=("score",lambda s:(s==1).sum()),
        negative_choices=("score",lambda s:(s==-1).sum()),nonresponders=("choice",lambda s:(s=="none").sum()),
        physics_failures=("termination",lambda s:(s=="physics_failure").sum())).reset_index()
    summary=summary.merge(counts,on=["model","condition","delay_h"]);summary.to_csv(out/"summary.csv",index=False)
    stats=[]
    for model,md in cpi.groupby("model"):
        for delay,hd in md.groupby("delay_h"):
            for control in ["untrained","unpaired"]:
                pivot=hd[hd.condition.isin(["paired",control])].pivot(index="replicate",columns="condition",values="CPI")
                if not {"paired",control}<=set(pivot.columns):continue
                delta=(pivot.paired-pivot[control]).dropna().to_numpy();effect=float(delta.mean())
                null=[np.mean(delta*np.array(s)) for s in itertools.product([-1,1],repeat=len(delta))]
                stats.append(dict(model=model,delay_h=float(delay),comparison="paired minus "+control,n_reciprocal_pairs=len(delta),
                    effect=effect,test="two-sided exact paired sign flip on replicate CPI; symmetric-null assumption",
                    p=float(np.mean(np.abs(null)>=abs(effect)-1e-12))))
    order=np.argsort([r["p"] for r in stats]);last=0
    for rank,i in enumerate(order):last=max(last,min(1.,stats[i]["p"]*(len(stats)-rank)));stats[i]["p_holm"]=last
    (out/"statistics.json").write_text(json.dumps(stats,indent=2))
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    for model,part in d.groupby("model"):
        conditions=list(part.condition.unique());f,axes=plt.subplots(1,len(conditions),figsize=(5*len(conditions),5),squeeze=False)
        for ax,condition in zip(axes[0],conditions):
            for r in part[part.condition==condition].itertuples():
                folder=out/model/condition/f"{r.delay_h:g}h"/f"rep{r.replicate:02d}_odor{'AB'.index(r.paired_odor)}_side{int(r.b_left)}"
                h=pd.read_parquet(folder/"physics_history.parquet")
                side=1 if ((r.paired_odor=='B')==bool(r.b_left)) else -1
                ax.plot(h.x,h.y*side,lw=.65,alpha=.65,color="#2679b4" if r.delay_h<1 else "#d26a2c")
                if len(h):ax.scatter(h.x.iloc[-1],h.y.iloc[-1]*side,s=12,color="black")
            replay=np.load(folder/"replay.npz");p=replay["arena_polygon"];p=np.vstack([p,p[0]])
            ax.plot(p[:,0],p[:,1],color="gray",lw=1);ax.set_aspect("equal");ax.margins(.1)
            from matplotlib.patches import Polygon
            cfg=meta['configs'][model];c=1/np.sqrt(2)
            for side in [-1,1]:
                along=np.array([c,side*c]);across=np.array([-side*c,c]);origin=np.array([cfg['stem_mm'],0.])
                corners=[origin+s*along+w*across for s,w in [(cfg['arm_mm']-cfg['collection_depth_mm'],-cfg['corridor_width_mm']/2),
                    (cfg['arm_mm'],-cfg['corridor_width_mm']/2),(cfg['arm_mm'],cfg['corridor_width_mm']/2),
                    (cfg['arm_mm']-cfg['collection_depth_mm'],cfg['corridor_width_mm']/2)]]
                ax.add_patch(Polygon(corners,color='#efb755',alpha=.3,zorder=0))
            ax.text(.98,.95,'Assigned target arm',transform=ax.transAxes,ha='right',fontsize=8)
            ax.text(.98,.05,'Other arm',transform=ax.transAxes,ha='right',fontsize=8)
            ax.set(title=condition,xlabel="x (mm)",ylabel="Target-aligned y (mm)")
        f.suptitle(model+": target-aligned moving paths; blue early / orange late; shaded tips = absorbing collection",fontsize=10)
        f.tight_layout();f.savefig(fig/(model+"_trajectories.png"),dpi=140);plt.close(f)
    if "paired" in set(cpi.condition):
        models=list(cpi.model.unique());f,axes=plt.subplots(1,len(models),figsize=(6*len(models),4.5),squeeze=False)
        for ax,model in zip(axes[0],models):
            part=cpi[(cpi.model==model)&(cpi.condition=="paired")]
            for j,(delay,h) in enumerate(part.groupby("delay_h"),1):
                vals=h.CPI.to_numpy()
                if len(vals)>1 and np.ptp(vals)>0:ax.violinplot([vals],positions=[j],showextrema=False)
                ax.scatter(j+np.linspace(-.08,.08,len(vals)),vals,s=35,c="#247ba0",zorder=3)
            delays=sorted(part.delay_h.unique());ax.set_xticks(range(1,len(delays)+1),[f"{t:g} h" for t in delays])
            n=part.groupby('delay_h').replicate.nunique().min()
            ax.set_ylim(-1.1,1.1);ax.axhline(0,color="gray",lw=.7);ax.set(title=model+f' | N={n} reciprocal pairs per delay',ylabel="Actual-choice CPI",xlabel="Retention delay")
        f.suptitle("Physical pilot — dots are reciprocal replicate pairs, not individual flies")
        tests=[r for r in stats if r['comparison']=='paired minus unpaired']
        f.text(.5,.01,'Paired vs unpaired: exact paired sign-flip; '+ '; '.join(f"{r['model']} {r['delay_h']:g}h Holm p={r['p_holm']:g}" for r in tests),ha='center',fontsize=7)
        f.tight_layout(rect=[0,.05,1,1]);f.savefig(fig/"paired_cpi.png",dpi=150);plt.close(f)
    from matplotlib.path import Path as PolygonPath
    checks=[];ticks=0;rows=0;outside=0;max_memory_mean_error=0.;stalled=[];steering=[]
    sys.path.insert(0,str(ROOT/'src'))
    from fly_sud.panel_b import memory_factory
    for p in out.glob("*/*/*/rep*/summary.json"):
        r=json.loads(p.read_text());h=pd.read_parquet(p.parent/"physics_history.parquet");s=pd.read_parquet(p.parent/"state_history.parquet")
        assert len(h)==r["physics_ticks"] and len(s)==r["control_rows"]
        assert np.isfinite(h[["x","y","z","heading_rad"]]).all().all()
        assert h.simulation_time_s.is_monotonic_increasing and h.state_index.between(0,len(s)-1).all()
        if r['choice']=='none' and len(h):
            tail=h[h.simulation_time_s>=max(0,float(h.simulation_time_s.iloc[-1])-10)]
            distance=float(np.linalg.norm(np.diff(tail[['x','y']].to_numpy(),axis=0),axis=1).sum())
            if distance<1.:stalled.append(dict(trial=str(p.parent.relative_to(out)),last_10s_path_mm=distance,final_z=float(h.z.iloc[-1])))
        assert (s.instantaneous_teaching_signal==0).all()
        if r["termination"]=="test_window_complete":
            cfg=meta["configs"][r["model"]];assert len(s)==round(cfg["test_duration_s"]/cfg["control_dt_s"])
        cfg=meta['configs'][r['model']];moving=s[s.physics_active]
        innate=cfg['innate_value']*np.array([moving.odor_a_left+moving.odor_b_left,moving.odor_a_right+moving.odor_b_right]).T
        values=moving[['value_left','value_right']].to_numpy()
        def replay_turn(v):
            total=innate+cfg['motor_value_gain']*v
            contrast=(total[:,0]-total[:,1])/(np.abs(total).sum(axis=1)+cfg['motor_denominator_floor'])
            odor_turn=np.clip(cfg['turn_gain']*contrast+moving.turn_noise.to_numpy(),-cfg['max_turn'],cfg['max_turn'])
            wall=moving.wall_turn.to_numpy() if 'wall_turn' in moving else 0
            wind=moving.wind_turn.to_numpy() if 'wind_turn' in moving else 0
            return np.clip(odor_turn+wall+wind,-cfg['max_turn'],cfg['max_turn'])
        actual=replay_turn(values);zeroed=replay_turn(np.zeros_like(values))
        np.testing.assert_allclose(actual,moving.turn.to_numpy(),atol=1e-12)
        np.testing.assert_allclose(moving.action_left,cfg['base_drive']*(1-actual),atol=1e-12)
        steering.append(dict(model=r['model'],condition=r['condition'],delay_h=r['delay_h'],replicate=r['replicate'],paired_odor=r['paired_odor'],b_left=r['b_left'],
            mean_abs_value_contribution_to_turn=float(np.mean(np.abs(actual-zeroed))),saturated_turn_fraction=float(np.mean(np.abs(actual)>=cfg['max_turn']-1e-12))))
        states=np.load(p.parent/"learning_states.npz")
        assert np.all(states["naive_aversive"]==0) and np.all(states["naive_long"]==0)
        replay=np.load(p.parent/'replay.npz')
        # Body-center containment, allowing 0.5 mm at contact surfaces.
        outside+=int((~PolygonPath(replay['arena_polygon']).contains_points(h[['x','y']].to_numpy(),radius=1.)).sum())
        # Independently reconstruct aggregate test memories from the archived pre-test state.
        cfg=meta['configs'][r['model']]
        for index in sorted(set([0,len(s)//2,len(s)-1])):
            mem=memory_factory(len(states['kc_root_ids']),cfg,r['model'])
            for key in ['aversive','short','long','intermediate']:
                if 'pre_test_'+key in states:setattr(mem,key,states['pre_test_'+key].copy())
            mem.ethanol=float(states['pre_test_ethanol']);elapsed=float(s.iloc[index].memory_elapsed_s)
            if elapsed:mem.advance(elapsed,np.zeros(len(states['kc_root_ids'])),learning=False)
            for key in ['aversive','short','long','intermediate']:
                if hasattr(mem,key):max_memory_mean_error=max(max_memory_mean_error,abs(float(getattr(mem,key).mean())-float(s.iloc[index][key+'_mean'])))
        checks.append(str(p.relative_to(out)));ticks+=len(h);rows+=len(s)
    with zipfile.ZipFile(out/"source_snapshot.zip") as z:
        for name,digest in meta["source_sha256"].items():assert hashlib.sha256(z.read(Path(name).as_posix())).hexdigest()==digest
    (out/"audit.json").write_text(json.dumps(dict(status="data_integrity_passed",trials=len(checks),physics_ticks=ticks,control_rows=rows,
        physics_failures=int(counts.physics_failures.sum()),source_archive_verified=True,
        outside_arena_physics_rows=outside,containment_passed=outside==0,max_reconstructed_memory_mean_error=max_memory_mean_error,
        memory_reconstruction_passed=max_memory_mean_error<1e-9,stalled_nonresponse_trials=stalled),indent=2))
    (out/'analysis_provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method='Exact sign-flip per replicate, Holm over all contrasts; no individual-fly pseudoreplication.'),indent=2))
    pd.DataFrame(steering).to_csv(out/'steering_replay_audit.csv',index=False)
    print(summary.to_string(index=False));print(json.dumps(stats,indent=2))
if __name__=="__main__":main()
