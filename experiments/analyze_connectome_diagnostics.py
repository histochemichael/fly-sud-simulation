"""Analyze diagnostic interventions with seed-blocked, exact sign-flip tests."""
from pathlib import Path
import sys,json,itertools,hashlib,zipfile
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/sys.argv[1];d=pd.read_csv(out/"motor_choices.csv");fig=out/"figures";fig.mkdir(exist_ok=True)
    summaries=d.groupby("mode").agg(n=("seed","size"),unique_seeds=("seed","nunique"),mean_target_signed_y=("target_signed_y","mean"),
        target_choices=("choice",lambda x:(x=="target").sum()),other_choices=("choice",lambda x:(x=="other").sum()),no_choice=("choice",lambda x:(x=="none").sum()),
        failures=("termination",lambda x:(x=="physics_failure").sum()))
    summaries.to_csv(out/"motor_summary.csv")
    # Average the two counterbalanced placements within seed before inference.
    blocked=d[d["mode"].isin(["approach","neutral","avoid"])].groupby(["seed","mode"]).target_signed_y.mean().unstack()
    stats=[];rng=np.random.default_rng(92071)
    for a,b in [("approach","neutral"),("avoid","neutral"),("approach","avoid")]:
        delta=(blocked[a]-blocked[b]).to_numpy();observed=float(delta.mean())
        null=np.array([np.mean(delta*np.array(signs)) for signs in itertools.product([-1,1],repeat=len(delta))])
        boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(axis=1)
        stats.append(dict(comparison=a+" minus "+b,n_seed_blocks=len(delta),n_placements_per_seed=2,effect_mean_signed_y_mm=observed,
            bootstrap_seed_block_ci95_mm=np.quantile(boot,[.025,.975]).tolist(),
            test="two-sided exact paired sign-flip on seed-averaged target-signed final y; symmetric-null assumption",
            p_value=float(np.mean(np.abs(null)>=abs(observed)-1e-12))))
    order=np.argsort([x["p_value"] for x in stats]);running=0
    for rank,i in enumerate(order):
        running=max(running,min(1,(len(stats)-rank)*stats[i]["p_value"]));stats[i]["p_holm"]=running
    (out/"motor_statistics.json").write_text(json.dumps(stats,indent=2))
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    f,axs=plt.subplots(1,3,figsize=(14,4.5),sharex=True,sharey=True)
    for ax,mode in zip(axs,["approach","neutral","avoid"]):
        for r in d[d["mode"]==mode].itertuples():
            folder=out/"motor"/f"{r.seed}_{r.target_side}_{mode}"
            h=pd.read_parquet(folder/"physics_history.parquet")
            # Reflect target-right trials for comparison; show complete trajectories.
            ax.plot(h.x,h.y*r.target_side,alpha=.5,lw=.8,color="#217a9f")
            ax.scatter(h.x.iloc[-1],h.y.iloc[-1]*r.target_side,s=10,color="#e18b27")
        ax.scatter([19,19],[9,-9],s=60,c=["#e18b27","#888888"])
        ax.text(19.5,9,"B: clamp target",fontsize=8);ax.text(19.5,-9,"A: other",fontsize=8)
        ax.axvline(12,color="gray",ls=":",lw=.8)
        ax.set_title(mode.upper()+" | n=16 trials / 8 seeds");ax.set_xlabel("Forward position (mm)");ax.set_aspect("equal",adjustable="box");ax.margins(.2)
    axs[0].set_ylabel("Lateral position toward B (mm)")
    f.suptitle("Learning disabled: complete physical trajectories (target side aligned)")
    f.tight_layout();f.savefig(fig/"motor_trajectories.png",dpi=160);plt.close(f)
    f,ax=plt.subplots(figsize=(8,4.5))
    data=[blocked[m].to_numpy() for m in ["approach","neutral","avoid"]]
    ax.violinplot(data,showextrema=False)
    for j,values in enumerate(data,1):ax.scatter(np.full(len(values),j),values,color="#d48123",zorder=3)
    for values in blocked[["approach","neutral","avoid"]].to_numpy():ax.plot([1,2,3],values,color="gray",alpha=.25,lw=.7)
    ax.set_xticks([1,2,3],["Approach","Neutral","Avoid"]);ax.set_ylabel("Target-signed final y (mm)\nMean of two placements per seed")
    ax.set_title("Eight matched seed blocks; dots are seed averages")
    f.text(.5,.01,"; ".join(x["comparison"]+": Holm p="+format(x["p_holm"],".4g") for x in stats),ha="center",fontsize=8)
    f.tight_layout(rect=[0,.05,1,1]);f.savefig(fig/"motor_violin_dots.png",dpi=160);plt.close(f)
    p=pd.read_parquet(out/"perception_prechoice.parquet");sw=pd.read_csv(out/"perception_identity_swaps.csv")
    report=dict(prior_trials=p.trial.nunique(),prechoice_samples=len(p),min_active_kc=int(p.active_kc.min()),
        median_active_kc=float(p.active_kc.median()),max_learned_motor_signal=float(p.learned_motor_absmax.max()),
        max_abs_learned_turn_delta=float(p.learned_turn_delta.abs().max()),
        median_abs_learned_turn_delta=float(p.learned_turn_delta.abs().median()),
        motor_noise_stationary_sd=.12,median_swapped_identity_cosine=float(sw.original_swapped_cosine.median()),
        min_swapped_relative_difference=float(sw.relative_identity_difference.min()),swap_samples=len(sw),
        physical_trials=len(d),physics_failures=int((d.termination=="physics_failure").sum()))
    (out/"diagnostic_summary.json").write_text(json.dumps(report,indent=2))
    curve=pd.read_csv(out/"memory_timecourse.csv")
    f,ax=plt.subplots(figsize=(8,4.5))
    for model,c in curve.groupby("model"):ax.plot(c.delay_h,c.internal_value_difference,label=model)
    ax.axhline(0,color="gray",lw=.7);ax.axvspan(12,15,color="gray",alpha=.2,label="Paper behavior transition (not fitted)")
    ax.set(xlabel="Retention delay (hours)",ylabel="Fixed-probe internal value difference\nNot a behavioral preference index",title="Frozen memory models: timing fails held-out comparison")
    ax.legend(fontsize=8);f.tight_layout();f.savefig(fig/"memory_timing.png",dpi=160);plt.close(f)
    audit=[]
    for r in d.itertuples():
        folder=out/"motor"/f"{r.seed}_{r.target_side}_{r.mode}"
        h=pd.read_parquet(folder/"physics_history.parquet");s=pd.read_parquet(folder/"state_history.parquet");m=np.load(folder/"learning_states.npz")
        assert len(h)==r.physics_ticks and len(s)==r.control_rows
        assert np.isfinite(h[["x","y","z","heading_rad"]]).all().all()
        assert h.simulation_time_s.is_monotonic_increasing and h.state_index.between(0,len(s)-1).all()
        assert np.array_equal(m["initial"],m["final"]) and not m["learning_enabled"]
        assert (s.teaching_signal==0).all() and (s.ethanol_state==0).all()
        audit.append(dict(seed=int(r.seed),side=int(r.target_side),mode=r.mode,physics_ticks=len(h),state_rows=len(s)))
    (out/"audit.json").write_text(json.dumps(dict(status="passed",trials=len(audit),physics_ticks=sum(r["physics_ticks"] for r in audit),trials_detail=audit),indent=2))
    archived=json.loads((out/"metadata.json").read_text())["source_sha256"]
    with zipfile.ZipFile(out/"source_snapshot.zip") as snapshot:
        assert all(hashlib.sha256(snapshot.read(Path(name).as_posix())).hexdigest()==digest for name,digest in archived.items())
    provenance_files=[Path(__file__),ROOT/"tests/test_connectome_diagnostics.py",ROOT/"CONNECTOME_DIAGNOSTICS.md"]
    with zipfile.ZipFile(out/"analysis_snapshot.zip","w",zipfile.ZIP_DEFLATED) as snapshot:
        for path in provenance_files:snapshot.write(path,path.relative_to(ROOT))
    provenance=dict(source_archive_verified=True,analysis_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_files},
        prior_source_archive_sha256=hashlib.sha256((ROOT/"results/connectome_time_paper_v2/source_snapshot.zip").read_bytes()).hexdigest(),
        graph_manifest_sha256=hashlib.sha256((ROOT/"data/connectome_v783/graph_manifest.json").read_bytes()).hexdigest(),
        statistical_seed=92071,bootstrap_samples=10000,seed_blocks=8,tests=3)
    (out/"analysis_provenance.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps(report,indent=2));print(summaries.to_string());print(json.dumps(stats,indent=2))
if __name__=="__main__":main()
