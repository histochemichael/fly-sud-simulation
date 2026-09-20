"""Actual saved outcomes; no synthetic individual points or paper effect sizes."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/(sys.argv[1] if len(sys.argv)>1 else "results/connectome_time_v2")
    d=pd.read_csv(out/"choices.csv");fig=out/"figures";fig.mkdir(exist_ok=True)
    summaries=[]
    for key,g in d.groupby(["model","condition","delay_h"]):
        responded=g[g.completed];n=len(responded);k=int(responded.paired_choice.sum())
        ci=binomtest(k,n).proportion_ci(method="wilson") if n else None
        summaries.append(dict(model=key[0],condition=key[1],delay_h=key[2],n_total=len(g),n_responded=n,
            paired_choices=k,other_choices=n-k,nonresponses=len(g)-n,
            p_paired_among_responders=k/n if n else None,ci_low=ci.low if ci else None,ci_high=ci.high if ci else None,
            preference_index=(2*k/n-1) if n else None,completion=n/len(g),neural_value=g.neural_paired_minus_unpaired_value.mean()))
    s=pd.DataFrame(summaries);s.to_csv(out/"summary.csv",index=False)
    tests=[]
    for model in d.model.unique():
        for h in sorted(d.delay_h.unique()):
            paired=d[(d.model==model)&(d.delay_h==h)&(d.condition=="paired")].set_index(["batch","individual"])
            for condition in ["unpaired","untrained","dan_silenced","dan_acquisition","dan_consolidation"]:
                control=d[(d.model==model)&(d.delay_h==h)&(d.condition==condition)].set_index(["batch","individual"])
                a=paired.paired_choice.astype(bool);b=control.paired_choice.astype(bool).reindex(a.index)
                wins=int((a&~b).sum());losses=int((~a&b).sum())
                pv=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.
                tests.append(dict(model=model,delay_h=h,comparison="paired vs "+condition,test="Exact paired McNemar, paired-odor choice vs all other outcomes",
                    n_pairs=len(a),discordant_pairs=wins+losses,wins=wins,losses=losses,p_raw=pv))
    order=np.argsort([t["p_raw"] for t in tests]);previous=0
    for rank,index in enumerate(order):
        previous=max(previous,min(1,(len(tests)-rank)*tests[index]["p_raw"]));tests[index]["p_holm"]=previous
    (out/"statistics.json").write_text(json.dumps({"family_size":len(tests),"tests":tests,
      "warning":"Small numerical pilot, one anatomical brain. Nonresponses count as not choosing paired odor for McNemar; completion separately reported. No biological inference or equivalence claims."},indent=2))
    colors={"paired":"#e7a43e","unpaired":"#1baaaa","untrained":"#8492a6","dan_silenced":"#bb55cc","dan_acquisition":"#466cce","dan_consolidation":"#da6672"}
    figobj,axes=plt.subplots(3,2,figsize=(13,12),sharex=True)
    for j,model in enumerate(["consolidating","decay_only"]):
        for cond,color in colors.items():
            g=s[(s.model==model)&(s.condition==cond)].sort_values("delay_h")
            axes[0,j].plot(g.delay_h,g.neural_value,"o-",color=color,label=cond)
            axes[1,j].plot(g.delay_h,g.p_paired_among_responders,"o-",color=color)
            axes[2,j].plot(g.delay_h,g.completion,"o-",color=color)
        axes[0,j].set_title(model);axes[0,j].axhline(0,color="k",lw=.7)
        axes[1,j].axhline(.5,color="k",ls=":",lw=.7);axes[1,j].set_ylim(-.05,1.05)
        axes[2,j].set_ylim(-.05,1.05)
        for ax in axes[:,j]:ax.set_xscale("log");ax.set_xticks([.5,3,6,12,24],labels=["0.5","3","6","12","24"]);ax.grid(alpha=.2)
        axes[2,j].set_xlabel("Hours after training")
    axes[0,0].set_ylabel("Internal paired-minus-unpaired value\n(model units, NOT preference index)")
    axes[1,0].set_ylabel("Paired-odor choice / responders\nMissing point = no responders")
    axes[2,0].set_ylabel("Responders / all tested")
    axes[0,0].legend(fontsize=8,ncol=2)
    figobj.suptitle("Connectome-based temporal pilot | 4 numerical flies per condition/delay/model\nEndpoint signs calibrated; intermediate delays are predictions, not experimental validation",fontsize=12)
    figobj.tight_layout(rect=[0,0,.99,.95]);figobj.savefig(fig/"time_dependence.png",dpi=160);plt.close(figobj)
    # All actual test paths at both endpoints, never truncated at first choice.
    figobj,axes=plt.subplots(2,6,figsize=(18,7),sharex=True,sharey=True)
    conditions=list(colors)
    for col,cond in enumerate(conditions):
        for row,delay in enumerate([.5,24]):
            ax=axes[row,col]
            for f in (out/"consolidating"/cond/f"{delay:g}h").glob("*/physics_history.parquet"):
                p=pd.read_parquet(f,columns=["x","y"]);ax.plot(p.y,p.x,lw=.7,alpha=.8)
            ax.scatter([-9,9],[19,19],s=30,marker="x",c="black");ax.axhline(12,color="gray",ls=":",lw=.7)
            ax.set_title(f"{cond}\n{delay:g} h");ax.set_aspect("equal");ax.set_xlabel("Lateral mm")
    axes[0,0].set_ylabel("Forward mm");axes[1,0].set_ylabel("Forward mm")
    figobj.suptitle("Complete recorded physical trajectories | consolidating model | odor identities counterbalanced")
    figobj.tight_layout();figobj.savefig(fig/"trajectories.png",dpi=140);plt.close(figobj)
    # Actual recorded latency dots + violins only where a density is estimable.
    figobj,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,model in zip(axes,["consolidating","decay_only"]):
        for k,delay in enumerate(sorted(d.delay_h.unique())):
            v=d[(d.model==model)&(d.condition=="paired")&(d.delay_h==delay)&d.completed].choice_latency_s.dropna().to_numpy()
            if len(v)>1 and np.ptp(v)>0:ax.violinplot(v,positions=[k],showextrema=False)
            ax.scatter(k+np.linspace(-.07,.07,len(v)),v,s=20,c="#cc8822")
            ax.text(k,2.08,f"n={len(v)}/4",ha="center",fontsize=8)
        ax.set_xticks(range(5),labels=[".5","3","6","12","24"]);ax.set_ylim(0,2.2);ax.set_title(model)
        ax.set_xlabel("Hours after training");ax.set_ylabel("Choice latency (s), responders only")
    figobj.suptitle("Actual paired-condition latencies | no imputed times for nonresponses")
    figobj.tight_layout();figobj.savefig(fig/"latency_violin_dots.png",dpi=160);plt.close(figobj)
    print(s.to_string(index=False));print("Smallest raw p",min(t["p_raw"] for t in tests))
if __name__=="__main__":main()

