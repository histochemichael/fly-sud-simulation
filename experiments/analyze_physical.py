"""Summaries retain nonresponses and use Wilson intervals for responders."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest,fisher_exact
ROOT=Path(__file__).resolve().parents[1]

def analyze():
    root=ROOT/"results/physical_batch"
    df=pd.read_csv(root/"choices.csv")
    rows=[]
    for c,g in df.groupby("condition",sort=False):
        a=int((g.choice=="A").sum());b=int((g.choice=="B").sum());n=len(g);m=a+b
        ci=binomtest(b,m).proportion_ci(method="wilson") if m else None
        values=np.where(g.choice=="B",1,np.where(g.choice=="A",-1,0))
        boots=np.random.default_rng(2042).choice(values,(10000,n)).mean(axis=1)
        rows.append(dict(condition=c,n=n,a=a,b=b,nonresponse=n-m,completed=m,
                         pi_responders=(b-a)/m if m else None,pi_all=(b-a)/n,
                         ci95_low=2*ci.low-1 if ci else None,ci95_high=2*ci.high-1 if ci else None,
                         pi_all_ci95_low=float(np.quantile(boots,.025)),pi_all_ci95_high=float(np.quantile(boots,.975)),
                         percent_b_responders=100*b/m if m else None,
                         percent_b_all=100*b/n,completion_rate=m/n,
                         median_latency_s=float(g.latency_s.median()),chance_p=binomtest(b,m).pvalue if m else None))
    summary=pd.DataFrame(rows);summary.to_csv(root/"summary.csv",index=False)
    ps={}
    for other in ["untrained","unpaired","dan_silenced"]:
        p=summary.set_index("condition").loc["paired"];q=summary.set_index("condition").loc[other]
        ps[f"paired_vs_{other}_responders_fisher_p"]=fisher_exact([[p.b,p.a],[q.b,q.a]]).pvalue
    p=df[df.condition=="paired"].set_index("fly_id");d=df[df.condition=="dan_silenced"].set_index("fly_id")
    pB=p.choice=="B";dB=d.choice=="B";up=int((pB & ~dB).sum());down=int((~pB & dB).sum())
    ps["matched_dan_discordant_B_loss"]=up;ps["matched_dan_discordant_B_gain"]=down
    ps["matched_dan_exact_p"]=binomtest(up,up+down).pvalue if up+down else 1.
    (root/"statistics.json").write_text(json.dumps(ps,indent=2))
    print(summary.to_string(index=False));print(ps)
    return summary,ps

if __name__=="__main__": analyze()
