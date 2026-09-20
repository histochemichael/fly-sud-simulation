"""Evaluate frozen models without fitting intermediate delays or motor outcomes."""
from pathlib import Path
import sys,json,copy
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from fly_sud.connectome import Connectome,TemporalMemory,train_and_delay
def value(g,m,identity):
    odor=np.zeros((2,2));odor[identity]=1
    _,_,kc=g.response(odor)
    return float(g.decode(kc,m,np.ones(len(g.ids["DAN"])))[0].mean())
def main():
    out=ROOT/(sys.argv[1] if len(sys.argv)>1 else "results/connectome_time_paper_v2")
    configs=json.loads((out/"config_snapshot.json").read_text());reports=[];curve=[]
    for model,cfg in configs.items():
        graph=Connectome(ROOT/"data/connectome_v783",cfg)
        memory=TemporalMemory(len(graph.ids["KC"]),cfg,model)
        train_and_delay(graph,memory,"paired",1,0)
        first_positive=None;last_t=0
        for t in np.arange(0,24.00001,.25):
            if t:memory.advance((t-last_t)*3600,np.zeros(len(graph.ids["KC"])),learning=False)
            preference=value(graph,memory,1)-value(graph,memory,0)
            if first_positive is None and preference>0:first_positive=float(t)
            curve.append(dict(model=model,delay_h=float(t),neural_value=preference))
            last_t=t
        # Numerical integration sensitivity, NOT a biological validation.
        v=[]
        for step in [60.,30.,10.]:
            cfg2={**cfg,"biological_step_s":step}
            g=Connectome(ROOT/"data/connectome_v783",cfg2)
            m=TemporalMemory(len(g.ids["KC"]),cfg2,model)
            train_and_delay(g,m,"paired",1,.5)
            v.append((step,value(g,m,1)-value(g,m,0)))
        reports.append(dict(model=model,first_positive_internal_value_h=first_positive,
           reported_paper_transition_h=[12,15],internal_timing_matches_paper=first_positive is not None and 12<=first_positive<=15,
           caution="Internal value crossing is not itself a behavioral transition.",
           early_value_step_sensitivity=[dict(dt_s=dt,value=x) for dt,x in v],
           relative_60_vs_10_error=abs(v[0][1]-v[-1][1])/max(abs(v[-1][1]),1e-12)))
    d=pd.read_csv(out/"choices.csv");summ=pd.read_csv(out/"summary.csv")
    for report in reports:
        s=summ[(summ.model==report["model"])&(summ.condition=="paired")&(summ.delay_h.isin([.5,24]))]
        report["physical_endpoints"]=s.to_dict("records")
        report["endpoint_fit_is_validation"]=False
        report["dan_phase_effect_is_independent_validation"]=False
    record=dict(status="Exploratory validation, not a successful biological replication",
       intermediate_data_policy="0.5/24h signs calibrated; 12–15h source-text timing not fitted.",
       qualitative_source="https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/",
       paper_raw_individual_data_available=False,
       caveats=["Only four numerical flies per condition/delay/model.",
                "One female anatomy, not the paper's male cohorts.",
                "Physical arena test is 2 s, paper test is 2 min.",
                "No measured chemical odor tuning or validated MBON-to-motor mapping.",
                "No quantitative paper effect-size or statistical replication claim.",
                "DAN acquisition/consolidation invariance is imposed by the default rule."],
       reports=reports)
    pd.DataFrame(curve).to_csv(out/"frozen_neural_timecourse.csv",index=False)
    (out/"validation_report.json").write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
if __name__=="__main__":main()

