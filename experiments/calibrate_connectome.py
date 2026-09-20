"""Endpoint-only qualitative calibration; held-out delays are never scored."""
from pathlib import Path
import sys,json,itertools,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from fly_sud.connectome import Connectome,TemporalMemory,train_and_delay
def preference(graph,memory):
    v=[]
    for identity in range(2):
        odor=np.zeros((2,2));odor[identity]=1
        _,_,kc=graph.response(odor)
        v.append(float(graph.decode(kc,memory,np.ones(len(graph.ids["DAN"])))[0].mean()))
    return v[1]-v[0]
def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--paper",action="store_true");args=parser.parse_args()
    base=json.loads((ROOT/"config/connectome_time.json").read_text())
    if args.paper:base.update(protocol="paper_paired",center_motor_readout=True,master_seed=9301,training_interval_s=3000)
    out=ROOT/("data/connectome_v783/calibration_paper" if args.paper else "data/connectome_v783/calibration");out.mkdir(exist_ok=False)
    candidates=[];selected={}
    for model in base["models"]:
        for avtau,constau,avscale in itertools.product([1.,2.,4.],[3.,6.,12.],[1.,2.,4.]):
            if model=="decay_only" and constau!=6:continue
            cfg={**base,"aversive_tau_h":avtau,"consolidation_tau_h":constau,"aversive_scale":avscale,"motor_value_gain":50.}
            graph=Connectome(ROOT/"data/connectome_v783",cfg);values=[]
            for delay in base["calibration_delays_h"]:
                mem=TemporalMemory(len(graph.ids["KC"]),cfg,model)
                train_and_delay(graph,mem,"paired",1,delay)
                values.append(preference(graph,mem))
            # A design margin in model units, NOT a digitized paper effect size.
            loss=max(0.,values[0]+.001)**2+max(0.,.001-values[1])**2
            regularizer=(np.log2(avtau/2)**2+np.log2(constau/6)**2+np.log2(avscale)**2)
            candidates.append(dict(model=model,aversive_tau_h=avtau,consolidation_tau_h=constau,
                                   aversive_scale=avscale,early_value=values[0],late_value=values[1],
                                   loss=loss,regularizer=regularizer,feasible=loss==0))
        eligible=[r for r in candidates if r["model"]==model]
        best=min(eligible,key=lambda r:(r["loss"],r["regularizer"],r["aversive_tau_h"],r["consolidation_tau_h"]))
        selected[model]={**base,**{k:best[k] for k in ["aversive_tau_h","consolidation_tau_h","aversive_scale"]},"motor_value_gain":50.}
        print(model,best,flush=True)
    result={"method":"Qualitative sign calibration at 0.5 and 24 hours only; .001 is an arbitrary model-unit margin.",
            "paper_numeric_data_used":False,"intermediate_delays_used":False,
            "warning":"Endpoints underdetermine mechanisms; fitted endpoint signs are not validation. Generic odors, adapted schedule and decoder are not measured.",
            "candidates":candidates,"selected":selected}
    (out/"endpoint_calibration.json").write_text(json.dumps(result,indent=2))
    for model,cfg in selected.items():(out/f"{model}.json").write_text(json.dumps(cfg,indent=2))
    print("Feasible candidate counts",{m:sum(r["feasible"] for r in candidates if r["model"]==m) for m in selected})
if __name__=="__main__":main()
