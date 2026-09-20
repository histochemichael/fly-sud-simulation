"""Freeze qualitative endpoint calibration BEFORE physical behavioral evaluation."""
from pathlib import Path
import sys,json,itertools,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from fly_sud.connectome import Connectome,train_and_delay
from fly_sud.panel_b import FunctionalReadout,memory_factory
def main():
    out=ROOT/"data/connectome_v783/panel_b_v3";out.mkdir(exist_ok=False)
    base=json.loads((ROOT/"data/connectome_v783/calibration_paper/consolidating.json").read_text())
    base.update(master_seed=194901,test_duration_s=120.,adaptation_epsilon=.01,motor_value_gain=4.,turn_gain=120.,
        motor_denominator_floor=.05,stem_mm=30.,arm_mm=60.,corridor_width_mm=10.,collection_depth_mm=8.,odor_length_mm=40.,
        capture_policy="absorbing collection region; approximation to collection vials",readout="functional concentration-normalized population contrast",
        fitted_delays_h=[.5,24.],held_out_delays_h=[3.,6.,12.,15.,48.],control_dt_s=.02)
    g=Connectome(ROOT/"data/connectome_v783",base)
    candidates=[];selected={};curves=[]
    for model in ["consolidating","cascade"]:
        for tau,stage,avscale in itertools.product([2.,4.,8.],[6.,12.,24.],[.5,1.,2.,4.]):
            cfg={**base,"aversive_tau_h":tau,"consolidation_tau_h":stage,"aversive_scale":avscale}
            g.config=cfg;decoder=FunctionalReadout(g);mem=memory_factory(len(g.ids["KC"]),cfg,model)
            train_and_delay(g,mem,"paired",1,0)
            vals=[];previous=0
            for t in [.5,24.]:
                mem.advance((t-previous)*3600,np.zeros(len(g.ids["KC"])),learning=False);previous=t
                vals.append(decoder.probe(mem,1)-decoder.probe(mem,0))
            loss=max(0,vals[0]+.05)**2+max(0,.05-vals[1])**2
            regularizer=np.log2(tau/4)**2+np.log2(stage/12)**2+np.log2(avscale)**2
            candidates.append(dict(model=model,aversive_tau_h=tau,consolidation_tau_h=stage,aversive_scale=avscale,
                early_value=vals[0],late_value=vals[1],loss=loss,regularizer=regularizer))
        best=min([r for r in candidates if r["model"]==model],key=lambda r:(r["loss"],r["regularizer"],r["aversive_tau_h"],r["consolidation_tau_h"],r["aversive_scale"]))
        cfg={**base,**{k:best[k] for k in ["aversive_tau_h","consolidation_tau_h","aversive_scale"]}}
        selected[model]=cfg;(out/(model+".json")).write_text(json.dumps(cfg,indent=2));print(model,best,flush=True)
        g.config=cfg;decoder=FunctionalReadout(g)
        for condition in ["paired","unpaired","untrained","dan_silenced"]:
            m=memory_factory(len(g.ids["KC"]),cfg,model);train_and_delay(g,m,condition,1,0)
            for i,t in enumerate(np.arange(0,48.001,.25)):
                if i:m.advance(900,np.zeros(len(g.ids["KC"])),learning=False)
                curves.append(dict(model=model,condition=condition,delay_h=t,value_difference=decoder.probe(m,1,condition!="dan_silenced")-decoder.probe(m,0,condition!="dan_silenced")))
    pd.DataFrame(candidates).to_csv(out/"candidates.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"frozen_predictions.csv",index=False)
    (out/"calibration_policy.json").write_text(json.dumps(dict(
        targets="Only internal value signs with arbitrary +/-0.05 margin at .5 and 24 hours; not digitized paper CPI.",
        held_out="3,6,12,15,48 h are excluded from selection; inspecting these after freezing is evaluation, not blind to published findings.",
        readout="Functional hypothesis; not identified MBON-to-descending neuron wiring. No time-dependent action overrides.",
        model_selection="Both models evaluated; do not select by best behavioral p-value.",
        motor_settings="Engineering constants, no endpoint behavioral calibration. Validate with separate clamps.",
        candidates=len(candidates),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"src/fly_sud/panel_b.py"]},
        ),indent=2))
if __name__=="__main__":main()
