"""V4 individual-fly behavioral calibration followed by independent-seed validation."""
from pathlib import Path
import sys,json,argparse,hashlib,zipfile,platform,importlib.metadata,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from fly_sud.panel_b_trial import run_panel_trial

POLICY=dict(
 version="individual_behavior_v4",memory_model="consolidating; V3 kinetics unchanged",
 calibration_seed=410723,validation_seed=920617,
 candidate_gains=[5.,20.,80.],calibration_replicates=2,validation_replicates=8,
 flies_per_reciprocal_pair=4,delays_h=[.5,24.],
 calibration_selection="Minimize sum over delays of (abs(CPI)-0.35)^2 plus 4 for each wrong sign, 4*(nonresponse fraction), and 100*(physics failure fraction). Ties choose lower gain.",
 calibration_gate="Both signs correct; each absolute CPI between 0.125 and 0.75 inclusive; response >=80%; zero physics failures.",
 validation_success="Paired CPI negative at 0.5h and positive at 24h; each magnitude in [0.05,0.8]; both choices represented per paired cell; all untrained and unpaired |CPI|<=0.3; response >=80% in each cell; zero physics failures; paired-minus-unpaired sign correct and Holm-adjusted exact replicate sign-flip p<0.05 at both delays.",
 dan_analysis="Exploratory retrieval DAN-silencing contrast at 24h; not a calibrated criterion.",
 inference_unit="Reciprocal pair of odor-group PIs, not individual flies; 2 flies per odor group, smaller than paper.",
 limitations=["Not a quantitative paper reproduction or independently specified time-course validation.",
 "0.5h and 24h are calibration data points, NOT held-out times; validation only holds out random seeds.",
 "Variability distributions are assumed, not measured biological distributions.",
 "Real FlyWire subset; assumed functional readout and motor policy; no whole-brain expansion.",
 "DAN-to-MBON retrieval gate only; DAN-to-KC plasticity not implemented.",
 "Absorbing collection stops physics, logical test remains 120s; no fabricated post-collection trajectory.",
 "Online associative learning, not deep learning. No outcome labels injected into controller."])

def configs(gain,seed):
 c=json.loads((ROOT/"data/connectome_v783/panel_b_v3/consolidating.json").read_text())
 return {**c,"master_seed":seed,"individual_controller_v4":True,"behavioral_gain":gain,
         "motor_noise_sd":.18,"test_duration_s":120.}

def batch(out,cfg,conditions,reps,workers):
 out.mkdir(parents=True,exist_ok=False)
 (out/"config.json").write_text(json.dumps(cfg,indent=2))
 tasks=[]
 for condition in conditions:
  for delay in POLICY["delays_h"]:
   for rep in range(reps):
    for paired in [0,1]:
     for side in [False,True]:
      path=out/condition/f"{delay:g}h"/f"rep{rep:02d}_odor{paired}_side{int(side)}"
      tasks.append((cfg,str(ROOT/"data/connectome_v783"),"consolidating",condition,delay,rep,paired,side,str(path)))
 rows=[];start=time.monotonic()
 with ProcessPoolExecutor(max_workers=workers) as pool:
  for f in as_completed([pool.submit(run_panel_trial,t) for t in tasks]):
   r=f.result();rows.append(r)
   print(f"{out.name} {len(rows)}/{len(tasks)} {r['condition']} {r['delay_h']}h {r['choice']} {r['choice_latency_s']} elapsed={time.monotonic()-start:.0f}s",flush=True)
 df=pd.DataFrame(rows).sort_values(["condition","delay_h","replicate","paired_odor","b_left"])
 df.to_csv(out/"choices.csv",index=False)
 return df

def metrics(df):
 rows=[]
 for (condition,delay),g in df.groupby(["condition","delay_h"]):
  rows.append(dict(condition=condition,delay_h=float(delay),cpi=float(g.score.mean()),
       response=float(g.captured.mean()),failure=float((g.termination=="physics_failure").mean())))
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument("--output",default="results/individual_behavior_v4")
 p.add_argument("--workers",type=int,default=8);a=p.parse_args()
 out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False)
 files=list((ROOT/"src").rglob("*.py"))+list((ROOT/"experiments").glob("*individual_behavior*.py"))+list(ROOT.glob("INDIVIDUAL_BEHAVIOR*PROTOCOL.md"))+list((ROOT/"tests").glob("*individual_behavior*.py"))
 manifest=dict(policy=POLICY,platform=platform.platform(),processor=platform.processor(),
   packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
   source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
   data_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/"data/connectome_v783").glob("*") if f.is_file()})
 (out/"preregistered_policy.json").write_text(json.dumps(manifest,indent=2))
 with zipfile.ZipFile(out/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as z:
  for f in files:z.write(f,f.relative_to(ROOT))
 scores=[]
 for gain in POLICY["candidate_gains"]:
  df=batch(out/f"calibration_gain_{gain:g}",configs(gain,POLICY["calibration_seed"]),["paired"],POLICY["calibration_replicates"],a.workers)
  cells=metrics(df);loss=0;gate=True
  for r in cells:
   signed=r["cpi"]*(-1 if r["delay_h"]==.5 else 1)
   loss+=(abs(r["cpi"])-.35)**2+4*(signed<=0)+4*(1-r["response"])+100*r["failure"]
   gate &= signed>0 and .125<=abs(r["cpi"])<=.75 and r["response"]>=.8 and r["failure"]==0
  scores.append(dict(gain=gain,loss=loss,gate=bool(gate),cells=cells))
 (out/"calibration_selection.json").write_text(json.dumps(scores,indent=2))
 eligible=[r for r in scores if r["gate"]]
 if not eligible:
  (out/"STATUS.json").write_text(json.dumps(dict(status="calibration_gate_failed",note="No validation launched. Do not retune this completed calibration; declare a new revision."),indent=2))
  print("CALIBRATION GATE FAILED",flush=True);return
 chosen=min(eligible,key=lambda r:(r["loss"],r["gain"]))
 frozen=configs(chosen["gain"],POLICY["validation_seed"])
 (out/"frozen_validation_config.json").write_text(json.dumps(frozen,indent=2))
 print(f"FROZEN gain={chosen['gain']}; starting independent-seed validation",flush=True)
 batch(out/"validation",frozen,["paired","unpaired","untrained","dan_silenced"],POLICY["validation_replicates"],a.workers)
 (out/"STATUS.json").write_text(json.dumps(dict(status="validation_complete",note="Run independent analysis; success not assumed."),indent=2))
if __name__=="__main__":main()
