"""Frozen, non-overwriting physical panel-b experiment."""
from pathlib import Path
import sys,json,argparse,hashlib,zipfile,importlib.metadata,platform,time
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from fly_sud.panel_b_trial import run_panel_trial
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--workers",type=int,default=8)
    p.add_argument("--replicates",type=int,default=2);p.add_argument("--diagnostic",action="store_true")
    p.add_argument("--duration",type=float,default=120);p.add_argument("--models",nargs="+",default=["consolidating","cascade"])
    p.add_argument("--conditions",nargs="+",default=["paired","unpaired","untrained"])
    p.add_argument("--delays",type=float,nargs="+",default=[.5,24.]);a=p.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False)
    models=["consolidating"] if a.diagnostic else a.models
    configs={m:{**json.loads((ROOT/f"data/connectome_v783/panel_b_v3/{m}.json").read_text()),"test_duration_s":a.duration} for m in models}
    files=list((ROOT/"src").rglob("*.py"))+list((ROOT/"experiments").glob("*panel_b*.py"))
    metadata=dict(configs=configs,arguments=vars(a),platform=platform.platform(),processor=platform.processor(),
        navigation=dict(wall_range_mm=4.,wall_gain=6.,wall_max_turn=1.2,upwind_gain=1.2,wall_contact_bit=2,
            note='Assumed geometric obstacle and upwind reflexes; controller development uses clamps, not learned outcomes.'),
        packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        data_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/"data/connectome_v783").glob("*") if f.is_file()},
        caveats=["No quantitative CPI fitting. No deep learning.","One female anatomy; paper male cohorts.",
                 "Absorbing collection regions approximate collection vials; held intervals are not MuJoCo locomotion.",
                 "Two numerical flies per reciprocal odor group per replicate, not paper's 50.",
                 "Generic odors; functional readout not reconstructed motor circuitry; adapted unpaired ordering.",
                 "Development diagnostics are not panel-b results."])
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    with zipfile.ZipFile(out/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,f.relative_to(ROOT))
    conditions=["clamp_approach","clamp_avoid","clamp_neutral"] if a.diagnostic else a.conditions
    delays=[0.] if a.diagnostic else a.delays;tasks=[]
    for model in models:
        for condition in conditions:
            for delay in delays:
                for replicate in range(a.replicates):
                    for paired in [0,1]:
                        for b_left in [False,True]:
                            path=out/model/condition/f"{delay:g}h"/f"rep{replicate:02d}_odor{paired}_side{int(b_left)}"
                            tasks.append((configs[model],str(ROOT/"data/connectome_v783"),model,condition,delay,replicate,paired,b_left,str(path)))
    print(f"START {len(tasks)} trials, duration {a.duration}s; diagnostic={a.diagnostic}",flush=True)
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run_panel_trial,t) for t in tasks]):
            r=future.result();rows.append(r)
            print(f"{len(rows)}/{len(tasks)} {r['model']} {r['condition']} {r['delay_h']}h {r['choice']} capture={r['choice_latency_s']} {r['termination']} wall={time.monotonic()-start:.0f}s",flush=True)
    import pandas as pd
    pd.DataFrame(rows).sort_values(["model","condition","delay_h","replicate","paired_odor","b_left"]).to_csv(out/"choices.csv",index=False)
    print("COMPLETE",flush=True)
if __name__=="__main__":main()
