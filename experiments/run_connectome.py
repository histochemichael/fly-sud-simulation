"""Run a frozen-config cohort; refuse to overwrite existing results."""
from pathlib import Path
import sys,json,argparse,hashlib,importlib.metadata,time,os
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from fly_sud.connectome_trial import run_trial
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True)
    p.add_argument("--batches",type=int,default=2);p.add_argument("--per-batch",type=int,default=2)
    p.add_argument("--workers",type=int,default=4);p.add_argument("--physical",action="store_true")
    p.add_argument("--models",nargs="+",default=["consolidating","decay_only"])
    p.add_argument("--calibration",default="data/connectome_v783/calibration")
    p.add_argument("--delays",type=float,nargs="+");p.add_argument("--conditions",nargs="+")
    a=p.parse_args();out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False)
    configs={m:json.loads((ROOT/a.calibration/f"{m}.json").read_text()) for m in a.models}
    (out/"config_snapshot.json").write_text(json.dumps(configs,indent=2))
    meta=dict(physical=a.physical,batches=a.batches,per_batch=a.per_batch,workers=a.workers,platform=sys.platform,cpu_count=os.cpu_count(),
        packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/"src").rglob("*.py"))+list((ROOT/"experiments").glob("*connectome*.py"))},
        dataset_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"data/connectome_v783").glob("*") if p.is_file()},
        warnings=["One reconstructed female brain shared by all numerical flies, not independent biological brains.",
                  "Conditions seed-matched within delay; delays use independent numerical seeds.",
                  "Neural value is not preference index. Two-second physical test is not the paper's duration.",
                  "Training and retention are stationary internal-state simulations, not motor trajectories.",
                  "No Hugging Face or GitHub publication by this runner."])
    (out/"metadata.json").write_text(json.dumps(meta,indent=2));tasks=[]
    import zipfile
    with zipfile.ZipFile(out/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as snapshot:
        for name in meta["source_sha256"]:snapshot.write(ROOT/name,name)
    for model,cfg in configs.items():
        for c in a.conditions or cfg["conditions"]:
            for h in a.delays or cfg["delays_h"]:
                for b in range(a.batches):
                    for i in range(a.per_batch):
                        path=out/model/c/f"{h:g}h"/f"batch{b:02d}_fly{i:03d}"
                        tasks.append((cfg,str(ROOT/"data/connectome_v783"),model,c,h,b,i,str(path),a.physical))
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(run_trial,t) for t in tasks]):
            r=f.result();rows.append(r)
            print(f"{len(rows)}/{len(tasks)} {r['model']} {r['condition']} {r['delay_h']}h {r['choice']} ({time.monotonic()-start:.1f}s)",flush=True)
    import pandas as pd
    pd.DataFrame(rows).sort_values(["model","condition","delay_h","batch","individual"]).to_csv(out/"choices.csv",index=False)
    print("COMPLETE",out,flush=True)
if __name__=="__main__":main()
