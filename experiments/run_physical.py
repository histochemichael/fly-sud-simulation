import argparse,json,sys,time,os,hashlib,importlib.metadata
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from fly_sud.physical import run_physical_trial

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--n-flies",type=int,default=100);p.add_argument("--workers",type=int,default=6)
    p.add_argument("--output",default="results/physical_batch");p.add_argument("--representatives",action="store_true")
    p.add_argument("--perturbation",action="store_true")
    p.add_argument("--continue-after-choice",action="store_true",help="Record to the same test deadline while freezing the first-choice outcome")
    args=p.parse_args(); out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/"config/physical.json").read_text())
    cfg["n_flies"]=args.n_flies
    if args.continue_after_choice:cfg["continue_after_choice"]=True
    (out/"config_snapshot.json").write_text(json.dumps(cfg,indent=2))
    versions={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}
    source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/"src").rglob("*.py")}
    (out/"metadata.json").write_text(json.dumps(dict(backend="FlyGym 2.1 MuJoCo physics",packages=versions,source_sha256=source_hashes,master_seed=cfg["master_seed"],training="online rate circuit; stationary exposures except filmed representatives",published=False),indent=2))
    conditions=["untrained","paired","unpaired"]+(["dan_silenced"] if args.perturbation else [])
    tasks=[(cfg,c,i,str(out/f"{c}_{i:03d}"),args.representatives,args.representatives or i==0) for c in conditions for i in range(args.n_flies)]
    start=time.time(); rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_physical_trial,t) for t in tasks]
        for f in as_completed(futures):
            r=f.result();rows.append(r)
            print(f"{len(rows)}/{len(tasks)} {r['condition']} {r['fly_id']} {r['choice']} ({time.time()-start:.1f}s)",flush=True)
    import pandas as pd
    pd.DataFrame(rows).sort_values(["condition","fly_id"]).to_csv(out/"choices.csv",index=False)
    print(out,flush=True)
