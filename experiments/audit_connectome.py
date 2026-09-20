"""Audit all saved histories, source hashes and matched-cohort design."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/(sys.argv[1] if len(sys.argv)>1 else "results/connectome_time_v2")
    rows=pd.read_csv(out/"choices.csv");meta=json.loads((out/"metadata.json").read_text())
    assert len(rows)==240 and not rows.duplicated(["model","condition","delay_h","batch","individual"]).any()
    # Noise seeds are condition/model-matched and distinct between delay cohorts.
    assert (rows.groupby(["delay_h","batch","individual"]).seed.nunique()==1).all()
    assert rows[["delay_h","batch","individual","seed"]].drop_duplicates().seed.nunique()==20
    assert (rows.groupby(["model","condition","delay_h"]).paired_odor.nunique()==2).all()
    checks=[];total_state=0;total_physics=0
    for f in out.rglob("summary.json"):
        s=json.loads(f.read_text());d=pd.read_parquet(f.parent/"state_history.parquet")
        p=pd.read_parquet(f.parent/"physics_history.parquet");z=np.load(f.parent/"learning_states.npz")
        assert len(d)==s["control_rows"] and len(p)==s["physics_ticks"]
        assert p.state_index.between(0,len(d)-1).all()
        assert (np.diff(d.biological_time_s)>=-1e-8).all() and (np.diff(p.biological_time_s)>0).all()
        assert np.isfinite(p[["x","y","z","heading_rad"]].to_numpy()).all()
        for column,size in {"orn_rates":2281,"pn_rates":685,"kc_rates":5177,"dan_rates":331,"mbon_appetitive":96,"mbon_aversive":96,"aversive_memory":5177,"short_memory":5177,"long_memory":5177}.items():
            array=np.stack(d[column].to_numpy())
            assert array.shape==(len(d),size) and np.isfinite(array).all()
        assert np.isclose(z["pre_test_biological_time_s"]-z["post_training_biological_time_s"],s["delay_h"]*3600)
        for k in ["aversive","short","long"]:
            assert not z["naive_"+k].any()
            assert np.isfinite(z["final_"+k]).all() and (z["final_"+k]>=-1e-12).all()
        if s["condition"]=="untrained":assert not z["final_aversive"].any() and not z["final_long"].any()
        test=d[d.phase=="test"];assert not test.instantaneous_teaching_signal.any()
        if s["condition"]=="dan_silenced":assert all(not a.any() for a in test.dan_rates)
        if s["termination_reason"]!="physics_failure":assert np.isclose(p.test_time_s.iloc[-1],2.)
        if s["choice_latency_s"] is not None:assert p.test_time_s.iloc[-1]>=s["choice_latency_s"]
        checks.append({"trial":str(f.parent.relative_to(out)),"passed":True})
        total_state+=len(d);total_physics+=len(p)
    assert len(checks)==len(rows)
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in meta["dataset_sha256"].items()),"Connectome changed after run initialization"
    hashes={key:hashlib.sha256((ROOT/key).read_bytes()).hexdigest()==value for key,value in meta["source_sha256"].items()}
    if not all(hashes.values()):
        import zipfile
        with zipfile.ZipFile(out/"source_snapshot.zip") as snapshot:
            assert all(hashlib.sha256(snapshot.read(Path(name).as_posix())).hexdigest()==digest for name,digest in meta["source_sha256"].items())
    result=dict(trials=len(checks),state_rows=total_state,physics_rows=total_physics,source_hashes_verified=True,current_source_matches=all(hashes.values()),
                failures=int((rows.termination_reason=="physics_failure").sum()),all_checks_passed=True,checks=checks)
    (out/"audit.json").write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!="checks"})
if __name__=="__main__":main()
