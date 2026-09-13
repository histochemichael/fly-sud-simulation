"""Verify replay reruns match originals through their recorded endpoint."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def verify(oldname,newname,n):
    base=ROOT/'results'/newname;report=[]
    for p in sorted(base.glob('*/summary.json')):
        orig=ROOT/'results'/oldname/p.parent.name
        a=json.loads((orig/'summary.json').read_text());b=json.loads(p.read_text())
        assert a['choice']==b['choice'] and a['latency_s']==b['latency_s'],p.parent.name
        assert a['random_seed']==b['random_seed']
        old=pd.read_parquet(orig/'state_history.parquet');new=pd.read_parquet(p.parent/'state_history.parquet')
        cols=[c for c in old.columns if c not in ['final_choice','choice_latency_s']]
        pd.testing.assert_frame_equal(old[cols],new.iloc[:len(old)][cols],check_exact=True)
        oldp=pd.read_parquet(orig/'physics_history.parquet');newp=pd.read_parquet(p.parent/'physics_history.parquet')
        pd.testing.assert_frame_equal(oldp,newp.iloc[:len(oldp)][oldp.columns],check_exact=True)
        la=np.load(orig/'learning_states.npz');lb=np.load(p.parent/'learning_states.npz')
        for key in la.files:assert np.array_equal(la[key],lb[key])
        report.append({'fly':p.parent.name,'prefix_exact':True,'choice_preserved':True,'additional_physics_rows':len(newp)-len(oldp),'end_s':b['observation_end_s'],'termination':b['termination_reason']})
    assert len(report)==n,(len(report),n)
    (base/'continuation_validation.json').write_text(json.dumps(report,indent=2));print(newname,len(report),'exact prefix matches; original choices unchanged')
if __name__=='__main__':
    import sys
    verify(sys.argv[1],sys.argv[2],int(sys.argv[3]))
