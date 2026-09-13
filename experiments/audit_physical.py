"""Validate release-oriented invariants without changing experimental data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'results/physical_batch'
counts={'flies':0,'control_rows':0,'physics_rows':0}
for trial in sorted(base.glob('*_0*')):
    if not trial.is_dir(): continue
    s=pd.read_parquet(trial/'state_history.parquet')
    p=pd.read_parquet(trial/'physics_history.parquet')
    m=np.load(trial/'learning_states.npz')
    assert np.all(m['naive']==0)
    assert np.array_equal(m['learned'],m['final'])
    assert s.state_index.is_unique and p.state_index.isin(s.state_index).all()
    assert p.simulation_time_s.is_monotonic_increasing
    test=s[s.phase=='test']
    assert (test.teaching_signal==0).all()
    weights=test.filter(regex='^kc_weight_').to_numpy()
    assert np.allclose(weights,m['learned'])
    assert np.isfinite(p[['x','y','z','heading_rad']].to_numpy()).all()
    counts['flies']+=1;counts['control_rows']+=len(s);counts['physics_rows']+=len(p)
assert counts['flies']==400
counts['checks']='naive zero; test weights frozen; zero test reward; complete state joins; finite poses; monotonic times'
(base/'validation.json').write_text(json.dumps(counts,indent=2))
print(counts)
