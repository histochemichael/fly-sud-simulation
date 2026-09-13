"""Cross-check displayed figures against immutable outcome data."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu,binomtest
ROOT=Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'output/individual_results/statistics.json').read_text())
p=ROOT/s['source'];assert hashlib.sha256(p.read_bytes()).hexdigest()==s['source_sha256']
d=pd.read_csv(p);assert len(d)==400
for row in s['summary']:
    g=d[d.condition==row['condition']];assert len(g)==100
    assert all(int((g.choice==k).sum())==row[k] for k in ['A','B','none'])
    assert int(g.latency_s.notna().sum())==row['n_latency']
old=json.loads((ROOT/'results/physical_batch/statistics.json').read_text())
assert s['tests'][0]['p']==old['paired_vs_untrained_responders_fisher_p']
assert s['tests'][1]['p']==old['paired_vs_unpaired_responders_fisher_p']
assert s['tests'][2]['p']==old['matched_dan_exact_p']
pvals=np.array([x['p'] for x in s['tests']]);order=np.argsort(pvals);prev=0
for rank,idx in enumerate(order):
    prev=max(prev,min(1,pvals[idx]*(6-rank)))
    assert np.isclose(s['tests'][idx]['p_holm_six_tests'],prev)
assert s['tests'][-1]['n1']==78
print('PASS: 400 actual choices, all plotted counts, original choice p-values, six-test Holm correction, 78 complete DAN pairs.')
