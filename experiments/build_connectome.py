"""Build a neuron-resolved measured-connectivity feedforward substrate."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"data/connectome_v783"
def main():
    manifest=json.loads((D/"source_manifest.json").read_text())
    for rec in manifest["files"]:
        with (D/"raw"/rec["file"]).open("rb") as f:
            assert hashlib.file_digest(f,"sha256").hexdigest()==rec["sha256"]
    n=pd.read_csv(D/"raw/annotations.tsv",sep="\t",dtype=str).fillna("")
    classes={"olfactory":"ORN","ALPN":"PN","Kenyon_Cell":"KC","MBON":"MBON","DAN":"DAN"}
    n=n[n.cell_class.isin(classes)].copy()
    n["role"]=n.cell_class.map(classes);n["root_id"]=n.root_id.astype("int64")
    # IDs are never converted to floating point.
    n=n.sort_values("root_id").reset_index(drop=True)
    raw=pd.read_feather(D/"raw/proofread_connections_783.feather")
    print("Connectivity columns:",raw.columns.tolist(),flush=True)
    count=next(c for c in ["syn_count","synapse_count","weight","count"] if c in raw.columns)
    raw=raw.rename(columns={"pre_pt_root_id":"pre","post_pt_root_id":"post",count:"synapses"})
    roots=set(n.root_id)
    e=raw[raw.pre.isin(roots)&raw.post.isin(roots)].copy()
    pair=e.groupby(["pre","post"],as_index=False).synapses.sum()
    pair=pair[pair.synapses>=5].copy()
    roles=n.set_index("root_id").role
    pair["pre_role"]=pair.pre.map(roles);pair["post_role"]=pair.post.map(roles)
    used=[("ORN","PN"),("PN","KC"),("KC","MBON"),("DAN","KC"),("DAN","MBON")]
    n.to_parquet(D/"neurons.parquet",index=False)
    e.to_parquet(D/"all_subnetwork_edges_by_neuropil.parquet",index=False)
    pair["used_in_dynamics"]=[(a,b) in used and (a,b)!=("DAN","KC") for a,b in zip(pair.pre_role,pair.post_role)]
    pair.to_parquet(D/"edges.parquet",index=False)
    counts={}
    for a,b in used:
        pre=n[n.role==a].root_id.tolist();post=n[n.role==b].root_id.tolist()
        pi={r:i for i,r in enumerate(pre)};qi={r:i for i,r in enumerate(post)}
        s=pair[(pair.pre_role==a)&(pair.post_role==b)]
        m=sparse.csr_matrix((s.synapses.astype(float),(s.post.map(qi),s.pre.map(pi))),shape=(len(post),len(pre)))
        sparse.save_npz(D/f"{a}_{b}_counts.npz",m)
        counts[f"{a}->{b}"]={"pairs":len(s),"synapses":int(s.synapses.sum()),"connected_targets":int((m.sum(axis=1).A.ravel()>0).sum())}
    facts=dict(dataset=manifest["dataset"],annotation_count=139255,selected_neurons=len(n),roles=n.role.value_counts().to_dict(),minimum_aggregated_pair_synapses=5,selected_pairs=len(pair),used_pairs=int(pair.used_in_dynamics.sum()),blocks=counts,
        scope="Measured ORN->PN->KC->MBON and DAN->MBON feedforward blocks. DAN->KC exported but inactive in this version. Not whole-brain dynamics.",
        assumptions=["Synapse counts normalized within each selected input block, not measured conductance.",
                     "Generic odor tuning, rectified steady-state rates, KC sparsification, plasticity and motor decoding are assumed.",
                     "Other selected edges retained for auditing but not simulated; no recurrent or descending-neuron dynamics.",
                     "Publication-era annotation sides retained; current Codex corrected imagery handedness may differ."])
    (D/"graph_manifest.json").write_text(json.dumps(facts,indent=2))
    print(json.dumps(facts,indent=2))
if __name__=="__main__":main()
