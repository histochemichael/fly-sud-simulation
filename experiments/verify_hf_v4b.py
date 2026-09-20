"""Verify every released file against the public Hub metadata."""
from pathlib import Path
import hashlib,json
from huggingface_hub import HfApi
ROOT=Path(__file__).resolve().parents[1]
stage=ROOT/"publication/hf-fly-sud-v4b"
info=HfApi().dataset_info("Histochemichael/fly-sud-simulation",files_metadata=True,token=False)
remote={f.rfilename:f for f in info.siblings}
entries=json.loads((stage/"MANIFEST.json").read_text())["files"]
manifest=stage/"MANIFEST.json"
entries.append({"path":"MANIFEST.json","bytes":manifest.stat().st_size,
                "sha256":hashlib.sha256(manifest.read_bytes()).hexdigest()})
failures=[]
for e in entries:
    f=remote.get(e["path"])
    if f is None:
        failures.append([e["path"],"missing"])
        continue
    if f.size!=e["bytes"]:
        failures.append([e["path"],"size"])
    if f.lfs is not None:
        if f.lfs.sha256!=e["sha256"]:
            failures.append([e["path"],"LFS hash"])
    else:
        content=(stage/e["path"]).read_bytes()
        blob=hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
        if blob!=f.blob_id:
            failures.append([e["path"],"Git blob hash"])
receipt={"repository":info.id,"commit":info.sha,"public":not info.private,
         "expected_files":len(entries),"remote_files":len(remote),"failures":failures}
(ROOT/"publication/HF_PUBLICATION_VERIFICATION.json").write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
assert not info.private and not failures

