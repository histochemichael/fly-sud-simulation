"""Publish scoped HF artifacts in resumable batches; never prints credentials."""
from pathlib import Path
from huggingface_hub import HfApi
import sys
ROOT=Path(__file__).resolve().parents[1]
stage=ROOT/"publication/hf-fly-sud-v4b"
api=HfApi()
try:
    result=api.upload_folder(repo_id="Histochemichael/fly-sud-simulation",
        repo_type="dataset",folder_path=stage,
        allow_patterns=["README.md","MANIFEST.json","results/**","tables/**","protocols/**"],
        commit_message="Publish V4b results, comparison, protocols and individual choices")
    print("Results commit",result.oid,flush=True)
    import json
    manifest=json.loads((stage/"MANIFEST.json").read_text())
    paths=[e["path"] for e in manifest["files"] if e["path"].startswith("raw/")]
    for start in range(0,len(paths),70):
        batch=paths[start:start+70]
        print("Uploading raw files",start+1,"through",start+len(batch),"of",len(paths),flush=True)
        result=api.upload_folder(repo_id="Histochemichael/fly-sud-simulation",
            repo_type="dataset",folder_path=stage,allow_patterns=batch,
            commit_message="Preserve raw simulation records batch "+str(start//70+1))
        print("Batch commit",result.oid,flush=True)
    print("Complete dataset commit",result.oid,flush=True)
except Exception as exc:
    print("Upload failed:",type(exc).__name__,flush=True)
    sys.exit(1)
