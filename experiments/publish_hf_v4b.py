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
    remote={f.rfilename:f for f in api.dataset_info("Histochemichael/fly-sud-simulation",files_metadata=True).siblings}
    def uploaded(e):
        f=remote.get(e["path"])
        return f is not None and f.size==e["bytes"] and (f.lfs is None or f.lfs.sha256==e["sha256"])
    paths=[e["path"] for e in manifest["files"] if e["path"].startswith("raw/") and not uploaded(e)]
    def upload_batch(start):
        batch=paths[start:start+28]
        print("Uploading raw files",start+1,"through",start+len(batch),"of",len(paths),flush=True)
        result=api.upload_folder(repo_id="Histochemichael/fly-sud-simulation",
            repo_type="dataset",folder_path=stage,allow_patterns=batch,
            commit_message="Preserve remaining raw simulation records batch "+str(start//28+1))
        print("Batch commit",result.oid,flush=True)
        return result.oid
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(upload_batch,range(0,len(paths),28)))
    print("Complete dataset commit",api.dataset_info("Histochemichael/fly-sud-simulation").sha,flush=True)
except Exception as exc:
    print("Upload failed:",type(exc).__name__,flush=True)
    sys.exit(1)
