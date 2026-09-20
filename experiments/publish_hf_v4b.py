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
    result=api.upload_folder(repo_id="Histochemichael/fly-sud-simulation",
        repo_type="dataset",folder_path=stage,
        commit_message="Complete V4 failed calibration and V4b raw trial records")
    print("Complete dataset commit",result.oid,flush=True)
except Exception as exc:
    print("Upload failed:",type(exc).__name__,flush=True)
    sys.exit(1)

