"""Fetch official immutable FlyWire v783 publication resources; no account needed."""
from pathlib import Path
import urllib.request, hashlib, json, time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/connectome_v783/raw"
COMMIT="ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0"
FILES=[
("annotations.tsv",f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{COMMIT}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",None),
("proofread_connections_783.feather","https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content","f48f972d262323a102aed49af1396b8a"),
("proofread_root_ids_783.npy","https://zenodo.org/api/records/10676866/files/proofread_root_ids_783.npy/content","e0e6c19732fd8c7a4e39a2d170105421")]
def digest(p,kind):
    with p.open("rb") as f:return hashlib.file_digest(f,kind).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True); records=[]
    for name,url,md5 in FILES:
        dest=OUT/name
        if not dest.exists():
            partial=dest.with_suffix(dest.suffix+".partial")
            with urllib.request.urlopen(url,timeout=120) as src,partial.open("wb") as dst:
                count=0;last=time.monotonic()
                while chunk:=src.read(1024*1024):
                    dst.write(chunk);count+=len(chunk)
                    if time.monotonic()-last>15:
                        print(name,round(count/1e6,1),"MB",flush=True);last=time.monotonic()
            if md5 and digest(partial,"md5")!=md5:raise ValueError("Checksum mismatch: "+name)
            partial.rename(dest)
        if md5 and digest(dest,"md5")!=md5:raise ValueError("Checksum mismatch: "+name)
        records.append(dict(file=name,url=url,md5=digest(dest,"md5"),sha256=digest(dest,"sha256"),bytes=dest.stat().st_size))
        print("Verified",name,flush=True)
    (OUT.parent/"source_manifest.json").write_text(json.dumps(dict(dataset="FlyWire FAFB v783",annotation_tag="v2.1.0",annotation_commit=COMMIT,connectivity_doi="10.5281/zenodo.10676866",connectivity_license="CC-BY-4.0",files=records),indent=2))
if __name__=="__main__":main()

