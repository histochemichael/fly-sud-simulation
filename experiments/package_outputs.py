"""Package local clips and record immutable output checksums; never uploads."""
from pathlib import Path
import hashlib,json,sys,platform,zipfile
from datetime import datetime,timezone
from importlib.metadata import version,PackageNotFoundError
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output'
clips=[OUT/f'{c}_{p}.mp4' for c in ['untrained','paired','unpaired','dan_silenced'] for p in ['training','test']]
with zipfile.ZipFile(OUT/'video_examples.zip','w',compression=zipfile.ZIP_STORED) as z:
    for p in clips+[OUT/'DATA_GUIDE.md',OUT/'media_manifest.json']:z.write(p,p.name)
packages={}
for name in ['flygym','mujoco','numpy','pandas','pyarrow','scipy','matplotlib','Pillow','reportlab','imageio','imageio-ffmpeg']:
    try:packages[name]=version(name)
    except PackageNotFoundError:packages[name]='not in this interpreter; see batch metadata'
source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','experiments','config','tests'] for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.ps1']}
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'packages':packages,'source_sha256':source,'uploaded':False,'narration':'Windows SAPI Microsoft Zira Desktop, local synthetic voice','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'fly_ethanol_reward.mp4',OUT/'representative_figures.pdf',OUT/'video_examples.zip',OUT/'DATA_GUIDE.md']}}
(OUT/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
supp=ROOT/'results/physical_pilot/dan_silenced_000'
(supp/'config_snapshot.json').write_text((ROOT/'config/physical.json').read_text())
(supp/'metadata.json').write_text(json.dumps({'note':'Supplemental predetermined representative; not added to statistical batch or original pilot choices.csv','packages':packages,'source_sha256':source,'recorded_physical_training':True},indent=2))
print('Local video bundle and release manifest saved')
