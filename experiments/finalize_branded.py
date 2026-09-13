"""Verify branded media and preserve local release provenance."""
from pathlib import Path
import json,hashlib,platform,sys,subprocess
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';DEST=OUT/(sys.argv[1] if len(sys.argv)>1 else 'branded_revision')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((DEST/'media_manifest.json').read_text())
assert m['duration_s']==200 and len(m['files'])==9
assert sum(c['duration_s'] for c in m['chapters'][:4])==26
assert {'dan_training','dan_test','trajectories'} <= {c['chapter'] for c in m['chapters']}
assert all(x['audio_streams']==0 and x['fully_decoded'] and sha(DEST/x['file'])==x['sha256'] for x in m['files'])
hardware=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',
    "$cpu=Get-CimInstance Win32_Processor; $gpu=Get-CimInstance Win32_VideoController; $sys=Get-CimInstance Win32_ComputerSystem; @{cpu=$cpu.Name; cores=$cpu.NumberOfCores; logical_processors=$cpu.NumberOfLogicalProcessors; gpu=@($gpu.Name); ram_bytes=$sys.TotalPhysicalMemory} | ConvertTo-Json -Compress"],text=True))
files=list(DEST.glob('*.mp4'))+list(DEST.glob('*.png'))+[DEST/'README.md',DEST/('sleek_examples.zip' if DEST.name=='sleek_revision' else 'branded_examples.zip'),OUT/'representative_figures.pdf',OUT/'DATA_GUIDE.md']
if DEST.name=='sleek_revision':files += [OUT/'assets/impulse_neuro_logo_transparent.png',DEST/'logo_edit_prompt.md']
files+=list((OUT/'individual_results').glob('*.png'))+list((OUT/'individual_results').glob('*.json'))+list((OUT/'individual_results').glob('*.csv'))+list((OUT/'individual_results').glob('*.mp4'))
for name in ['continuation_batch','continuation_pilot']:
    files += [ROOT/'results'/name/p for p in ['metadata.json','config_snapshot.json','continuation_validation.json']]
sources=list((ROOT/'src').rglob('*.py'))+list((ROOT/'experiments').glob('*.py'))+list((ROOT/'config').glob('*.json'))
r={'created_utc':datetime.now(timezone.utc).isoformat(),'hardware_current_host':hardware,
   'python':sys.version,'platform':platform.platform(),'original_individual_tests':400,'filmed_representatives':4,
   'simulation_backend':'CPU MuJoCo; 8 workers in original batch','gpu_physics':False,'experiments_rerun_for_editorial_update':False,
   'body_segments':69,'actuated_joint_dofs':42,'gait_oscillators':6,'uploaded':False,
   'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
   'files':{str(p.relative_to(ROOT)):sha(p) for p in files}}
(DEST/'release_manifest.json').write_text(json.dumps(r,indent=2))
print('Verified: 3:20 film, 26-second intro, DAN sections, nine silent videos, current hardware and hashes.')
