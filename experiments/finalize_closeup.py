"""Release audit for the local close-up revision; never uploads."""
from pathlib import Path
import json,hashlib,sys,platform
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';DEST=OUT/'closeup_revision'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reports=[]
for name,n in [('continuation_batch',400),('continuation_pilot',4)]:
    rows=json.loads((ROOT/'results'/name/'continuation_validation.json').read_text())
    assert len(rows)==n and all(r['prefix_exact'] and r['choice_preserved'] and r['end_s']==2 and r['termination']=='test_window_complete' for r in rows)
    reports.extend(rows)
media=json.loads((DEST/'media_manifest.json').read_text())
assert len(media)==9 and all(r['audio_streams']==0 and r['fully_decoded'] and sha(DEST/r['file'])==r['sha256'] for r in media)
sources=list((ROOT/'src').rglob('*.py'))+list((ROOT/'experiments').glob('*.py'))+list((ROOT/'config').glob('*.json'))
artifacts=list(DEST.glob('*.mp4'))+list(DEST.glob('*.png'))+[DEST/'README.md',DEST/'closeup_examples.zip',OUT/'representative_figures.pdf',OUT/'DATA_GUIDE.md']
for name in ['continuation_batch','continuation_pilot']:
    artifacts += [ROOT/'results'/name/p for p in ['metadata.json','config_snapshot.json','continuation_validation.json']]
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),
          'uploaded':False,'audio_streams':0,'exact_prefix_matches':len(reports),'observation_deadline_s':2,
          'additional_physics_rows':sum(r['additional_physics_rows'] for r in reports),'original_scores_unchanged':True,
          'interpretation':'Supplemental post-choice observation, not independent experimental replicates. No accelerated-time/free-ranging experiment.',
          'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
          'files':{str(p.relative_to(ROOT)):sha(p) for p in artifacts}}
(DEST/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Verified 404 exact continuations, nine silent videos and artifact hashes.')
