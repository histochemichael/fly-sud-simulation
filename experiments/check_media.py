"""Decode finished media, inspect sample frames, and save a manifest."""
from pathlib import Path
import json,hashlib,subprocess
import imageio.v2 as iio
import imageio_ffmpeg
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output'
paths=[OUT/'fly_ethanol_reward.mp4']+[OUT/f'{c}_{p}.mp4' for c in ['untrained','paired','unpaired','dan_silenced'] for p in ['training','test']]
manifest=[]
for path in paths:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
    reader=iio.get_reader(path);meta=reader.get_meta_data();reader.close()
    manifest.append({'file':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'duration_s':meta['duration'],'size':meta['size'],'fps':meta['fps'],'fully_decoded':True})
reader=iio.get_reader(paths[0]);sheet=Image.new('RGB',(960,810),'white')
for k,t in enumerate([4,18,35,63,83,105]):
    im=Image.fromarray(reader.get_data(t*25));im.thumbnail((480,270));sheet.paste(im,((k%2)*480,(k//2)*270))
reader.close();sheet.save(OUT/'video_quality_check.png')
(OUT/'media_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
