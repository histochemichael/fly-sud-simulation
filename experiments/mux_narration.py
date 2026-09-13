from pathlib import Path
import wave,json,subprocess
import numpy as np
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1];out=ROOT/"output"
sections=json.loads((out/"narration.json").read_text())
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
arrays=[];rate=22050
for s in sections:
    path=out/f"narration_{s['chapter']}.wav"
    with wave.open(str(path)) as w:
        data=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16);sr=w.getframerate();channels=w.getnchannels()
        assert channels==1
    duration=len(data)/sr;print(s["chapter"],round(duration,2),s["duration"])
    # Fit overlong narration once using speech tempo, then pad to chapter duration.
    target=s["duration"]-1
    tempo=max(1.,duration/target)
    fixed=out/f"narration_{s['chapter']}_fit.wav"
    subprocess.run([ffmpeg,"-y","-i",str(path),"-af",f"atempo={tempo},apad","-t",str(s["duration"]),"-ar",str(rate),"-ac","1",str(fixed)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    with wave.open(str(fixed)) as w: arrays.append(np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16))
with wave.open(str(out/"narration.wav"),"wb") as w:
    w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(np.concatenate(arrays).tobytes())
subprocess.run([ffmpeg,"-y","-i",str(out/"fly_ethanol_reward_silent.mp4"),"-i",str(out/"narration.wav"),"-c:v","copy","-c:a","aac","-b:a","128k","-movflags","+faststart","-shortest",str(out/"fly_ethanol_reward.mp4")],check=True)
