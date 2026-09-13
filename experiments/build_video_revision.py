"""Silent, explicitly labelled footage with measured group/live trajectories."""
import json,sys,subprocess,zipfile,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
import imageio.v2 as iio
import imageio_ffmpeg
from build_film import ROOT,OUT,Clip,font,text,wrap,base,COLORS,NAMES,ACCENT,TEXT,MUTED,BG,PANEL
DEST=OUT/'video_revision';DEST.mkdir(exist_ok=True)
CONDITIONS=['untrained','paired','unpaired','dan_silenced']
atlas=Image.open(OUT/'assets/jfrc2_brain.png').convert('RGB');atlas.thumbnail((350,180))
GROUP='#7891A3';LIVE='#FFD34D'

class Subject(Clip):
    def __init__(self,c):
        super().__init__(c)
        self.group=[]
        for p in sorted((ROOT/'results/physical_batch').glob(c+'_*/state_history.parquet')):
            s=pd.read_parquet(p,columns=['phase','x','y','odor_b_side'])
            s=s[s.phase=='test']; sign=1 if s.odor_b_side.iloc[0]=='left' else -1
            self.group.append(np.column_stack((s.y.to_numpy()*sign,s.x.to_numpy())))
        assert len(self.group)==100
        s=self.states[self.states.phase=='test'];sign=1 if self.summary['odor_b_side']=='left' else -1
        self.path=np.column_stack((s.y.to_numpy()*sign,s.x.to_numpy()))
        self.times=s.phase_time_s.to_numpy()
        # Fixed cue-aligned view: B on the right, regardless of randomized source side.
        self.mapbase=Image.new('RGB',(224,278),BG);d=ImageDraw.Draw(self.mapbase)
        text(d,(8,4),'TEST PATHS | n=100',18,TEXT,True)
        text(d,(9,24),'mm: lateral +/-24; forward 0-30',12,MUTED)
        d.rectangle((10,35,214,223),outline='#596E80',width=1)
        for pts in self.group:d.line([self.mappt(*p) for p in pts],fill=GROUP,width=1)
        text(d,(10,229),'GRAY: ALL GROUP FLIES',15,GROUP,True)
        text(d,(10,249),'YELLOW: CURRENT FLY',15,LIVE,True)
        for a,b,color,label in [(-9,19,'#61C9C4','ODOR A'),(9,19,'#E8A1FF','ODOR B')]:
            u,v=self.mappt(a,b);d.ellipse((u-5,v-5,u+5,v+5),fill=color)
            text(d,(u-25,v-25),label,14,color,True)
    @staticmethod
    def mappt(lateral,forward):return (int(112+float(lateral)*4.1),int(215-float(forward)*5.9))
    def map(self,t,training):
        im=self.mapbase.copy();d=ImageDraw.Draw(im)
        if not training:
            pts=self.path[self.times<=t]
            if len(pts)>1:d.line([self.mappt(*p) for p in pts],fill=LIVE,width=3)
            if len(pts):
                u,v=self.mappt(*pts[-1]);d.ellipse((u-4,v-4,u+4,v+4),fill=LIVE)
        else:
            d.rectangle((12,118,212,165),fill=BG);text(d,(25,122),'TEST GROUP CONTEXT',14,TEXT,True);text(d,(26,143),'Live path starts in test',14,MUTED)
        return im

def panel(clips,phase,t,conditions=None):
    train=phase=='training';conds=conditions or CONDITIONS[:3]
    im=base('SILENT FILM | PLAYBACK 0.1x',
        'TRAINING: when does odor predict reward?' if train else 'TEST: ODORS ONLY - NO ETHANOL OR REWARD',
        'Reward is delivered uniformly during training, not at a maze location.' if train else 'Gray = all 100 physical test paths. Yellow = current fly. Cue-aligned map: Odor B always on right.')
    d=ImageDraw.Draw(im)
    for col,c in enumerate(conds):
        x=32+624*col;clip=clips[c];frame,row,ended=clip.get(phase,t)
        d.rounded_rectangle((x,200,x+608,1028),radius=14,fill=PANEL)
        text(d,(x+14,209),NAMES[c],31,COLORS[c],True)
        meaning={'untrained':'NO ODOR WAS REWARD-CONDITIONED','paired':'ODOR B WAS PAIRED WITH ETHANOL','unpaired':'ETHANOL SEPARATE FROM BOTH ODORS','dan_silenced':'ODOR B PAIRED; DAN OFF IN TEST'}[c]
        text(d,(x+14,251),meaning,23,TEXT,True)
        event=str(row.event).upper() if train else 'NO REWARD NOW - MEMORY-ONLY TEST'
        if train and row.teaching_signal>0:event='REWARD NOW: '+('ODOR B + ETHANOL' if c in ['paired','dan_silenced'] else 'ETHANOL WITHOUT ODOR')
        d.rounded_rectangle((x+8,291,x+600,341),radius=6,fill='#624513' if row.teaching_signal>0 else '#20374D')
        text(d,(x+18,300),event,23,ACCENT if row.teaching_signal else TEXT,True)
        text(d,(x+12,351),f"SESSION {int(row.conditioning_session)} | {t:.2f} s" if train else f'FLY 000 | TEST {min(t,float(clip.times[-1])):.2f} s',21,MUTED)
        im.paste(frame.resize((360,208)),(x+8,386))
        im.paste(clip.map(t,train),(x+376,352))
        text(d,(x+14,605),'REWARD '+('ON' if row.teaching_signal else 'OFF'),26,ACCENT if row.teaching_signal else MUTED,True)
        text(d,(x+14,640),f'Learned memory: A {row.weight_a:.3f} | B {row.weight_b:.3f}',23,TEXT,True)
        if ended and not train:
            choice=clip.summary['choice'];label='NO CHOICE / TIME LIMIT' if choice=='none' else 'CHOSE ODOR '+choice
            text(d,(x+14,675),label+' | final frame held',21,ACCENT,True)
        else:text(d,(x+14,675),'TRAINING MOVEMENT' if train else 'NAVIGATING WITH FIXED MEMORY',21,MUTED)
        text(d,(x+14,719),'ACTUAL ANATOMICAL REFERENCE',21,TEXT,True)
        im.paste(atlas,(x+10,752))
        text(d,(x+370,716),'SIMULATED RATES',19,ACCENT,True)
        vals=[('AL / PN',(row.al_left+row.al_right)/2),('KC',np.mean([row[f'kc_activity_{s}_{k:02d}'] for s in ['l','r'] for k in range(24)])),('MBON',(row.mbon_left+row.mbon_right)/2),('DAN',row.dan)]
        for j,(label,value) in enumerate(vals):
            yy=751+j*48;text(d,(x+370,yy),f'{label}  {value:.2f}',19,TEXT)
            d.rectangle((x+371,yy+25,x+584,yy+34),fill='#31495D');d.rectangle((x+371,yy+25,x+371+int(213*np.clip(value,0,1)),yy+34),fill=ACCENT)
        text(d,(x+12,944),'JFRC2 / VFB; Jenett et al. 2012; Ito et al. 2014',17,MUTED)
        text(d,(x+12,971),'Anatomy is static. Rates are NOT measured imaging.',19,TEXT,True)
        text(d,(x+12,997),'No anatomical registration or whole-brain simulation.',18,MUTED)
    return im

def historical_y():
    df=pd.read_csv(ROOT/'results/first_experiment_seed42/trajectories.csv')
    im=base('EARLIER PLANAR EXPERIMENT - NOT PHYSICAL BODY PATHS','Earlier Y-maze trajectories: group and example','These are the earlier planar results. They are NOT overlaid on the different physical-arena experiment.')
    d=ImageDraw.Draw(im)
    for i,c in enumerate(CONDITIONS[:3]):
        x=50+625*i;sub=df[df.condition==c]
        text(d,(x+15,222),NAMES[c],32,COLORS[c],True)
        def pt(a,b):return (x+275+a*155,760-b*210)
        d.line([pt(-1.15,1.25),pt(0,0),pt(1.15,1.25)],fill=TEXT,width=4);d.line([pt(0,-1.25),pt(0,0)],fill=TEXT,width=4)
        for fid,g in sub.groupby('fly_id'):
            d.line([pt(a,b) for a,b in zip(g.x,g.y)],fill=GROUP,width=1)
        g=sub[sub.fly_id==0];d.line([pt(a,b) for a,b in zip(g.x,g.y)],fill=LIVE,width=4)
        text(d,(x+20,280),'GRAY: ALL GROUP TRAJECTORIES',23,GROUP,True)
        text(d,(x+20,313),'YELLOW: PLANAR FLY 000',23,LIVE,True)
    return im

def title_card(title,lines):
    im=base('IMPULSE NEURO | SILENT EXPERIMENT FILM',title,'Read the phase labels, reward labels and path legend while the experiment plays.')
    d=ImageDraw.Draw(im)
    for i,line in enumerate(lines):wrap(d,(75,280+i*150),line,1740,38,TEXT)
    return im

def build(preview=False):
    clips={c:Subject(c) for c in CONDITIONS}
    stills={'training':panel(clips,'training',2.6),'test':panel(clips,'test',.65),'earlier_y':historical_y()}
    stills['intro']=title_card('How experience changes a simulated fly',["1. TRAIN: compare paired reward with untrained and unpaired controls.","2. TEST: reward is absent. Watch the yellow path against the gray group.","3. COMPARE: physical simulation results beside Kaun et al. (2011).",'Actual anatomical reference images; separate simulated rates. No audio.'])
    stills['legend']=title_card('Reward is tied to an odor, not a location',["PAIRED: Odor B and ethanol occur together. Odor A is not rewarded.","UNPAIRED: ethanol is delivered without odor; odors occur separately.","UNTRAINED: neither odor is reward-conditioned.","TEST: both odors are present, but ethanol and teaching reward are OFF."])
    stills['results']=Image.open(OUT/'figures/results.png').convert('RGB')
    stills['limits']=title_card('What the images and results mean',["The physical model learns an odor association, not a complete addiction mechanism.","Brain reference: JFRC2, Virtual Fly Brain. Jenett et al. (2012); Ito et al. (2014).",'CC BY 4.0; reference image resized only. Simulated rates are displayed separately.',"Gray paths: all 100 group members. Yellow: predetermined fly 000; no success selection."])
    for k,v in stills.items():v.save(DEST/f'{k}.png')
    if preview:return
    chapters=[('intro',10),('legend',16),('training',30),('test',20),('earlier_y',12),('results',20),('limits',16)]
    writer=iio.get_writer(DEST/'fly_ethanol_reward_revised.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-preset','fast'])
    for name,dur in chapters:
        print(name,flush=True)
        for f in range(dur*25):writer.append_data(np.asarray(panel(clips,name,f/250) if name in ['training','test'] else stills[name]))
    writer.close()
    for c in CONDITIONS:
        for phase,dur in [('training',30),('test',20)]:
            path=DEST/f'{c}_{phase}.mp4'
            if c!='dan_silenced':
                col=CONDITIONS.index(c);start=26 if phase=='training' else 56
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-ss',str(start),'-i',str(DEST/'fly_ethanol_reward_revised.mp4'),'-t',str(dur),'-vf',f'crop=608:828:{32+624*col}:200','-c:v','libx264','-crf','19','-preset','fast','-an',str(path)],check=True)
            else:
                w=iio.get_writer(path,fps=25,codec='libx264',quality=8,macro_block_size=1)
                for f in range(dur*25):w.append_data(np.asarray(panel(clips,phase,f/250,[c]).crop((32,200,640,1028))))
                w.close()
            print(c,phase,flush=True)
    (DEST/'chapters.json').write_text(json.dumps(chapters,indent=2))
    files=list(DEST.glob('*.mp4'));manifest=[]
    for p in files:
        check=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-i',str(p)],capture_output=True,text=True)
        assert 'Audio:' not in check.stderr
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(p),'-f','null','-'],check=True)
        manifest.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'audio_streams':0,'fully_decoded':True})
    (DEST/'media_manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(DEST/'video_examples_revised.zip','w',compression=zipfile.ZIP_STORED) as z:
        for p in files:
            if p.name!='fly_ethanol_reward_revised.mp4':z.write(p,p.name)
        z.write(DEST/'README.md','README.md')
    print('All revised videos validated: no audio',flush=True)

if __name__=='__main__':build('--preview' in sys.argv)
