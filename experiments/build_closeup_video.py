"""Large fly close-up and a corner map, including real post-choice observations."""
import sys,json,subprocess,zipfile,hashlib
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
import imageio.v2 as iio
import imageio_ffmpeg
from build_film import ROOT,OUT,base,text,wrap,ACCENT,TEXT,MUTED,BG,PANEL,NAMES,COLORS
from build_overhead_video import circuit_overlay,arrow
from build_video_revision import title_card,CONDITIONS
DEST=OUT/'closeup_revision';DEST.mkdir(exist_ok=True)
GRAY='#8497A6';YELLOW='#FFD34D'

class FlyClip:
    def __init__(self,c,preview=False):
        self.condition=c;p=ROOT/'results/continuation_pilot'/f'{c}_000'
        self.states=pd.read_parquet(p/'state_history.parquet');self.summary=json.loads((p/'summary.json').read_text());self.rep=np.load(p/'replay.npz')
        original=json.loads((ROOT/'results/physical_pilot'/f'{c}_000'/'summary.json').read_text())
        self.choice_pos=(original['last_x_mm'],original['last_y_mm'])
        physics=pd.read_parquet(p/'physics_history.parquet');physics['phase']=self.states.phase.to_numpy()[physics.state_index]
        test_start=float(self.states[self.states.phase=='test'].simulation_time_s.iloc[0])
        self.frames=[Image.open(OUT/'closeup_frames'/c/f'{i:04d}.png').convert('RGB') for i in range(len(self.rep['state_index']))]
        self.phase={};self.paths={}
        for phase in ['training','test']:
            ids=[i for i,s in enumerate(self.rep['state_index']) if self.states.iloc[s].phase==phase]
            rows=[self.states.iloc[self.rep['state_index'][i]] for i in ids]
            times=np.array([r.simulation_time_s if phase=='training' else r.phase_time_s for r in rows])
            self.phase[phase]=(ids,rows,times)
            s=physics[physics.phase==phase];s=s.iloc[np.unique(np.r_[np.arange(0,len(s),100),len(s)-1])]
            self.paths[phase]=(s[['x','y']].to_numpy(),s.simulation_time_s.to_numpy()-(test_start if phase=='test' else 0))
        self.side=1 if self.summary['odor_b_side']=='left' else -1;self.group=[]
        for file in sorted((ROOT/'results/continuation_batch').glob(c+'_*/state_history.parquet')):
            states=pd.read_parquet(file,columns=['phase','odor_b_side']);side=1 if states.odor_b_side.iloc[0]=='left' else -1
            s=pd.read_parquet(file.parent/'physics_history.parquet');s=s[states.phase.to_numpy()[s.state_index]=='test'];s=s.iloc[np.unique(np.r_[np.arange(0,len(s),100),len(s)-1])]
            self.group.append(np.column_stack((s.x,s.y*side*self.side)))
        if not preview:assert len(self.group)==100
        full=np.concatenate(self.group+[p[0] for p in self.paths.values()]);self.xmin=min(-5,float(full[:,0].min())-2);self.xmax=max(44,float(full[:,0].max())+2)
        self.ymax=max(23,float(np.abs(full[:,1]).max())+2)
        self.background=self.draw_map_background(224,242)
    def get(self,phase,t):
        ids,rows,times=self.phase[phase];i=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(ids)-1));return self.frames[ids[i]].copy(),rows[i]
    def point(self,p,w=224,h=242):
        scale=min((w-24)/(2*self.ymax),(h-65)/(self.xmax-self.xmin))
        return (w/2-float(p[1])*scale,h-32-(float(p[0])-self.xmin)*scale)
    def draw_map_background(self,w,h):
        im=Image.new('RGB',(w,h),'#0E1B2B');d=ImageDraw.Draw(im)
        text(d,(8,4),'FULL TEST PATHS | 2 s',16,TEXT,True)
        for side in [-1,1]:
            is_b=side==self.side;color='#423A27' if is_b and self.condition in ['paired','dan_silenced'] else '#1D3740'
            polygon=[self.point(p,w,h) for p in [(12,2.5*side),(26,2.5*side),(26,17*side),(12,17*side)]]
            d.polygon(polygon,fill=color)
        for p in self.group:
            mapped=[self.point(q,w,h) for q in p];d.line(mapped,fill=GRAY,width=1)
            assert all(0<a<w and 22<b<h-20 for a,b in mapped),'A path falls outside the map'
        for side in [-1,1]:
            b=side==self.side;u,v=self.point((19,9*side),w,h)
            d.ellipse((u-4,v-4,u+4,v+4),fill=ACCENT if b else '#66D3CF')
            text(d,(u-5,v-20),'B' if b else 'A',15,TEXT,True)
        text(d,(8,h-22),f'GRAY: GROUP ({len(self.group)})',14,GRAY,True)
        return im
    def minimap(self,phase,t):
        im=self.background.copy();d=ImageDraw.Draw(im);path,times=self.paths[phase];p=path[times<=t]
        if len(p)>1:d.line([self.point(q) for q in p],fill=YELLOW,width=2)
        if len(p):
            q=self.point(p[-1]);d.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),fill=YELLOW)
        if len(p)>4:
            q=np.array(self.point(p[-1]));v=q-np.array(self.point(p[-5]));v=v/(np.linalg.norm(v)+1e-9)
            arrow(d,q-v*12,q,YELLOW,2)
        if phase=='test' and self.summary['latency_s'] is not None and t>=self.summary['latency_s']:
            u,v=self.point(self.choice_pos);d.ellipse((u-4,v-4,u+4,v+4),outline='white',width=1)
        return im

def panel(clips,phase,t,conditions=None):
    train=phase=='training';im=base('SILENT | CLOSE OVERHEAD + FULL-PATH CORNER MAP',
            'TRAINING: close-up of the fly and its learning circuit' if train else 'TEST: choice is marked, but the recording continues',
            'Corner map: gray full test group; yellow current training path.' if train else 'All test paths run to the same 2-second deadline unless physics fails. Original first-choice scores are unchanged.')
    d=ImageDraw.Draw(im)
    for col,c in enumerate(conditions or CONDITIONS[:3]):
        x=32+624*col;clip=clips[c];frame,row=clip.get(phase,t)
        d.rounded_rectangle((x,200,x+608,1028),radius=14,fill=PANEL)
        text(d,(x+14,208),NAMES[c],30,COLORS[c],True)
        label={'untrained':'Neither odor reward-conditioned','paired':'ODOR B: previously paired with ethanol','unpaired':'Ethanol separate from both odors','dan_silenced':'B paired; DAN off only during test'}[c]
        text(d,(x+14,247),label,24,TEXT,True)
        event=str(row.event).upper() if train else 'TEST: NO ETHANOL / NO TEACHING REWARD'
        if row.teaching_signal>0:event='REWARD NOW: '+('ODOR B + ETHANOL' if c in ['paired','dan_silenced'] else 'ETHANOL ALONE')
        d.rounded_rectangle((x+8,284,x+600,330),radius=5,fill='#684910' if row.teaching_signal else '#20374D')
        text(d,(x+17,292),event,23,ACCENT if row.teaching_signal else TEXT,True)
        when=f'Session {int(row.conditioning_session)} | {t:.2f} s' if train else f'Test {t:.2f} / 2.00 s'
        text(d,(x+12,338),when+f' | Memory A {row.weight_a:.2f}, B {row.weight_b:.2f}',21,TEXT)
        im.paste(frame,(x,365));im.paste(clip.minimap(phase,t),(x+380,370))
        text(d,(x+388,615),'YELLOW: THIS FLY',17,YELLOW,True)
        text(d,(x+388,639),'WHITE RING: FIRST CHOICE',14,TEXT)
        text(d,(x+388,661),'Zones mark choice, not reward.',14,MUTED)
        text(d,(x+12,373),'CLOSE OVERHEAD',18,TEXT,True)
        latency=clip.summary['latency_s']
        status='Uniform training exposure; map markers are test options.' if train else 'Before first choice - no reward is present.'
        if not train and latency is not None and t>=latency:status=f"POST-CHOICE OBSERVATION | {clip.summary['choice']} recorded at {latency:.3f} s"
        if not train and t>=1.96 and latency is None:status='NO CHOICE by 2 s | complete observation shown'
        text(d,(x+12,718),status,19,ACCENT if not train and latency is not None and t>=latency else TEXT,True)
        text(d,(x+12,748),'ILLUSTRATIVE CIRCUITS ON REFERENCE ANATOMY',20,TEXT,True)
        circuit_overlay(im,x+10,778,row,t)
        text(d,(x+370,778),'SIMULATED ACTIVITY',18,ACCENT,True)
        for j,(label,value) in enumerate([('PN',(row.pn_left+row.pn_right)/2),('KC',(row.kc_left+row.kc_right)/2),('MBON',(row.mbon_left+row.mbon_right)/2),('DAN',row.dan)]):
            yy=805+36*j;text(d,(x+370,yy),f'{label}: {value:.2f}',17,TEXT)
            d.rectangle((x+370,yy+22,x+586,yy+28),fill='#31495D');d.rectangle((x+370,yy+22,x+370+int(216*np.clip(value,0,1)),yy+28),fill='#FF5849')
        text(d,(x+12,957),'Bright paths use recorded activity and learned weights.',18,TEXT,True)
        text(d,(x+12,982),'Illustrative wiring, not anatomically registered.',18,MUTED)
        text(d,(x+12,1005),'JFRC2 / VFB; Jenett 2012; Ito 2014; CC BY 4.0',16,MUTED)
    return im

def complete_paths(clips):
    im=base('SUPPLEMENTAL OBSERVATIONS - NOT NEW PREFERENCE SCORES','Full trajectories: no stop at first choice','Gray = complete two-second group paths. Yellow = filmed fly. A white ring marks its original first choice.')
    d=ImageDraw.Draw(im)
    for i,c in enumerate(CONDITIONS[:3]):
        clip=clips[c];x=40+624*i;text(d,(x+10,212),NAMES[c],30,COLORS[c],True)
        graph=clip.draw_map_background(590,660);draw=ImageDraw.Draw(graph);p,t=clip.paths['test']
        draw.line([clip.point(q,590,660) for q in p],fill=YELLOW,width=4)
        q=np.array(clip.point(p[-1],590,660));v=q-np.array(clip.point(p[-5],590,660));v=v/(np.linalg.norm(v)+1e-9)
        arrow(draw,q-v*22,q,YELLOW,4)
        if clip.summary['latency_s'] is not None:
            u,v=clip.point(clip.choice_pos,590,660);draw.ellipse((u-6,v-6,u+6,v+6),outline='white',width=2)
        im.paste(graph,(x,265));text(d,(x+15,952),'Physical two-choice arena, not the old planar Y-maze.',21,MUTED)
    return im

def build(preview=False):
    clips={c:FlyClip(c,preview) for c in CONDITIONS}
    stills={'training':panel(clips,'training',2.6),'test':panel(clips,'test',1.6),'trajectories':complete_paths(clips)}
    stills['intro']=title_card('A closer fly view, with complete paths',["The fly fills the main overhead view; the arena map is in the corner.","Supplemental recordings continue through the original two-second test window.","A white ring marks the first choice. Movement after it is observation, not a new score.","Circuit paths animate from recorded activity. No audio."])
    stills['legend']=title_card('The choice threshold no longer stops the camera',["Before: each trial ended as soon as its first choice was recorded.","Now: the same seeded trial continues; its first choice and latency remain fixed.","Gray group paths and the yellow fly path continue past the choice zones.","The same two-second deadline remains. No missing path has been drawn or guessed."])
    stills['results']=Image.open(OUT/'figures/results.png').convert('RGB')
    stills['limits']=title_card('Preserved results; newly observed continuations',["Original preference scores remain unchanged; continuation segments are excluded.","All pre-choice control and physics rows are checked against the original recordings.","This is not accelerated biological time or a free-ranging experiment.","The old truncated planar Y-maze is replaced here by complete physical-test paths."])
    for k,im in stills.items():im.save(DEST/f'{k}.png')
    for phase,t in [('training',2.6),('test',1.6)]:panel(clips,phase,t,['dan_silenced']).crop((32,200,640,1028)).save(DEST/f'dan_{phase}.png')
    if preview:return
    for name,count in [('continuation_batch',400),('continuation_pilot',4)]:
        v=json.loads((ROOT/'results'/name/'continuation_validation.json').read_text());assert len(v)==count and all(r['prefix_exact'] for r in v)
    chapters=[('intro',10),('legend',16),('training',30),('test',20),('trajectories',12),('results',20),('limits',16)]
    target=DEST/'fly_ethanol_reward_closeup.mp4';writer=iio.get_writer(target,fps=25,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-preset','fast'])
    for name,duration in chapters:
        print(name,flush=True)
        for f in range(duration*25):writer.append_data(np.asarray(panel(clips,name,f/250) if name in ['training','test'] else stills[name]))
    writer.close()
    for c in CONDITIONS:
        for phase,dur in [('training',30),('test',20)]:
            path=DEST/f'{c}_{phase}.mp4'
            if c!='dan_silenced':
                start=26 if phase=='training' else 56;col=CONDITIONS.index(c)
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-ss',str(start),'-i',str(target),'-t',str(dur),'-vf',f'crop=608:828:{32+624*col}:200','-c:v','libx264','-crf','19','-preset','fast','-an',str(path)],check=True)
            else:
                w=iio.get_writer(path,fps=25,codec='libx264',quality=8,macro_block_size=1)
                for f in range(dur*25):w.append_data(np.asarray(panel(clips,phase,f/250,[c]).crop((32,200,640,1028))))
                w.close()
            print(c,phase,flush=True)
    manifest=[]
    for p in DEST.glob('*.mp4'):
        probe=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-i',str(p)],capture_output=True,text=True);assert 'Audio:' not in probe.stderr
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(p),'-f','null','-'],check=True)
        manifest.append({'file':p.name,'audio_streams':0,'fully_decoded':True,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (DEST/'media_manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(DEST/'closeup_examples.zip','w',compression=zipfile.ZIP_STORED) as z:
        for p in DEST.glob('*.mp4'):
            if p!=target:z.write(p,p.name)
        z.write(DEST/'README.md','README.md')
    print('Nine silent videos validated',flush=True)
if __name__=='__main__':build('--preview' in sys.argv)
