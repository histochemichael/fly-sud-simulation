"""Data-driven illustrative circuit overlays and unobstructed overhead replays."""
import json,sys,subprocess,zipfile,hashlib
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
import imageio.v2 as iio
import imageio_ffmpeg
from build_video_revision import Subject,CONDITIONS,GROUP,LIVE,atlas,historical_y,title_card
from build_film import ROOT,OUT,base,text,wrap,COLORS,NAMES,ACCENT,TEXT,MUTED,BG,PANEL
from render_overhead import project
DEST=OUT/'overhead_revision';DEST.mkdir(exist_ok=True)

class OverheadSubject(Subject):
    def __init__(self,c):
        super().__init__(c)
        folder=OUT/'overhead_frames'/c
        self.overhead=[Image.open(folder/f'{i:04d}.png').convert('RGB') for i in range(len(self.images))]
        self.cameras=json.loads((folder/'camera.json').read_text())
        self.cached={}
        for phase,ids in self.byphase.items():
            rows=[self.states.iloc[self.rep['state_index'][i]] for i in ids]
            times=np.array([r.simulation_time_s if phase=='training' else r.phase_time_s for r in rows])
            self.cached[phase]=(ids,rows,times)
        self.side=1 if self.summary['odor_b_side']=='left' else -1
        self.group_world=[np.column_stack((p[:,1],p[:,0]*self.side,np.zeros(len(p)))) for p in self.group]
        self.layers={}
        for phase,cam in self.cameras.items():
            overlay=Image.new('RGBA',(608,352),(0,0,0,0));d=ImageDraw.Draw(overlay)
            def pp(p):return tuple(project([p],cam)[0])
            if phase=='test':
                for side in [-1,1]:
                    is_b=side==self.side
                    color=(245,181,82,38) if is_b and c in ['paired','dan_silenced'] else (97,201,196,28)
                    poly=[pp((a,b*side,0)) for a,b in [(12,2.5),(26,2.5),(26,17),(12,17)]]
                    d.polygon(poly,fill=color,outline=color[:3]+(135,))
            # Keep every path visible: no centered status mask or data clipping.
            for pts in self.group_world:d.line([tuple(p) for p in project(pts,cam)],fill=(145,164,178,170),width=1)
            self.layers[phase]=overlay
    def get_overhead(self,phase,t):
        ids,rows,times=self.cached[phase];j=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(ids)-1))
        return self.overhead[ids[j]].copy(),rows[j],j==len(ids)-1
    def arena(self,phase,t):
        frame,row,ended=self.get_overhead(phase,t);cam=self.cameras[phase]
        frame=Image.alpha_composite(frame.convert('RGBA'),self.layers[phase]);d=ImageDraw.Draw(frame)
        states=self.states[self.states.phase==phase]
        key='simulation_time_s' if phase=='training' else 'phase_time_s';s=states[states[key]<=float(row[key])]
        pts=np.column_stack((s.x,s.y,np.zeros(len(s))))
        pix=project(pts,cam)
        if len(pix)>1:d.line([tuple(p) for p in pix],fill=LIVE,width=3)
        q=project([[row.x,row.y,0]],cam)[0]
        d.ellipse((q[0]-9,q[1]-9,q[0]+9,q[1]+9),outline=LIVE,width=2)
        # Heading arrow comes from the recorded pose, not an invented destination.
        end=project([[row.x+2*np.cos(row.heading_rad),row.y+2*np.sin(row.heading_rad),0]],cam)[0]
        arrow(d,q,end,LIVE,3)
        for side in [-1,1]:
            is_b=side==self.side;u,v=project([[19,side*9,0]],cam)[0]
            left=u<304;lx=8 if left else 460
            label='ODOR B' if is_b else 'ODOR A';color=ACCENT if is_b and self.condition in ['paired','dan_silenced'] else '#73DAD1'
            text(d,(lx,47),label,23,color,True)
            status='PREVIOUSLY' if is_b and self.condition in ['paired','dan_silenced'] else 'NOT REWARD-'
            status2='REWARDED CUE' if is_b and self.condition in ['paired','dan_silenced'] else 'CONDITIONED'
            text(d,(lx,77),status,16,TEXT,True);text(d,(lx,98),status2,16,TEXT,True)
            arrow(d,(lx+72,130),(u,v+6),color,2)
            d.ellipse((u-12,v-12,u+12,v+12),outline=color,width=2)
        text(d,(9,6),'OVERHEAD | actual body replay',19,TEXT,True)
        text(d,(9,306),'GRAY: test group (100)  |  YELLOW: current fly',19,TEXT,True)
        text(d,(9,330),'Training is uniform; markers show later test options.' if phase=='training' else 'Shaded areas: choice zones. NO REWARD anywhere in test.',17,ACCENT if phase=='test' else TEXT)
        return frame.convert('RGB'),row,ended

def arrow(d,a,b,color,width=2):
    a=np.array(a,float);b=np.array(b,float);d.line([tuple(a),tuple(b)],fill=color,width=width)
    v=b-a;v=v/(np.linalg.norm(v)+1e-9);n=np.array([-v[1],v[0]])
    d.polygon([tuple(b),tuple(b-v*8+n*4),tuple(b-v*8-n*4)],fill=color)

def curve(a,b,c):
    t=np.linspace(0,1,18)[:,None];return (1-t)**2*np.asarray(a)+2*(1-t)*t*np.asarray(b)+t*t*np.asarray(c)

def circuit_overlay(im,x,y,row,t):
    im.paste(atlas,(x,y));d=ImageDraw.Draw(im)
    # Positions and paths are illustrative, NOT anatomically registered.
    for side,short,sign in [('left','l',-1),('right','r',1)]:
        pn=np.array([x+175+sign*103,y+127]);mb=np.array([x+175+sign*35,y+116])
        for k in range(24):
            kc=np.array([x+175+sign*(35+(k%6)*8),y+23+(k//6)*10])
            activity=float(row[f'kc_activity_{short}_{k:02d}']);weight=float(row[f'kc_weight_{k:02d}'])
            paths=[(curve(pn,[pn[0]+sign*20,kc[1]+35],kc),activity),(curve(kc,[mb[0]+sign*16,kc[1]+50],mb),activity*weight)]
            for path,value in paths:
                q=float(np.clip(value,0,1));color=(int(70+185*q),int(45+35*q),int(48+25*q))
                d.line([tuple(p) for p in path],fill=color,width=1+int(3*q))
                if q>.02:
                    idx=int((t*4+k*.071)%1*(len(path)-1));u,v=path[idx]
                    d.ellipse((u-1-q,v-1-q,u+1+q,v+1+q),fill=(255,170,110))
            d.ellipse((kc[0]-2,kc[1]-2,kc[0]+2,kc[1]+2),fill=(int(75+180*np.clip(activity,0,1)),55,55))
        for p,label,value in [(pn,'PN',row[f'pn_{side}']),(mb,'MBON',row[f'mbon_{side}'])]:
            d.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),fill=(255,80,65) if value>.01 else '#4F4145')
            text(d,(p[0]-18,p[1]+6),label,12,TEXT,True)
    dan=np.array([x+175,y+72]);value=float(np.clip(row.dan,0,1))
    for sign in [-1,1]:
        end=np.array([x+175+sign*35,y+110]);d.line([tuple(dan),tuple(end)],fill=(int(70+185*value),int(50+130*value),40),width=1+int(3*value))
    d.ellipse((dan[0]-6,dan[1]-6,dan[0]+6,dan[1]+6),fill=ACCENT if value else '#4F4145')
    text(d,(dan[0]-15,dan[1]-21),'DAN',12,TEXT,True);text(d,(x+152,y+3),'KCs',13,TEXT,True)

def panel(clips,phase,t,conditions=None):
    train=phase=='training'
    im=base('SILENT | RECORDED POSES + MODEL ACTIVITY',
            'TRAINING: Odor B predicts reward only when paired' if train else 'TEST: follow the fly toward its two odor options',
            'No paths are hidden. Gray = test group reference; yellow = this fly during training.' if train else 'No reward is delivered. Choices are recorded at zone entry, before reaching the odor source.')
    d=ImageDraw.Draw(im)
    for col,c in enumerate(conditions or CONDITIONS[:3]):
        x=32+624*col;frame,row,ended=clips[c].arena(phase,t)
        d.rounded_rectangle((x,200,x+608,1028),radius=14,fill=PANEL)
        text(d,(x+14,208),NAMES[c],30,COLORS[c],True)
        label={'untrained':'No reward-conditioned odor','paired':'B = ethanol-paired cue; A = unpaired cue','unpaired':'Ethanol was separate from both odors','dan_silenced':'B paired in training; DAN gated off in test'}[c]
        text(d,(x+14,247),label,24,TEXT,True)
        event=str(row.event).upper() if train else 'TEST: NO ETHANOL / NO TEACHING REWARD'
        if row.teaching_signal>0:event='REWARD NOW: '+('B + ETHANOL' if c in ['paired','dan_silenced'] else 'ETHANOL ALONE')
        d.rounded_rectangle((x+8,284,x+600,330),radius=5,fill='#684910' if row.teaching_signal else '#20374D')
        text(d,(x+17,292),event,23,ACCENT if row.teaching_signal else TEXT,True)
        when=f'Session {int(row.conditioning_session)} | {t:.2f} s' if train else f'Test {min(t,clips[c].times[-1]):.2f} s'
        text(d,(x+12,338),when+f'  |  Memory A {row.weight_a:.2f}, B {row.weight_b:.2f}',21,TEXT)
        im.paste(frame,(x,365))
        if ended and not train:
            choice=clips[c].summary['choice'];status='NO CHOICE: time limit' if choice=='none' else 'CHOSE ODOR '+choice
            text(d,(x+12,718),status+' | final pose held',21,ACCENT,True)
        else:text(d,(x+12,718),'LIVE HEADING: yellow arrow follows the recorded fly',20,MUTED)
        text(d,(x+12,748),'ILLUSTRATIVE CIRCUITS ON REFERENCE ANATOMY',20,TEXT,True)
        circuit_overlay(im,x+10,778,row,t)
        text(d,(x+370,778),'SIMULATED ACTIVITY',18,ACCENT,True)
        for j,(label,value) in enumerate([('PN',(row.pn_left+row.pn_right)/2),('KC',(row.kc_left+row.kc_right)/2),('MBON',(row.mbon_left+row.mbon_right)/2),('DAN',row.dan)]):
            yy=805+36*j;text(d,(x+370,yy),f'{label}: {value:.2f}',17,TEXT)
            d.rectangle((x+370,yy+22,x+586,yy+28),fill='#31495D');d.rectangle((x+370,yy+22,x+370+int(216*np.clip(value,0,1)),yy+28),fill='#FF5849')
        text(d,(x+12,957),'Bright red paths = active; dim paths = inactive.',19,TEXT,True)
        text(d,(x+12,982),'Paths are stylized, not anatomically registered.',18,MUTED)
        text(d,(x+12,1005),'JFRC2 / VFB; Jenett 2012; Ito 2014; CC BY 4.0',16,MUTED)
    return im

def build(preview=False):
    clips={c:OverheadSubject(c) for c in CONDITIONS}
    stills={'training':panel(clips,'training',2.6),'test':panel(clips,'test',.65),'earlier_y':historical_y()}
    stills['intro']=title_card('Watch the fly, its choices, and its circuit',["OVERHEAD: actual saved body poses, with a live path and heading arrow.","TRAJECTORIES: all group paths stay visible; no box hides the middle.","CIRCUIT: logged PN, KC, MBON and DAN activity drives illustrative red paths.","TEST: no reward anywhere. Labels identify the previously conditioned odor."])
    stills['legend']=title_card('Where was reward conditioned?',["PAIRED TRAINING: Odor B + ethanol, delivered uniformly across the training arena.","TEST: Odor A and Odor B appear as two spatial options; ethanol is absent.","Choice zones are shaded. A recorded choice can occur before reaching the source.","The gray group is aligned to this fly's odor sides. It is not a drawn Y-shaped arena."])
    stills['results']=Image.open(OUT/'figures/results.png').convert('RGB')
    stills['limits']=title_card('An animated model circuit, not biological imaging',["The brain background is anatomical. Red paths are an illustrative model overlay.","Activity and weights come from saved simulation states. Moving dots depict flow, not spikes.","No anatomical wiring, full connectome, or delayed consolidation was added.","The earlier planar Y-maze appears separately. Scientific outcomes are unchanged."])
    for k,im in stills.items():im.save(DEST/f'{k}.png')
    for phase,t in [('training',2.6),('test',.65)]:panel(clips,phase,t,['dan_silenced']).crop((32,200,640,1028)).save(DEST/f'dan_{phase}.png')
    if preview:return
    chapters=[('intro',10),('legend',16),('training',30),('test',20),('earlier_y',12),('results',20),('limits',16)]
    target=DEST/'fly_ethanol_reward_overhead.mp4'
    writer=iio.get_writer(target,fps=25,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-preset','fast'])
    for name,duration in chapters:
        print(name,flush=True)
        for f in range(duration*25):writer.append_data(np.asarray(panel(clips,name,f/250) if name in ['training','test'] else stills[name]))
    writer.close()
    for c in CONDITIONS:
        for phase,duration in [('training',30),('test',20)]:
            path=DEST/f'{c}_{phase}.mp4'
            if c!='dan_silenced':
                start=26 if phase=='training' else 56;col=CONDITIONS.index(c)
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-ss',str(start),'-i',str(target),'-t',str(duration),'-vf',f'crop=608:828:{32+624*col}:200','-c:v','libx264','-crf','19','-preset','fast','-an',str(path)],check=True)
            else:
                w=iio.get_writer(path,fps=25,codec='libx264',quality=8,macro_block_size=1)
                for f in range(duration*25):w.append_data(np.asarray(panel(clips,phase,f/250,[c]).crop((32,200,640,1028))))
                w.close()
            print(c,phase,flush=True)
    manifest=[]
    for p in DEST.glob('*.mp4'):
        probe=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-i',str(p)],capture_output=True,text=True)
        assert 'Audio:' not in probe.stderr
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(p),'-f','null','-'],check=True)
        manifest.append({'file':p.name,'audio_streams':0,'fully_decoded':True,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (DEST/'media_manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(DEST/'overhead_examples.zip','w',compression=zipfile.ZIP_STORED) as z:
        for p in DEST.glob('*.mp4'):
            if p!=target:z.write(p,p.name)
        z.write(DEST/'README.md','README.md')
    print('Complete: nine silent videos verified',flush=True)
if __name__=='__main__':build('--preview' in sys.argv)
