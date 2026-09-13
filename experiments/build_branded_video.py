"""Experiment-first silent film: branded intro, four conditions, evidence and outlook."""
import json,subprocess,zipfile,hashlib,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as iio
import imageio_ffmpeg
from build_closeup_video import FlyClip,panel,GRAY,YELLOW,arrow
from build_film import ROOT,OUT,base,text,wrap,BG,PANEL,TEXT,MUTED,ACCENT,NAMES,COLORS
from build_video_revision import CONDITIONS
DEST=OUT/'sleek_revision';DEST.mkdir(exist_ok=True)
LOGO_PATH=OUT/'assets/impulse_neuro_logo_transparent.png'
LOGO=Image.open(LOGO_PATH).convert('RGBA');LOGO.thumbnail((94,74),Image.Resampling.LANCZOS)
assert LOGO.getchannel('A').getextrema()[0]==0
def brand(im,header=True):
    im=im.copy();d=ImageDraw.Draw(im)
    if header:
        d.rectangle((0,0,1920,76),fill=BG)
        text(d,(48,26),'IMPULSE NEURO  /  EXPERIMENT FILM',22,'#76C9EF',True)
        d.line((48,72,1748,72),fill='#23364B',width=1)
    im.paste(LOGO,(1900-LOGO.width,8),LOGO);return im
def card(title,subtitle=''):
    return brand(base('IMPULSE NEURO | ONLINE ASSOCIATIVE LEARNING',title,subtitle))
def lines(im,items,y=275,size=42,gap=140):
    d=ImageDraw.Draw(im)
    for head,body in items:
        text(d,(70,y),head,30,ACCENT,True)
        wrap(d,(70,y+42),body,1760,size,TEXT);y+=gap
    return im
def introduction():
    a=card('Can learned reward steer a simulated fly?','EXPERIMENT / PAPER / QUESTION')
    lines(a,[('EXPERIMENT','Odor conditioning, then a reward-free physical choice test.'),
             ('REFERENCE','Kaun et al. (2011) | A Drosophila model for alcohol reward'),
             ('NATURE NEUROSCIENCE','14:612-619 | doi:10.1038/nn.2805')],gap=205)
    text(ImageDraw.Draw(a),(70,934),'Mechanism demonstration, not a full replication of the paper.',32,MUTED)
    b=card('Four conditions. One reward-free choice assay.','100 simulated flies per condition | 400 original tests | 4 filmed representatives')
    d=ImageDraw.Draw(b)
    conditions=[('untrained','No cue-reward conditioning.'),('paired','Odor B paired with reward.'),('unpaired','Reward separated from odor.'),('dan_silenced','Paired training; DAN retrieval gate off.')]
    for i,(c,desc) in enumerate(conditions):
        x=64+(i%2)*920;y=265+(i//2)*245
        d.rounded_rectangle((x,y,x+885,y+210),radius=20,fill=PANEL)
        text(d,(x+26,y+22),NAMES[c],40,COLORS[c],True);wrap(d,(x+26,y+90),desc,825,35)
    text(d,(70,845),'3 conditioning sessions / fly | 1 test / fly | DAN seeds matched to paired',32,TEXT)
    text(d,(70,908),'Post-choice continuations reuse those tests; they are not extra independent replicates.',28,MUTED)
    c=card('Physics, gait and compute','MuJoCo 3.9.0: rigid-body dynamics and contact | FlyGym 2.1.0')
    lines(c,[('NEUROMECHFLY','69 body segments | 6 legs | 42 actuated joint motions (7 per leg)'),
             ('GAIT','6 coupled oscillators coordinate a tripod stepping pattern.'),
             ('LOCAL HARDWARE','Intel i7-9700F: 8 CPU workers | approximately 64 GB RAM'),
             ('GPU','RTX 5060 Ti installed; no GPU-accelerated physics used.')],y=235,size=37,gap=170)
    dcard=card('A functional circuit hypothesis','Synthetic connectivity on anatomical reference imagery; not reconstructed biological wiring')
    d=ImageDraw.Draw(dcard)
    nodes=[('ORN','Antennal odor input'),('AL / PN','Odor representation'),('KC','Mushroom-body code'),('MBON','Learned output')]
    for i,(label,role) in enumerate(nodes):
        x=65+i*460;d.rounded_rectangle((x,285,x+405,475),radius=18,fill=PANEL)
        text(d,(x+24,307),label,50,ACCENT,True);text(d,(x+24,386),role,29,TEXT)
        if i<3:arrow(d,(x+410,370),(x+450,370),ACCENT,4)
    text(d,(70,525),'24 synthetic KC identities | plastic KC-to-MBON weights | output steers gait',33,TEXT)
    text(d,(70,603),'DAN: reward teaching during training; assumed retrieval gate during test.',33,TEXT)
    text(d,(70,732),'HYPOTHESIS',32,ACCENT,True)
    wrap(d,(70,782),'Pairing increases B preference; retrieval-DAN silencing reduces the learned bias.',1740,43)
    text(d,(70,939),'Online reward-modulated learning, not deep learning or a complete addiction model.',29,MUTED)
    return {'paper_intro':a,'conditions_intro':b,'engine_intro':c,'circuit_intro':dcard}
def dan_comparison(clips,phase,t):
    im=card('DAN control: identical training' if phase=='training' else 'DAN control: retrieval gate off',
            'Paired and DAN-silenced representatives are seed-matched. Training is unchanged; only test retrieval differs.')
    source=panel(clips,phase,t,['paired','dan_silenced'])
    for i,x in enumerate([180,1132]):im.paste(source.crop((32+624*i,200,640+624*i,1028)),(x,200))
    d=ImageDraw.Draw(im);text(d,(830,465),'PAIRED',26,ACCENT,True);text(d,(830,510),'versus',26,MUTED);text(d,(830,555),'DAN OFF',26,COLORS['dan_silenced'],True)
    text(d,(830,615),'in test only',23,TEXT)
    return im
def trajectories(clips):
    im=card('Complete movement trajectories: all four conditions','Gray: all 100 group flies | yellow: filmed fly | white ring: first choice | arrows: direction of travel')
    d=ImageDraw.Draw(im)
    for i,c in enumerate(CONDITIONS):
        x=30+472*i;clip=clips[c];text(d,(x+12,210),NAMES[c],32,COLORS[c],True)
        graph=clip.draw_map_background(448,660);draw=ImageDraw.Draw(graph);p,_=clip.paths['test']
        pts=[clip.point(q,448,660) for q in p];draw.line(pts,fill=YELLOW,width=3)
        q=np.array(pts[-1]);v=q-np.array(pts[-5]);v/=np.linalg.norm(v)+1e-9;arrow(draw,q-v*20,q,YELLOW,3)
        if clip.summary['latency_s'] is not None:
            u,v=clip.point(clip.choice_pos,448,660);draw.ellipse((u-6,v-6,u+6,v+6),outline='white',width=2)
        im.paste(graph,(x,270))
    text(d,(50,956),'All recordings continue to 2 s. Choice-zone shading is not a wall or a reward-delivery area.',29,TEXT)
    return im
def endings():
    a=lines(card('What did we learn?','ORIGINAL SCORES / 100 SIMULATED FLIES PER CONDITION'),[
        ('LEARNED PREFERENCE','Paired: 68 B, 23 A, 9 no choice. B = 74.7% of responders.'),
        ('CONTROLS','Untrained: 40 B / 46 A. Unpaired: 42 B / 43 A. Near-even choice.'),
        ('RETRIEVAL INTERVENTION','DAN-off: 39 B / 47 A / 14 no choice. Learned weights remain intact.'),
        ('INTERPRETATION','The specified learning-and-steering mechanism is sufficient for cue preference.')],y=230,size=35,gap=175)
    b=lines(card('How does this relate to Kaun et al. (2011)?','QUALITATIVE ANALOGY, NOT QUANTITATIVE OR BIOLOGICALLY COMPLETE REPLICATION'),[
        ('AGREEMENT','Reward-paired cue preference; a retrieval-DAN dependence.'),
        ('NOT REPRODUCED','The paper\'s early aversion, delayed preference and consolidation.'),
        ('IMPORTANT MISMATCH','Our acquisition needs DAN teaching by construction; the paper found a retrieval requirement.'),
        ('DIFFERENT ASSAY','Compressed timing, generic odors and individual PI differ from the paper\'s group CPI.')],y=230,size=35,gap=175)
    text(ImageDraw.Draw(b),(70,957),'Source: Kaun et al., Nature Neuroscience 14:612-619 | doi:10.1038/nn.2805',27,MUTED)
    c=lines(card('Next steps: validate time before scaling the arena','PROPOSED ONLY / NO NEW TIME-COURSE OR FREE-RANGING EXPERIMENT'),[
        ('1. MATCH THE ASSAY','Reciprocal odors, biological test delays, independent seeds and replicate-group statistics.'),
        ('2. TEST SLOW-STATE HYPOTHESES','Predict held-out delays and perturbations; address the acquisition-DAN mismatch.'),
        ('3. VALIDATE ANY SPEEDUP','Compare accelerated dynamics with full-resolution physics before free-ranging studies.'),
        ('4. THEN EXPLORE FREE-RANGING FLIES','Record self-selected exposures; return to a controlled reward-free test.')],y=230,size=35,gap=175)
    return {'learned':a,'paper_limits':b,'next_steps':c}
def individual_result_frame(name):
    im=Image.new('RGB',(1920,1080),BG)
    plot=Image.open(OUT/'individual_results'/f'{name}.png').convert('RGB').resize((1600,1028),Image.Resampling.LANCZOS)
    im.paste(plot,(160,26));return brand(im,header=False)
CHAPTERS=[('paper_intro',6),('conditions_intro',6),('engine_intro',8),('circuit_intro',6),('training',30),('test',20),('dan_training',30),('dan_test',20),('trajectories',14),('results',20),('learned',12),('paper_limits',12),('next_steps',16)]
def build(preview=False):
    clips={c:FlyClip(c) for c in CONDITIONS};stills=introduction();stills.update(endings())
    stills.update(training=brand(panel(clips,'training',2.6)),test=brand(panel(clips,'test',1.6)),dan_training=dan_comparison(clips,'training',2.6),dan_test=dan_comparison(clips,'test',1.6),trajectories=trajectories(clips),results=individual_result_frame('choice'),results_latency=individual_result_frame('latency'))
    for k,im in stills.items():im.save(DEST/f'{k}.png')
    if preview:return
    for name,n in [('continuation_batch',400),('continuation_pilot',4)]:
        rows=json.loads((ROOT/'results'/name/'continuation_validation.json').read_text());assert len(rows)==n and all(r['prefix_exact'] and r['choice_preserved'] for r in rows)
    target=DEST/'fly_ethanol_reward_sleek.mp4';w=iio.get_writer(target,fps=25,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-preset','fast'])
    timeline=[];start=0
    for name,dur in CHAPTERS:
        print(name,flush=True);chapter_start=start;timeline.append({'chapter':name,'start_s':start,'duration_s':dur});start+=dur
        for f in range(dur*25):
            if name in ['training','test']:im=brand(panel(clips,name,f/250))
            elif name in ['dan_training','dan_test']:im=dan_comparison(clips,name[4:],f/250)
            elif name=='results':im=stills['results' if f<250 else 'results_latency']
            else:im=stills[name]
            im=im.copy()
            # Fade static cards only. Never blend or omit recorded fly movement.
            if name not in ['training','test','dan_training','dan_test'] and f<6:
                im=Image.blend(Image.new('RGB',im.size,BG),im,(f+1)/6)
                im.paste(LOGO,(1900-LOGO.width,8),LOGO)
            d=ImageDraw.Draw(im);d.rectangle((48,1074,1872,1076),fill='#263A50')
            d.rectangle((48,1074,48+int(1824*(chapter_start+f/25)/200),1076),fill='#76C9EF')
            w.append_data(np.asarray(im))
    w.close();ff=imageio_ffmpeg.get_ffmpeg_exe()
    for c in CONDITIONS:
        for phase in ['training','test']:
            name=f'{c}_{phase}.mp4'
            subprocess.run([ff,'-y','-v','error','-i',str(OUT/'closeup_revision'/name),'-i',str(LOGO_PATH),'-filter_complex','[1:v]scale=70:56:flags=lanczos,format=rgba[logo];[0:v][logo]overlay=W-w-8:8','-c:v','libx264','-preset','fast','-crf','19','-an',str(DEST/name)],check=True)
            print(name,flush=True)
    manifest=[]
    for p in DEST.glob('*.mp4'):
        probe=subprocess.run([ff,'-hide_banner','-i',str(p)],capture_output=True,text=True);assert 'Audio:' not in probe.stderr
        subprocess.run([ff,'-v','error','-i',str(p),'-f','null','-'],check=True)
        manifest.append({'file':p.name,'audio_streams':0,'fully_decoded':True,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    assert len(manifest)==9
    (DEST/'media_manifest.json').write_text(json.dumps({'chapters':timeline,'duration_s':start,'files':manifest},indent=2))
    with zipfile.ZipFile(DEST/'sleek_examples.zip','w',compression=zipfile.ZIP_STORED) as z:
        for p in DEST.glob('*.mp4'):
            if p!=target:z.write(p,p.name)
        z.write(DEST/'README.md','README.md')
    print('Nine silent branded videos validated; duration',start,flush=True)
if __name__=='__main__':build('--preview' in sys.argv)
