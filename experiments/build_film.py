"""Make an annotated scientific film from saved MuJoCo replay and state data."""
import json,math,sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output";FIG=OUT/"figures";FIG.mkdir(exist_ok=True,parents=True)
W,H=1920,1080
BG="#0B1423";PANEL="#132338";TEXT="#EFF4F7";MUTED="#A8B9CA";ACCENT="#F5B552"
COLORS={"untrained":"#8DABBE","paired":"#F5B552","unpaired":"#61C9C4","dan_silenced":"#BD9BDB"}
NAMES={"untrained":"UNTRAINED","paired":"PAIRED","unpaired":"UNPAIRED","dan_silenced":"DAN SILENCED"}
def font(size,bold=False): return ImageFont.truetype("C:/Windows/Fonts/"+("calibrib.ttf" if bold else "calibri.ttf"),size)
def text(draw,xy,s,size=30,fill=TEXT,bold=False): draw.text(xy,str(s),font=font(size,bold),fill=fill)
def wrap(draw,xy,s,width,size=30,fill=TEXT):
    x,y=xy;line=""
    for word in s.split():
        candidate=(line+" "+word).strip()
        if draw.textlength(candidate,font=font(size))>width and line:
            text(draw,(x,y),line,size,fill);y+=size*1.3;line=word
        else: line=candidate
    text(draw,(x,y),line,size,fill)
    return y+size*1.4
def base(k,title,subtitle):
    im=Image.new("RGB",(W,H),BG);d=ImageDraw.Draw(im)
    text(d,(48,32),"FLY ETHANOL REWARD  /  EXPERIMENT FILM",24,ACCENT,True)
    text(d,(48,83),title,58,TEXT,True);text(d,(50,151),subtitle,28,MUTED)
    text(d,(48,1035),k,22,MUTED);text(d,(1460,1035),"LOCAL SIMULATION  |  SEPT 2026",22,MUTED)
    return im

def heat(v):
    q=float(np.clip(v,0,1));a=np.array([30,62,80]);b=np.array([255,177,68])
    return tuple((a*(1-q)+b*q).astype(int))

def brain(d,x,y,row):
    text(d,(x+15,y),"MODEL BRAIN  |  SCHEMATIC CUTAWAY",20,MUTED,True)
    for cx in [x+158,x+450]:
        d.ellipse((cx-120,y+32,cx+120,y+278),fill="#112B3D",outline="#35566C",width=2)
    # Pathways are schematic. Intensity uses fixed [0,1] model rate scale.
    for side,cx in [("left",x+158),("right",x+450)]:
        for yy in [y+86,y+143,y+205]:
            d.line((cx,yy,cx,yy+45),fill="#6B8193",width=3)
        for key,label,yy in [("al","AL / PN",y+215),("mbon","MBON",y+93)]:
            val=float(row[f"{key}_{side}"])
            d.ellipse((cx-38,yy-23,cx+38,yy+23),fill=heat(val),outline="#7893A4",width=2)
            text(d,(cx-60,yy+23),label,19)
        # Individually logged Kenyon cells, 24 per side.
        short="l" if side=="left" else "r"
        for k in range(24):
            xx=cx-64+(k%8)*18;yy=y+140+(k//8)*17
            d.ellipse((xx,yy,xx+11,yy+11),fill=heat(row[f"kc_activity_{short}_{k:02d}"]))
        text(d,(cx-18,y+187),"KC",18)
        text(d,(cx-50,y+265),side.upper(),17,MUTED)
    cx=x+304;cy=y+63
    d.ellipse((cx-27,cy-22,cx+27,cy+22),fill=heat(row["dan"]))
    text(d,(cx-26,cy-45),"DAN",19,TEXT,True)
    d.line((cx-28,cy,x+205,y+106),fill=ACCENT,width=2)
    d.line((cx+28,cy,x+402,y+106),fill=ACCENT,width=2)
    text(d,(x+15,y+306),f"Memory A {row.weight_a:.3f}    B {row.weight_b:.3f}",24,TEXT,True)

class Clip:
    def __init__(self,condition):
        self.condition=condition
        p=ROOT/"results/physical_pilot"/f"{condition}_000"
        self.states=pd.read_parquet(p/"state_history.parquet")
        self.rep=np.load(p/"replay.npz")
        self.summary=json.loads((p/"summary.json").read_text())
        self.images=[Image.open(OUT/"frames"/condition/f"{i:04d}.png").convert("RGB") for i in range(len(self.rep["state_index"]))]
        self.byphase={}
        for phase in ["training","test"]:
            self.byphase[phase]=[i for i,s in enumerate(self.rep["state_index"]) if self.states.iloc[s].phase==phase]
    def get(self,phase,t):
        ids=self.byphase[phase]
        times=[float(self.states.iloc[self.rep["state_index"][i]][
            "simulation_time_s" if phase=="training" else "phase_time_s"]) for i in ids]
        ii=min(max(int(np.searchsorted(times,t,side="right"))-1,0),len(ids)-1);frame=ids[ii]
        return self.images[frame],self.states.iloc[self.rep["state_index"][frame]],ii==len(ids)-1

def comparison_panel(clips,phase,t,conditions=None):
    training=phase=="training"
    im=base("02  /  TRAINING" if training else "03  /  REWARD-FREE TEST",
            "Learning during experience" if training else "Testing memory through movement",
            "Three sessions, uniform cue delivery. Playback 0.1x. Exposure durations compressed for this model." if training else "Actual NeuroMechFly motion in a physical two-choice arena. Playback 0.1x. No teaching reward.")
    d=ImageDraw.Draw(im)
    for col,c in enumerate(conditions or ["untrained","paired","unpaired"]):
        x=32+624*col;clip=clips[c];frame,row,ended=clip.get(phase,t)
        d.rounded_rectangle((x,200,x+608,1028),radius=16,fill=PANEL)
        text(d,(x+20,214),NAMES[c],30,COLORS[c],True)
        status=f"Session {int(row.conditioning_session)} | {row.event}" if training else f"Fly 000 | B on {row.odor_b_side}"
        text(d,(x+20,255),status,23)
        im.paste(frame,(x,294))
        # Two-choice map uses actual logged position; all trials use +x forward.
        mx,my=x+440,320
        d.rounded_rectangle((mx,my,mx+150,my+126),radius=8,fill=BG)
        def mappt(px,py): return (mx+12+px/27*124,my+64-py/18*51)
        for sy in [-9,9]:
            pt=mappt(19,sy);b=(sy>0)==(row.odor_b_side=="left")
            d.ellipse((pt[0]-5,pt[1]-5,pt[0]+5,pt[1]+5),fill="#B78DF2" if b else "#61C9C4")
            text(d,(pt[0]+7,pt[1]-9),"B" if b else "A",16)
        p=mappt(min(max(row.x,0),26),min(max(row.y,-17),17))
        d.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill=ACCENT)
        text(d,(mx+10,my+105),"POSITION  /  mm",14,MUTED)
        text(d,(x+15,655),f"Ethanol {row.ethanol_state:.0f}   Teaching reward {row.teaching_signal:.1f}",24,ACCENT if row.teaching_signal else MUTED)
        brain(d,x,698,row)
        if ended and not training:
            label="NONRESPONSE" if clip.summary["choice"]=="none" else "CHOICE "+clip.summary["choice"]
            d.rounded_rectangle((x+18,590,x+590,631),radius=8,fill=BG)
            text(d,(x+30,592),label+"  |  final frame held",24,COLORS[c],True)
    return im

def chart(summary):
    order=["untrained","paired","unpaired","dan_silenced"]
    s=summary.set_index("condition").loc[order]
    fig,axes=plt.subplots(1,2,figsize=(14,5.4),gridspec_kw={"width_ratios":[1.3,1]})
    fig.patch.set_facecolor("white")
    xs=np.arange(4);vals=s.pi_responders.values
    axes[0].bar(xs,vals,color=[COLORS[c] for c in order],width=.62)
    axes[0].errorbar(xs,vals,yerr=[vals-s.ci95_low.values,s.ci95_high.values-vals],fmt="none",color="#24344A",capsize=5)
    axes[0].set_xticks(xs,["Untrained","Paired","Unpaired","DAN off"],fontsize=11)
    axes[0].set_ylim(-1,1.13);axes[0].axhline(0,color="#6F8094",lw=1)
    axes[0].set_ylabel("PI among completed choices (95% Wilson CI)")
    axes[0].set_title("OUR PHYSICAL SIMULATION\n100 flies per condition",loc="left",fontweight="bold")
    for i,c in enumerate(order):
        r=s.loc[c];axes[0].text(i,1.04,f"{int(r.b)} B / {int(r.completed)} choices",ha="center",fontsize=9)
        axes[0].text(i,-.92,f"{int(r.nonresponse)} no response",ha="center",fontsize=9)
    axes[1].axis("off");axes[1].set_title("KAUN ET AL. 2011\nPublished findings, Figure 1b",loc="left",fontweight="bold")
    items=[("30 minutes","Conditioned aversion","N = 8; p = 0.006"),
           ("24 hours","Conditioned preference","N = 8; p = 0.02"),
           ("Unpaired exposure","No consistent cue preference","Temporal association required")]
    for i,(time,result,detail) in enumerate(items):
        y=.88-i*.25;axes[1].text(0,y,time,fontweight="bold",fontsize=13)
        axes[1].text(0,y-.08,result,fontsize=13);axes[1].text(0,y-.15,detail,fontsize=11,color="#56657A")
    axes[1].text(0,.06,"Paper CPI averages reciprocal odor-pairing groups.\nIts N counts replicate groups; ours counts individuals.\nPaper bar heights are not digitized or invented.",fontsize=10,color="#56657A")
    for ax in axes[:1]:
        ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout();fig.savefig(FIG/"paper_comparison.png",dpi=170,bbox_inches="tight");plt.close(fig)
    return FIG/"paper_comparison.png"

def static_pages(summary,stats):
    pages=[]
    im=base("00  /  OVERVIEW","Can a simulated fly learn an ethanol cue?","A reproducible experiment with a physical body, an online learning circuit, and reward-free testing.")
    d=ImageDraw.Draw(im)
    text(d,(80,300),"A MEMORY THAT MOVES A BODY",40,ACCENT,True)
    wrap(d,(80,375),"NeuroMechFly walks in MuJoCo. Bilateral odor input activates a compact circuit. Experience changes its synapses. Later, the fly navigates with no teaching reward.",940,46)
    wrap(d,(80,760),"400 physical tests: 100 untrained, 100 paired, 100 unpaired, and 100 matched dopamine-silenced trials.",1620,38)
    photo=Image.open(OUT/"frames/paired/0020.png").resize((700,405));im.paste(photo,(1140,300))
    pages.append(("intro",im))
    im=base("01  /  WHAT WAS BUILT","From antennae to legs","The compact circuit is synthetic and biologically inspired. Its recorded rates drive the action.")
    d=ImageDraw.Draw(im)
    labels=[("ODOR","Two identities\nBilateral sensors"),("CIRCUIT","ORN / AL / PN\n24 Kenyon cells"),("MEMORY","KC to MBON plasticity\nDopamine modulation"),("BODY","6-leg gait controller\nMuJoCo contact physics")]
    for i,(label,body) in enumerate(labels):
        x=62+i*465;d.rounded_rectangle((x,290,x+420,540),radius=18,fill=PANEL)
        text(d,(x+25,320),label,34,ACCENT,True)
        for j,line in enumerate(body.split("\n")):text(d,(x+25,390+j*42),line,29)
        if i<3:text(d,(x+427,390),">",37,MUTED)
    wrap(d,(65,630),"Plasticity is updated online during conditioning. The model contains explicit activity variables for projection neurons, Kenyon cells, mushroom-body output neurons and a dopamine signal.",1720,37)
    wrap(d,(65,830),"10,000 physics steps per second. 100 circuit and sensor updates per second. Complete physics history joins to held control states; seeds and learned synapses are saved.",1720,32,MUTED)
    pages.append(("built",im))
    im=base("04  /  RESULTS","Simulation and published experiment","Compare the outcome, protocol and unit of replication before interpreting a preference index.")
    img=Image.open(chart(summary));img.thumbnail((1790,750));im.paste(img,((W-img.width)//2,232))
    pages.append(("results",im))
    im=base("05  /  INTERPRETATION","What the evidence supports","Behavioral similarity is a test of this mechanism, not biological validation of addiction.")
    d=ImageDraw.Draw(im)
    p=summary.set_index("condition").loc["paired"];u=summary.set_index("condition").loc["untrained"]
    bullets=[
        f"Physical paired trials: {int(p.b)} B choices, {int(p.a)} A choices, {int(p.nonresponse)} nonresponses.",
        "The original planar model and this physical model are different experiments.",
        "No 30-minute aversion or 24-hour consolidation is modeled.",
        "Brain colors show computed rates on a schematic cutaway, not brain imaging.",
        "Dopamine retrieval gating is a model assumption tested by a matched ablation.",
        "Nothing has been uploaded. Data, code, seeds and model parameters remain local."
    ]
    for i,line in enumerate(bullets):
        text(d,(68,282+i*107),f"{i+1:02d}",29,ACCENT,True)
        wrap(d,(130,278+i*107),line,1650,34)
    pages.append(("limits",im))
    for name,im in pages: im.save(FIG/f"{name}.png")
    return dict(pages)

def build():
    summary=pd.read_csv(ROOT/"results/physical_batch/summary.csv")
    stats=json.loads((ROOT/"results/physical_batch/statistics.json").read_text())
    clips={c:Clip(c) for c in ["untrained","paired","unpaired"]}
    static=static_pages(summary,stats)
    comparison_panel(clips,"training",2.6).save(FIG/"training.png")
    comparison_panel(clips,"test",.65).save(FIG/"testing.png")
    chapters=[("intro",12),("built",16),("training",30),("test",20),("results",20),("limits",22)]
    (OUT/"chapters.json").write_text(json.dumps(chapters))
    fps=25
    writer=iio.get_writer(OUT/"fly_ethanol_reward_silent.mp4",fps=fps,codec="libx264",quality=8,macro_block_size=1,ffmpeg_params=["-preset","fast","-pix_fmt","yuv420p"])
    for name,duration in chapters:
        print("Rendering chapter",name,flush=True)
        for frame in range(duration*fps):
            if name in ["training","test"]:
                im=comparison_panel(clips,name,frame/fps*.1)
            else: im=static[name].copy()
            d=ImageDraw.Draw(im);d.rectangle((0,H-5,int(W*(frame+1)/(duration*fps)),H),fill=ACCENT)
            writer.append_data(np.asarray(im))
    writer.close()
    for phase in ["training","test"]:
        for col,c in enumerate(["untrained","paired","unpaired"]):
            import subprocess,imageio_ffmpeg
            start=28 if phase=="training" else 58;duration=30 if phase=="training" else 20
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-ss",str(start),"-i",str(OUT/"fly_ethanol_reward_silent.mp4"),"-t",str(duration),"-vf",f"crop=608:828:{32+624*col}:200","-c:v","libx264","-crf","19","-preset","fast","-an",str(OUT/f"{c}_{phase}.mp4")],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print("Film and six condition clips saved",flush=True)

if __name__=="__main__":build()
