"""Scientific diagram from actual annotations/edges, not a brain photograph."""
from pathlib import Path
import sys,json,numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/(sys.argv[1] if len(sys.argv)>1 else "results/connectome_time_paper_v2")
    dest=out/"figures";dest.mkdir(exist_ok=True)
    n=pd.read_parquet(ROOT/"data/connectome_v783/neurons.parquet")
    e=pd.read_parquet(ROOT/"data/connectome_v783/edges.parquet")
    roles=["ORN","PN","KC","MBON","DAN"];colors=["#258cac","#e39431","#67993f","#ae4893","#cd5550"]
    fig,ax=plt.subplots(1,2,figsize=(13,6))
    for role,color in zip(roles,colors):
        s=n[n.role==role];x=pd.to_numeric(s.pos_x,errors="coerce");y=pd.to_numeric(s.pos_y,errors="coerce")
        ax[0].scatter(x,y,s=3 if role=="KC" else 8,c=color,alpha=.5,label=f"{role}: {len(s):,}")
    ax[0].set_title("Actual annotation reference positions\nNot neuronal morphology or traced axons")
    ax[0].set_xlabel("pos_x (source coordinates)");ax[0].set_ylabel("pos_y (source coordinates)")
    ax[0].legend(fontsize=8);ax[0].set_aspect("equal")
    mat=e.groupby(["pre_role","post_role"]).synapses.sum().unstack(fill_value=0).reindex(index=roles,columns=roles,fill_value=0)
    im=ax[1].imshow(np.log10(mat.to_numpy()+1),cmap="Blues")
    for i in range(5):
        for j in range(5):
            count=int(mat.iloc[i,j]);ax[1].text(j,i,f"{count:,}",ha="center",va="center",fontsize=8,color="white" if count>10000 else "black")
    ax[1].set_xticks(range(5),roles);ax[1].set_yticks(range(5),roles)
    ax[1].set_xlabel("Postsynaptic role");ax[1].set_ylabel("Presynaptic role")
    ax[1].set_title("Measured synapse counts\nAll retained within-subnetwork pairs, including unused edges")
    fig.suptitle("FlyWire FAFB v783 | 8,570 selected neurons | exact root IDs retained",fontsize=13)
    fig.text(.5,.01,"Dorkenwald et al. 2024; Schlegel et al. 2024; source Zenodo 10676866. Connectivity CC BY 4.0.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.93]);fig.savefig(dest/"connectome_anatomy.png",dpi=160);plt.close(fig)
if __name__=="__main__":main()

