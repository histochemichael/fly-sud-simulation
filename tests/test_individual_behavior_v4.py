import json
from pathlib import Path
import numpy as np
from fly_sud.panel_b import turn_command
from fly_sud.panel_b_trial import run_panel_trial

ROOT=Path(__file__).resolve().parents[1]

def config():
 c=json.loads((ROOT/"data/connectome_v783/panel_b_v3/consolidating.json").read_text())
 return {**c,"individual_controller_v4":True,"behavioral_gain":20.,"master_seed":410723,
         "motor_noise_sd":.18,"test_duration_s":.04}

def test_absolute_value_gain_preserves_memory_magnitude():
 c=config();odor=np.full((2,2),.2)
 _,small=turn_command(c,odor,np.array([.002,0]),0)
 _,large=turn_command(c,odor,np.array([.004,0]),0)
 assert 0<small<large<c["max_turn"]
 _,opposite=turn_command(c,odor,np.array([0,.002]),0)
 assert np.isclose(opposite,-small)
 _,blank=turn_command(c,odor,np.zeros(2),0)
 assert blank==0

def test_cpi_sem_uses_reciprocal_pairs_not_individual_flies():
 import sys,pandas as pd
 sys.path.insert(0,str(ROOT/"experiments"))
 from analyze_individual_behavior_v4 import aggregates,exact_signflip
 rows=[]
 for rep,groups in enumerate([[[1,1],[-1,-1]],[[1,-1],[1,1]]]):
  for odor,scores in zip("AB",groups):
   for score in scores:
    rows.append(dict(condition="paired",delay_h=.5,replicate=rep,paired_odor=odor,
       score=score,captured=True,learning_gain=1.,neural_value_difference=.2,choice_latency_s=5.))
 pairs,summary=aggregates(pd.DataFrame(rows))
 np.testing.assert_allclose(pairs.cpi,[0.,.5])
 assert summary.iloc[0]["count"]==2 and summary.iloc[0].flies==8
 assert summary.iloc[0]["mean"]==.25 and summary.iloc[0]["sem"]==.25
 assert exact_signflip([1]*8)==2/256
 assert exact_signflip([0]*8)==1

def test_individual_seed_reproducibility_and_no_cohort_clones(tmp_path):
 c=config()
 def run(out,condition="paired",side=False):
  return run_panel_trial((c,str(ROOT/"data/connectome_v783"),"consolidating",condition,.5,0,0,side,str(out)))
 a=run(tmp_path/"a");b=run(tmp_path/"b");other=run(tmp_path/"c",side=True);ctrl=run(tmp_path/"d","untrained")
 assert a==b
 assert len({a["seed"],other["seed"],ctrl["seed"]})==3
 assert len({a["gait_seed"],other["gait_seed"],ctrl["gait_seed"]})==3
 import pandas as pd
 pd.testing.assert_frame_equal(pd.read_parquet(tmp_path/"a/physics_history.parquet"),pd.read_parquet(tmp_path/"b/physics_history.parquet"))
 states=pd.read_parquet(tmp_path/"a/state_history.parquet")
 assert states.fly_id.eq(a["fly_id"]).all()
 assert a["choice"]=="none" and a["physics_ticks"]==400
