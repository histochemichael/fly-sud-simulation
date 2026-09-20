import numpy as np
from test_connectome import CFG,graph
from fly_sud.panel_b import CascadeMemory,FunctionalReadout,collection_side
def test_cascade_retention_semigroup():
    a=CascadeMemory(3,CFG);b=CascadeMemory(3,CFG)
    for m in (a,b):m.short[:]=[.1,.2,.3];m.intermediate[:]=.1;m.long[:]=.05;m.aversive[:]=.2
    a.advance(3600,np.zeros(3))
    for _ in range(60):b.advance(60,np.zeros(3))
    for key in ["short","intermediate","long","aversive"]:np.testing.assert_allclose(getattr(a,key),getattr(b,key),rtol=1e-11)
def test_cascade_requires_experience():
    m=CascadeMemory(3,CFG);m.advance(24*3600,np.ones(3),learning=False)
    assert not any(getattr(m,k).any() for k in ["short","intermediate","long","aversive"])
def test_cascade_learning_and_delayed_transfer():
    m=CascadeMemory(3,CFG);m.advance(600,np.ones(3),exposure=1,learning=True)
    assert m.short.mean()>m.intermediate.mean()>m.long.mean()>0
    before=m.long.copy();m.advance(3600,np.zeros(3));assert np.all(m.long>before)
def test_readout_blank_and_bilateral_symmetry(graph):
    r=FunctionalReadout(graph);m=CascadeMemory(len(graph.ids["KC"]),CFG);m.long[:]=.3
    np.testing.assert_allclose(r.values(r.features(np.zeros((2,2))),m),0)
    v=r.values(r.features(np.ones((2,2))*.2),m);assert v[0]==v[1]
def test_readout_memory_magnitude_not_erased(graph):
    r=FunctionalReadout(graph);m=CascadeMemory(len(graph.ids["KC"]),CFG);f=r.features(np.ones((2,2))*.2)
    m.long[:]=.1;v=r.values(f,m);m.long[:]=.2;np.testing.assert_allclose(r.values(f,m),2*v)
def test_collection_is_end_of_arm_not_fork():
    c=dict(stem_mm=30,arm_mm=60,corridor_width_mm=10,collection_depth_mm=8)
    assert collection_side([30,0,1],c)==0
    assert collection_side([70,40,1],c)==1
    assert collection_side([70,-40,1],c)==-1
    assert collection_side([70,0,1],c)==0

def test_physical_history_schema(tmp_path):
    import json,pandas as pd
    from pathlib import Path
    from fly_sud.panel_b_trial import run_panel_trial
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'data/connectome_v783/panel_b_v3/consolidating.json').read_text())
    cfg['test_duration_s']=.04
    out=tmp_path/'smoke'
    r=run_panel_trial((cfg,str(root/'data/connectome_v783'),'consolidating','clamp_neutral',0.,0,0,True,str(out)))
    h=pd.read_parquet(out/'physics_history.parquet')
    assert r['physics_ticks']==400 and h.shape==(400,6)
    assert h.state_index.between(0,1).all()

def test_wall_collision_masks_and_reflex():
    import json,mujoco
    from pathlib import Path
    from fly_sud.panel_b import make_y_sim,wall_steering
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'data/connectome_v783/panel_b_v3/consolidating.json').read_text())
    fly,sim,ctl,poly=make_y_sim(cfg)
    try:
        m=sim.mj_model
        wall=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'y_wall_0')
        body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,fly.name+'/c_thorax')
        assert int(m.geom_contype[wall]) & int(m.geom_conaffinity[body])
        assert not int(m.geom_contype[body])
        assert wall_steering(np.array([[10.,4.],[10.,3.5]]),poly)<0
        assert wall_steering(np.array([[10.,-3.5],[10.,-4.]]),poly)>0
    finally:sim.close()

def test_upwind_reflex_is_mirror_symmetric():
    from fly_sud.panel_b import upwind_steering
    c=dict(stem_mm=30,corridor_width_mm=10)
    assert upwind_steering([10,0],0,c)==0
    assert upwind_steering([10,0],np.pi/2,c)<0
    assert np.isclose(upwind_steering([40,8],.3,c),-upwind_steering([40,-8],-.3,c))

def test_actual_learned_readout_reverses_without_action_timer():
    import json
    from pathlib import Path
    from fly_sud.connectome import Connectome,train_and_delay
    from fly_sud.panel_b import memory_factory
    root=Path(__file__).resolve().parents[1]
    odor=np.array([[.14,.12],[.12,.14]])
    for model in ['consolidating','cascade']:
        cfg=json.loads((root/f'data/connectome_v783/panel_b_v3/{model}.json').read_text())
        graph=Connectome(root/'data/connectome_v783',cfg);r=FunctionalReadout(graph)
        for delay,sign in [(.5,1),(24.,-1)]:
            m=memory_factory(len(graph.ids['KC']),cfg,model);train_and_delay(graph,m,'paired',1,delay)
            features=r.features(odor);v=r.values(features,m)
            assert (v[0]-v[1])*sign>0
            m.aversive[:]=0;m.long[:]=0
            np.testing.assert_allclose(r.values(features,m),0)
