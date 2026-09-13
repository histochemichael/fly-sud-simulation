import json
from pathlib import Path
import numpy as np
from fly_sud.circuit import RewardCircuit, training_schedule

CFG=json.loads((Path(__file__).parents[1]/"config/physical.json").read_text())

def train(condition):
    c=RewardCircuit(CFG,19)
    for _,_,odor,r,_ in training_schedule(condition,CFG):
        for _ in range(round(CFG["training_exposure_s"]/CFG["control_dt_s"])):
            c.step(odor,r,True,CFG["control_dt_s"])
    return c

def test_online_pairing_and_temporal_unpairing():
    p=train("paired");u=train("unpaired")
    assert p.weights[1::2].mean()>0 and np.all(p.weights[::2]==0)
    assert np.all(u.weights==0)

def test_odor_identity_swap_reverses_learned_identity():
    c=RewardCircuit(CFG,19)
    for _ in range(75): c.step(np.array([[1.,1.],[0.,0.]]),1.,True,.01)
    assert c.weights[::2].mean()>0 and np.all(c.weights[1::2]==0)

def test_reward_free_retrieval_preserves_memory():
    c=train("paired");before=c.weights.copy()
    for _ in range(100): c.step(np.array([[.3,.3],[.4,.6]]),0.,False)
    np.testing.assert_array_equal(c.weights,before)

def test_bilateral_steering_and_silencing():
    c=train("paired");s=np.array([[.4,.6],[.6,.4]])
    normal,_,_=c.step(s,0.,False)
    silenced,_,_=c.step(s,0.,False,silence_dan=True)
    mirrored,_,_=c.step(s[:,::-1],0.,False)
    assert normal>0 and abs(silenced)<1e-12
    assert np.isclose(mirrored,-normal)
