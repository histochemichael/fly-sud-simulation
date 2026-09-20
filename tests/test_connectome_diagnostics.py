"""Diagnostic clamps must not masquerade as learning or alter the controller."""
import importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("diagnostic",ROOT/"experiments/diagnose_connectome.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
CFG=json.loads((ROOT/"data/connectome_v783/calibration_paper/consolidating.json").read_text())
def test_neutral_clamp_is_existing_innate_controller():
    odor=np.array([[.3,.2],[.2,.4]])
    drive,v,contrast,turn=module.motor_command(CFG,odor,0,0)
    total=CFG["innate_value"]*odor.sum(axis=0)
    np.testing.assert_array_equal(v,[0,0])
    assert np.isclose(contrast,(total[0]-total[1])/(abs(total).sum()+.05))
    np.testing.assert_allclose(drive,CFG["base_drive"]*np.array([1-turn,1+turn]))
def test_bilateral_equal_input_has_no_direction():
    for valence in [-1,0,1]:
        drive,_,contrast,turn=module.motor_command(CFG,np.full((2,2),.3),valence,0)
        assert contrast==0 and turn==0
        np.testing.assert_allclose(drive,[1,1])
def test_clamped_valence_changes_steering_sign():
    odor=np.array([[.2,.2],[.4,.2]])
    pos=module.motor_command(CFG,odor,1,0)
    neg=module.motor_command(CFG,odor,-1,0)
    assert pos[3]>0 and neg[3]<0
def test_direct_control_bypasses_odors():
    drive,_,_,turn=module.motor_command(CFG,np.zeros((2,2)),0,.5,direct=.3)
    assert turn==.3
    np.testing.assert_allclose(drive,[.7,1.3])
def test_silent_pattern_is_not_declared_identical():
    assert module.cosine(np.zeros(3),np.ones(3)) is None

