import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fly_sud.connectome import Connectome,TemporalMemory,train_and_delay,normalize_rows
from scipy import sparse
ROOT=Path(__file__).parents[1]
CFG=json.loads((ROOT/"config/connectome_time.json").read_text())
DATA=ROOT/"data/connectome_v783"
@pytest.fixture(scope="module")
def graph():
    if not (DATA/"neurons.parquet").exists():pytest.skip("Run official data importer")
    return Connectome(DATA,CFG)
def test_normalization_preserves_sparse_zero_rows():
    m=normalize_rows(sparse.csr_matrix([[1.,3.],[0.,0.]]))
    np.testing.assert_allclose(m.toarray(),[[.25,.75],[0.,0.]])
def test_ids_and_real_edges(graph):
    assert sum(len(x) for x in graph.ids.values())==8570
    assert all(x.dtype==np.dtype("int64") for x in graph.ids.values())
    n=pd.read_parquet(DATA/"neurons.parquet");e=pd.read_parquet(DATA/"edges.parquet")
    assert e.pre.isin(n.root_id).all() and e.post.isin(n.root_id).all()
    assert (e.synapses>=5).all() and not e.duplicated(["pre","post"]).any()
def test_odor_response_uses_measured_pathway(graph):
    _,_,kc=graph.response(np.array([[1.,1.],[0.,0.]]))
    assert 0<np.count_nonzero(kc)<len(kc)
    original=graph.pk
    try:
        graph.pk=sparse.csr_matrix(original.shape)
        _,_,zero=graph.response(np.ones((2,2)))
        assert not zero.any()
    finally:graph.pk=original
def test_memory_does_not_form_without_experience():
    m=TemporalMemory(4,CFG);m.advance(86400,np.ones(4),learning=False)
    assert not m.long.any() and not m.aversive.any()
def test_exact_retention_semigroup():
    a=TemporalMemory(4,CFG);b=TemporalMemory(4,CFG)
    for m in [a,b]:
        m.short[:]=[.1,.2,.3,.4];m.long[:]=.1;m.aversive[:]=.7;m.ethanol=.5
    a.advance(86400,np.zeros(4))
    for _ in range(144):b.advance(600,np.zeros(4))
    for key in ["short","long","aversive"]:np.testing.assert_allclose(getattr(a,key),getattr(b,key),rtol=1e-11,atol=1e-14)
    assert np.isclose(a.ethanol,b.ethanol)
def test_aversive_decay_and_consolidation():
    m=TemporalMemory(2,CFG);m.short[:]=.5;m.aversive[:]=.5
    m.advance(6*3600,np.zeros(2))
    assert (m.long>0).all() and (m.aversive<.5).all()
def test_dan_dependence_is_an_explicit_assumption():
    cfg={**CFG,"learning_dan_dependence":True}
    m=TemporalMemory(2,cfg);m.advance(600,np.ones(2),exposure=1,learning=True,dan_on=False)
    assert m.aversive.max()>0 and not m.short.any()
def test_retention_duration_and_naive_snapshot(graph):
    m=TemporalMemory(len(graph.ids["KC"]),CFG)
    naive,learned,retained=train_and_delay(graph,m,"paired",1,.5)
    assert not naive["long"].any() and not naive["aversive"].any()
    assert retained["biological_time_s"]-learned["biological_time_s"]==1800
def test_retrieval_silencing_is_appetitive_only(graph):
    m=TemporalMemory(len(graph.ids["KC"]),CFG);m.long[:]=.5;m.aversive[:]=.2
    _,_,kc=graph.response(np.ones((2,2)))
    v,a,b,_=graph.decode(kc,m,np.ones(len(graph.ids["DAN"])))
    vs,az,bz,_=graph.decode(kc,m,np.zeros(len(graph.ids["DAN"])))
    assert not az.any();np.testing.assert_array_equal(b,bz)
def test_bad_model_or_time_rejected():
    with pytest.raises(ValueError):TemporalMemory(2,CFG,"unknown")
    with pytest.raises(ValueError):TemporalMemory(2,CFG).advance(-1,np.zeros(2))

