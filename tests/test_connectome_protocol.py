import numpy as np
from fly_sud.connectome_protocol import paper_schedule,centered_motor_value
from test_connectome import CFG,graph
def test_paper_schedule_total_time_and_dose():
    for condition in CFG["conditions"]:
        epochs=list(paper_schedule(condition,1,CFG))
        assert sum(e[4] for e in epochs)==9600
        dose=sum(e[3]*e[4] for e in epochs)
        assert dose==(0 if condition=="untrained" else 1800)
        for identity in range(2):
            duration=sum(e[4] for e in epochs if e[2][identity].any())
            assert duration==(0 if condition=="untrained" else 1800)
        if condition=="unpaired":assert all(not (e[3] and e[2].any()) for e in epochs)
def test_equal_input_motor_calibration(graph):
    from fly_sud.connectome import TemporalMemory
    m=TemporalMemory(len(graph.ids["KC"]),CFG);m.long[:]=.5;m.aversive[:]=.2
    odor=np.array([[.4,.4],[.7,.7]]);_,_,kc=graph.response(odor)
    v,ref=centered_motor_value(graph,m,odor,kc,np.ones(len(graph.ids["DAN"])))
    np.testing.assert_allclose(v,0.,atol=1e-15)

