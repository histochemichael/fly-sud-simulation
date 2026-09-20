"""Paper paired schedule; separately documented dose-matched unpaired control."""
import numpy as np
def paper_schedule(condition,paired,c):
    zero=np.zeros((2,2));minus=zero.copy();plus=zero.copy()
    minus[1-paired]=1;plus[paired]=1
    for session in range(1,4):
        if condition=="untrained":minus_use=zero;plus_use=zero
        else:minus_use=minus;plus_use=plus
        reward=0. if condition in ("untrained","unpaired") else 1.
        yield session,1,minus_use,0.,600.,"unpaired odor"
        yield session,2,plus_use,reward,600.,"target odor + ethanol" if reward else "target odor without ethanol"
        if session<3:
            # Three equal 600-s ethanol exposures moved into the two 3000-s
            # gaps for the unpaired control (2 in gap 1, 1 in gap 2).
            # This unpaired ordering is our control design, not quoted from Kaun.
            for slot in range(5):
                r=float(condition=="unpaired" and ((session==1 and slot in (0,2)) or (session==2 and slot==0)))
                yield session,3+slot,zero,r,600.,"unpaired ethanol" if r else "inter-session air"

def centered_motor_value(graph,memory,odor,kc,dan):
    # Remove fixed graph/decoder asymmetry at equal bilateral sensory input.
    # This is an assumed controller calibration, not an anatomical motor circuit.
    actual=graph.decode(kc,memory,dan)[0]
    equal=np.repeat(np.asarray(odor).mean(axis=1)[:,None],2,axis=1)
    _,_,reference_kc=graph.response(equal)
    reference=graph.decode(reference_kc,memory,dan)[0]
    return actual-reference,reference

