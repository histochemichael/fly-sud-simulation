"""FlyWire-measured feedforward structure with explicitly assumed memory dynamics.

Rates are dimensionless steady-state responses, not spikes. Memory traces are
per-KC presynaptic gain factors on the measured KC->MBON synapses. The same
anatomical neuron is never duplicated for the two antennae.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy import sparse

def normalize_rows(m):
    totals=np.asarray(m.sum(axis=1)).ravel()
    return sparse.diags(1/np.maximum(totals,1))@m

class Connectome:
    def __init__(self,path,config):
        self.path=Path(path);self.config=config
        self.nodes=pd.read_parquet(self.path/"neurons.parquet")
        self.roles={role:self.nodes[self.nodes.role==role].reset_index(drop=True)
                    for role in ["ORN","PN","KC","MBON","DAN"]}
        self.ids={role:n.root_id.to_numpy(np.int64) for role,n in self.roles.items()}
        self.op=normalize_rows(sparse.load_npz(self.path/"ORN_PN_counts.npz"))
        self.pk=normalize_rows(sparse.load_npz(self.path/"PN_KC_counts.npz"))
        self.km=normalize_rows(sparse.load_npz(self.path/"KC_MBON_counts.npz"))
        self.dk=normalize_rows(sparse.load_npz(self.path/"DAN_KC_counts.npz"))
        self.dm=normalize_rows(sparse.load_npz(self.path/"DAN_MBON_counts.npz"))
        self.sides={role:np.array([0 if s=="left" else 1 if s=="right" else -1 for s in n.side])
                    for role,n in self.roles.items()}
        # These are generic odor patterns, not measured responses to the paper's chemicals.
        # Matching ORN types receive the same tuning on either anatomical side.
        orn=self.roles["ORN"];self.tuning=np.zeros((len(orn),2))
        for i,row in orn.iterrows():
            label=row.hemibrain_type or row.cell_type or "untyped_ORN"
            seed=int.from_bytes(hashlib.sha256((str(config["odor_encoding_seed"])+label).encode()).digest()[:8],"little")
            rng=np.random.default_rng(seed)
            self.tuning[i]=(rng.random(2)<config["odor_sparsity"])*rng.uniform(.7,1,2)
    def response(self,odor):
        odor=np.asarray(odor,float)
        if odor.shape!=(2,2) or not np.isfinite(odor).all():raise ValueError("Expected finite (odor, antenna) input")
        sides=self.sides["ORN"]
        inputs=odor[:,np.maximum(sides,0)].T.copy()
        inputs[sides<0]=odor.mean(axis=1)
        orn=np.sum(self.tuning*inputs,axis=1)
        pn=np.maximum(self.op@orn,0)
        drive=np.maximum(self.pk@pn,0)
        # Fixed odor-independent threshold; subtractive competition is an assumed
        # APL-like approximation, NOT an imported recurrent APL circuit.
        threshold=max(self.config["kc_threshold"],float(np.quantile(drive,self.config["kc_sparsity_quantile"])))
        kc=np.maximum(drive-threshold,0)
        return orn,pn,kc
    def decode(self,kc,memory,dan):
        app=np.asarray(self.km@(kc*memory.long)).ravel()
        av=np.asarray(self.km@(kc*memory.aversive)).ravel()
        gate=np.asarray(self.dm@dan).ravel()
        app*=gate
        values=self.config["appetitive_scale"]*app-self.config["aversive_scale"]*av
        side=np.zeros(2)
        for i in range(2):
            mask=self.sides["MBON"]==i
            side[i]=values[mask].mean() if mask.any() else 0.
        return side,app,av,gate

class TemporalMemory:
    def __init__(self,n,config,model="consolidating",gain=1.):
        if model not in ("consolidating","decay_only"):raise ValueError("Unknown model")
        self.config=config;self.model=model;self.gain=gain
        self.aversive=np.zeros(n);self.short=np.zeros(n);self.long=np.zeros(n)
        self.ethanol=0.;self.time_s=0.
    def advance(self,dt,kc,exposure=0.,learning=False,dan_on=True):
        if dt<=0 or not np.isfinite(dt):raise ValueError("dt must be positive")
        c=self.config;kc=np.asarray(kc)
        tau=c["ethanol_uptake_tau_s"] if exposure else c["ethanol_clearance_tau_s"]
        e0=self.ethanol;target=float(exposure)
        ebar=target+(e0-target)*(-np.expm1(-dt/tau))*tau/dt
        self.ethanol=target+(e0-target)*np.exp(-dt/tau)
        # Teaching uses interval-average internal ethanol; exact for the one-state
        # uptake/clearance model. Rate is frozen per numerical interval.
        q=c["acquisition_per_s"]*self.gain*ebar*kc if learning else np.zeros_like(kc)
        avdecay=1/(c["aversive_tau_h"]*3600);a=q+avdecay
        self.aversive=self.aversive*np.exp(-a*dt)+(q/a)*(-np.expm1(-a*dt))
        if c["learning_dan_dependence"] and not dan_on:q=np.zeros_like(q)
        b=1/(c["long_memory_tau_h"]*3600)
        if self.model=="decay_only":
            q=q*c["decay_only_appetitive_scale"]
            self.long=self.long*np.exp(-(q+b)*dt)+q/(q+b)*(-np.expm1(-(q+b)*dt))
        else:
            k=1/(c["consolidation_tau_h"]*3600);a=q+k
            s0=self.short.copy();sinf=q/a
            ea=np.exp(-a*dt);eb=np.exp(-b*dt)
            # Exact constant-input solution to a coupled first-order S -> L system.
            cross=np.divide(ea-eb,b-a,out=np.full_like(a,dt*eb),where=np.abs(b-a)>1e-12)
            self.long=self.long*eb+k*(sinf*(-np.expm1(-b*dt))/b+(s0-sinf)*cross)
            self.short=sinf+(s0-sinf)*ea
        self.time_s+=dt
        return ebar
    def snapshot(self):
        return {"aversive":self.aversive.copy(),"short":self.short.copy(),"long":self.long.copy(),
                "ethanol":float(self.ethanol),"biological_time_s":float(self.time_s)}

def schedule(condition,paired_odor,config):
    if condition not in config["conditions"]:raise ValueError("Unknown condition")
    if config.get("protocol")=="paper_paired":
        from .connectome_protocol import paper_schedule
        yield from paper_schedule(condition,paired_odor,config)
        return
    zero=np.zeros((2,2));a=zero.copy();a[1-paired_odor]=1;b=zero.copy();b[paired_odor]=1
    # Every condition has the same total duration and slots. Unpaired exposure
    # precedes the odors by one hour to minimize pharmacokinetic overlap.
    for s in range(1,config["conditioning_sessions"]+1):
        if condition=="untrained":slots=[(zero,0.,600.,"air"),(zero,0.,3600.,"gap"),(zero,0.,600.,"air"),(zero,0.,600.,"air")]
        elif condition=="unpaired":slots=[(zero,1.,600.,"ethanol alone"),(zero,0.,3600.,"gap"),(a,0.,600.,"unpaired odor"),(b,0.,600.,"target odor without ethanol")]
        else:slots=[(zero,0.,600.,"air"),(zero,0.,3600.,"gap"),(a,0.,600.,"unpaired odor"),(b,1.,600.,"target odor + ethanol")]
        # Explicit matched-duration experimental adaptation, not the exact paper schedule.
        for trial,(odor,ethanol,duration,label) in enumerate(slots,1):
            duration=config["training_interval_s"] if label=="gap" else config["training_exposure_s"]
            yield s,trial,odor,ethanol,duration,label
        if s<config["conditioning_sessions"]:
            yield s,5,zero,0.,config["training_interval_s"],"inter-session interval"

def train_and_delay(graph,memory,condition,paired_odor,delay_h,emit=None):
    cfg=memory.config;naive=memory.snapshot()
    def block(dt,odor,exposure,phase,session,trial,label,train):
        orn,pn,kc=graph.response(odor)
        biological_step=cfg["biological_step_s"] if exposure or np.any(odor) else cfg.get("quiet_step_s",600.)
        elapsed=0.
        while elapsed<dt-1e-9:
            step=min(biological_step,dt-elapsed)
            mean_ethanol=memory.advance(step,kc,exposure,train,not(condition=="dan_acquisition" and train))
            if emit:emit(memory,phase,session,trial,label,odor,orn,pn,kc,mean_ethanol,step)
            elapsed+=step
    for session,trial,odor,ethanol,duration,label in schedule(condition,paired_odor,cfg):
        block(duration,odor,ethanol,"training",session,trial,label,True)
    learned=memory.snapshot()
    block(delay_h*3600,np.zeros((2,2)),0.,"delay",0,0,"reward-free retention",False)
    return naive,learned,memory.snapshot()
