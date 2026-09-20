"""Panel-b revision: explicit functional readout, not a reconstructed motor circuit."""
import numpy as np
from scipy.linalg import expm
from .connectome import TemporalMemory

class CascadeMemory(TemporalMemory):
    """Hypothesis: short -> intermediate -> long trace; no time-triggered switch."""
    def __init__(self,n,config,model="cascade",gain=1.):
        super().__init__(n,config,"consolidating",gain)
        self.model="cascade";self.intermediate=np.zeros(n)
    def advance(self,dt,kc,exposure=0.,learning=False,dan_on=True):
        if dt<=0 or not np.isfinite(dt):raise ValueError("dt must be positive")
        c=self.config;tau=c["ethanol_uptake_tau_s"] if exposure else c["ethanol_clearance_tau_s"]
        e0=self.ethanol;ebar=exposure+(e0-exposure)*(-np.expm1(-dt/tau))*tau/dt
        self.ethanol=exposure+(e0-exposure)*np.exp(-dt/tau)
        q=c["acquisition_per_s"]*self.gain*ebar*np.asarray(kc) if learning else np.zeros_like(self.short)
        avdecay=1/(c["aversive_tau_h"]*3600);a=q+avdecay
        self.aversive=self.aversive*np.exp(-a*dt)+q/a*(-np.expm1(-a*dt))
        if c["learning_dan_dependence"] and not dan_on:q=np.zeros_like(q)
        k=1/(c["consolidation_tau_h"]*3600);b=1/(c["long_memory_tau_h"]*3600)
        y=np.stack([self.short,self.intermediate,self.long])
        if not np.any(q):
            y=expm(np.array([[-k,0,0],[k,-k,0],[0,k,-b]])*dt)@y
        else:
            def f(z):return np.stack([q*(1-z[0])-k*z[0],k*z[0]-k*z[1],k*z[1]-b*z[2]])
            # Bound RK4 steps independently of the caller's logging interval.
            count=max(1,int(np.ceil(dt/30)));h=dt/count
            for _ in range(count):
                a1=f(y);a2=f(y+h*a1/2);a3=f(y+h*a2/2);a4=f(y+h*a3)
                y+=h*(a1+2*a2+2*a3+a4)/6
        self.short,self.intermediate,self.long=y
        self.time_s+=dt
        return float(ebar)
    def snapshot(self):
        return {**super().snapshot(),"intermediate":self.intermediate.copy()}

def memory_factory(n,cfg,model,gain=1.):
    return CascadeMemory(n,cfg,gain=gain) if model=="cascade" else TemporalMemory(n,cfg,model,gain)

class FunctionalReadout:
    """Shared graph evaluated on each antenna's odor mixture.
    Counterfactual evaluations are a functional controller, not two anatomical brains.
    Normalize sensory concentration and MBON population gain; keep memory magnitude.
    """
    def __init__(self,graph):
        self.g=graph
        # Linear KC -> MBON population projection, exactly the mean of measured rows.
        self.output_coeff=np.asarray(graph.km.mean(axis=0)).ravel()
        self.app_coeff=np.asarray(graph.km.T@(np.asarray(graph.dm@np.ones(len(graph.ids["DAN"]))).ravel()/len(graph.ids["MBON"]))).ravel()
    def features(self,odor):
        odor=np.asarray(odor,float)
        if odor.shape!=(2,2) or not np.isfinite(odor).all() or np.any(odor<0):raise ValueError("Finite nonnegative odor x antenna input required")
        features=[]
        for antenna in range(2):
            x=odor[:,antenna]
            # Saturating normalization; blank air stays blank.
            normalized=x/(x.sum()+self.g.config.get("adaptation_epsilon",.01))
            features.append(self.g.response(np.repeat(normalized[:,None],2,axis=1))[2])
        return np.array(features)
    def values(self,features,memory,dan_on=True):
        numerator=(features@(self.app_coeff*memory.long))*self.g.config["appetitive_scale"]*float(dan_on)
        numerator-=(features@(self.output_coeff*memory.aversive))*self.g.config["aversive_scale"]
        denom=features@self.output_coeff
        return np.divide(numerator,denom,out=np.zeros(2),where=denom>1e-12)
    def probe(self,memory,identity,dan_on=True):
        odor=np.zeros((2,2));odor[identity]=1
        return float(self.values(self.features(odor),memory,dan_on).mean())

def turn_command(cfg,odor,values,noise):
    if cfg.get("individual_controller_v4",False):
        # Absolute learned-value difference: no division by memory magnitude.
        lateral=cfg["behavioral_gain"]*(values[0]-values[1])
        innate=cfg["innate_value"]*(odor[:,0].sum()-odor[:,1].sum())
        turn=float(np.clip(np.tanh(lateral+innate)+noise,-cfg["max_turn"],cfg["max_turn"]))
        return cfg["base_drive"]*np.array([1-turn,1+turn]),turn
    # Learned signed value adds to innate attraction; negative is not clipped away.
    total=cfg["innate_value"]*np.asarray(odor).sum(axis=0)+cfg["motor_value_gain"]*np.asarray(values)
    contrast=(total[0]-total[1])/(np.abs(total).sum()+cfg.get("motor_denominator_floor",.05))
    turn=float(np.clip(cfg["turn_gain"]*contrast+noise,-cfg["max_turn"],cfg["max_turn"]))
    return cfg["base_drive"]*np.array([1-turn,1+turn]),turn

def make_y_sim(cfg,heading=0.):
    """Tall-walled physical Y, finite collection regions, no open-arena choice line."""
    import mujoco
    from flygym import Simulation
    from flygym.compose import FlatGroundWorld,ActuatorType
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain.common import make_locomotion_fly
    from flygym_demo.complex_terrain.cpg_controller import CPGController,make_tripod_cpg_network
    from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps
    fly=make_locomotion_fly();world=FlatGroundWorld(half_size=120)
    stem=cfg["stem_mm"];arm=cfg["arm_mm"];half=cfg["corridor_width_mm"]/2
    d=arm/np.sqrt(2);v=half/np.sqrt(2)
    polygon=np.array([[-5,-half],[stem-half*(np.sqrt(2)-1),-half],
        [stem+d-v,-d-v],[stem+d+v,-d+v],[stem+half*np.sqrt(2),0],
        [stem+d+v,d-v],[stem+d-v,d+v],[stem-half*(np.sqrt(2)-1),half],[-5,half]])
    for i,(a,b) in enumerate(zip(polygon,np.roll(polygon,-1,axis=0))):
        delta=b-a;mid=(a+b)/2;angle=np.arctan2(delta[1],delta[0])
        world.mjcf_root.worldbody.add_geom(name=f"y_wall_{i}",type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(*mid,5),size=(np.linalg.norm(delta)/2,.25,5),quat=(np.cos(angle/2),0,0,np.sin(angle/2)),
            rgba=(.25,.3,.35,1),contype=2,conaffinity=0)
    # FlyGym body geoms have zero masks and ground contact is explicitly paired.
    # Enable ONLY wall contact via bit 2; do not introduce fly self-collisions.
    for geoms in fly.bodyseg_to_mjcfgeom.values():
        for geom in geoms:geom.conaffinity=2
    world.add_fly(fly,(0,0,.3),Rotation3D("quat",(np.cos(heading/2),0,0,np.sin(heading/2))))
    sim=Simulation(world,timestep=cfg["physics_dt_s"])
    ctl=CPGController(make_tripod_cpg_network(sim.timestep,seed=cfg.get("gait_seed",0)),PreprogrammedSteps(),fly.get_actuated_jointdofs_order(ActuatorType.POSITION))
    sim.reset();ctl.cpg_network.reset()
    return fly,sim,ctl,polygon

def wall_steering(antennae,polygon):
    """Assumed local obstacle reflex, independent of odor identity/reward/time."""
    a=polygon;b=np.roll(polygon,-1,axis=0);v=b-a
    distances=[]
    for point in antennae:
        t=np.clip(np.sum((point-a)*v,axis=1)/np.sum(v*v,axis=1),0,1)
        distances.append(float(np.linalg.norm(point-(a+t[:,None]*v),axis=1).min()))
    left,right=np.maximum(0,4.-np.array(distances))
    # Reflex must be able to override an opposing odor command before leg contact.
    return float(np.clip(6*(right-left),-1.2,1.2))

def collection_side(position,cfg):
    x,y=position[:2];d=x-cfg["stem_mm"]
    along=(d+abs(y))/np.sqrt(2);across=abs(d-abs(y))/np.sqrt(2)
    if along>=cfg["arm_mm"]-cfg["collection_depth_mm"] and across<=cfg["corridor_width_mm"]/2:return 1 if y>0 else -1
    return 0

def upwind_steering(position,heading,cfg):
    """Assumed upstream walking reflex in a symmetric prescribed flow field.
    No odor identity, memory, condition or test delay enters this computation.
    This is not CFD or an imported descending-neuron circuit.
    """
    x,y=position[:2]
    desired=np.sign(y)*np.pi/4 if x>cfg['stem_mm'] and abs(y)>cfg['corridor_width_mm']/4 else 0.
    return float(1.2*np.sin(desired-heading))
