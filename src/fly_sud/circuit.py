"""Small rate model with explicit ORN, AL/PN, KC, MBON and DAN variables.

This is a hypothesis circuit with synthetic connectivity, not FlyWire anatomy.
KC-to-MBON plasticity occurs online only while train=True.
"""
import numpy as np


class RewardCircuit:
    def __init__(self, config, seed):
        self.config = config
        rng = np.random.default_rng(seed)
        n = config["kc_count"]
        self.projection = np.zeros((n, 2))
        self.projection[np.arange(n), np.arange(n) % 2] = rng.uniform(.8, 1.2, n)
        self.weights = np.zeros(n)

    def step(self, odor, reward=0., train=False, dt=.01, silence_dan=False):
        # Odor has shape (identity A/B, antenna left/right).
        orn = np.asarray(odor).clip(0)
        pn = orn / (.25 + orn)
        kc = np.maximum(self.projection @ pn - self.config["kc_threshold"], 0.)
        dan = 0. if silence_dan else float(reward if train else self.config["retrieval_dan_gain"])
        if train:
            eligibility = kc.mean(axis=1)
            value = np.dot(self.weights, eligibility) / (eligibility.sum() + 1e-9)
            delta = self.config["learning_rate_per_s"] * dt * dan * (reward - value) * eligibility
            self.weights = np.clip(self.weights + delta, 0., 1.5)
        mbon = (self.weights @ kc) / (kc.sum(axis=0) + 1e-9)
        # Tonic retrieval gating is an explicit modeling assumption.
        value = self.config["innate_value"] * pn.sum(axis=0) + (0 if silence_dan else 1) * mbon
        contrast = (value[0] - value[1]) / (value.sum() + 1e-9)
        activity = {
            "orn_left": float(orn[:,0].mean()), "orn_right": float(orn[:,1].mean()),
            "al_left": float(pn[:,0].mean()), "al_right": float(pn[:,1].mean()),
            "pn_left": float(pn[:,0].mean()), "pn_right": float(pn[:,1].mean()),
            "kc_left": float(kc[:,0].mean()), "kc_right": float(kc[:,1].mean()),
            "mbon_left": float(mbon[0]), "mbon_right": float(mbon[1]), "dan": dan,
            "weight_a": float(self.weights[::2].mean()), "weight_b": float(self.weights[1::2].mean())
        }
        return float(contrast), activity, kc


def training_schedule(condition, config):
    # Unpaired reward is temporally separated from both odor cues.
    # Four equal-duration slots per session match total elapsed time and doses.
    zero = np.zeros((2,2))
    a = np.array([[1.,1.],[0.,0.]])
    b = a[::-1].copy()
    for session in range(1, config["conditioning_sessions"]+1):
        if condition == "untrained":
            slots = [(zero,0.,"naive baseline")]*4
        elif condition == "unpaired":
            slots = [(zero,1.,"ethanol alone"),(zero,0.,"odor-free gap"),(a,0.,"odor A"),(b,0.,"odor B")]
        else:
            slots = [(a,0.,"odor A"),(zero,0.,"odor-free gap"),(b,1.,"odor B + ethanol"),(zero,0.,"air")]
        for trial,(odor,ethanol,label) in enumerate(slots,1):
            yield session,trial,odor,ethanol,label
