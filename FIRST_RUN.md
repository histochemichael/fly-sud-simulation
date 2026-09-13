# First experiment: seed 42, N=100 per condition

The first full run used the checked-in default parameters and contained no reward during the Y-maze test.

V1 learning is online and reward-modulated: each fly's associative state changes during its discrete conditioning exposures. It is not deep learning and is not a biologically complete addiction model.

| Condition | B choices | Preference index | Completion rate | Mean latency (s) |
|---|---:|---:|---:|---:|
| Untrained | 53/100 | 0.06 | 0.74 | 18.967 |
| Paired ethanol reward | 100/100 | 1.00 | 0.93 | 17.982 |
| Unpaired reward | 60/100 | 0.20 | 0.62 | 18.787 |

Two-sided exact binomial tests against chance gave p=0.617 for untrained, p=1.58e-30 for paired, and p=0.0569 for unpaired. Fisher exact comparisons gave p=1.11e-17 for paired versus untrained and p=1.34e-14 for paired versus unpaired.

## Interpretation

The reward-modulated memory rule learned a large odor-B association only during predictive pairing. That stored association subsequently biased movement through bilateral odor sensing and closed-loop steering; there is no `if trained: choose B` rule and no reward during the test.

The untrained result is consistent with chance. The unpaired result is a marginal upward fluctuation at this seed, not strong evidence of an association. Odor-B side was reasonably balanced in every condition, and the paired result occurred for both left and right placement.

## Limitations exposed by this run

- The paired response saturated at 100%, indicating that this V1 learning/steering combination is too strong to estimate a graded biological effect size.
- Completion rates ranged from 62% to 93%. Timeout trials retain their final lateral tendency as a fallback choice and are explicitly marked `completed=false`; future work should add physical Y-maze walls and a splitter that forces a completed arm entry.
- The statistically efficient batch backend is planar and kinematic. The separate FlyGym test verifies the 69-segment MuJoCo body, but the N=300 conditioning batch is not yet driven through that musculoskeletal model.
- The ethanol state, reward signal, and associative weights are abstractions. There is no ethanol pharmacology, 24-hour consolidation, mushroom-body circuitry, or DAN perturbation in V1.

The recommended V2 step is to connect the learned olfactory value signal to FlyGym's turning controller, add explicit Kenyon-cell/MBON/DAN populations, and test whether simulated DAN silencing abolishes preference without impairing baseline locomotion or odor detection.
