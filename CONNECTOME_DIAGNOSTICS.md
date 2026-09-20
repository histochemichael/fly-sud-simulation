# Connectome diagnostics: perception, motor decoding, and memory timing

Completed locally on 2026-09-14. Existing experiments and production-model parameters were preserved. No new public upload.

## Bottom line

The failure does **not** establish that a whole-brain model is required. The diagnostic evidence localizes two actionable problems: the learned signal has little influence on steering, and the assumed memory kinetics switch sign too early. Basic odor encoding is present, and the physical body can approach or avoid a cue when supplied with a strong diagnostic value signal. This is a diagnosis of this implementation, not proof that the selected biological subnetwork is sufficient.

## 1. Perception before choice

Reanalyzed all 240 trials from the corrected connectome pilot, retaining only test control samples up to first choice (or the full test for nonresponders): 30,062 samples.

- Every sample had active PN and KC populations; the minimum was 164 active KCs and the median was 777. A completely silent sensory pathway is ruled out for these recorded inputs.
- Pure A and B inputs at concentrations 0.20–1.00 on the tested grid were distinguishable by their full-strength training KC prototypes. At 0.05, 0.10, and 0.15, both pure odors fell below the KC threshold. This is a real low-concentration limitation, not the explanation for silence in the recorded mixed-odor arena, which was not silent.
- At full strength, A/B prototype cosine similarity was 0.0458. These are assumed generic odor encodings, not measured responses to the paper's chemicals.
- Swapping A/B at the recorded bilateral sensor intensities in 1,615 sampled pre-choice states changed the KC pattern in every sampled case. Median original-versus-swapped cosine similarity was 0.9935; median relative pattern difference was 0.1148. Mixtures often remain very similar. This intervention demonstrates sensitivity, not a validated behavioral odor classifier.
- For paired trials, the mean fraction of retained aversive-plus-long memory magnitude on currently active KCs was about 63–64%. It was therefore not entirely inaccessible during retrieval.
- The largest learned centered motor value was 0.00001527. More importantly, the median absolute learned contribution to the *pre-clipping turn command* in paired trials was only 0.00254 (consolidating) or 0.00140 (decay-only), versus configured stationary motor-noise SD 0.12. The maximum across all conditions was 0.02455. This is a scale comparison, not an independent significance test; it does not account for noise autocorrelation or all closed-loop effects.

**Interpretation:** Sensory information exists, but concentration dependence, mixed-odor similarity, and especially the weak/assumed sensory-to-motor readout require attention. Do not describe this as a validated physiological odor pathway.

## 2. Motor decoding with learning disabled

Ran 56 new physical MuJoCo/FlyGym trials:

- 8 fixed seeds × 2 counterbalanced placements of odor B × 3 modes = 48 trials.
- 4 of those seeds × 2 constant-turn signs = 8 direct-steering controls.
- Same two-second duration, geometry, sensor noise, motor noise, CPG gait, first-choice rule, and continued post-choice logging as the prior corrected experiment.
- Approach injected +odor-B concentration at each antenna into the value input; avoidance injected its negative; neutral injected zero. The existing gain, normalization and steering formula were retained. Direct controls used turns +0.3 or -0.3.
- No ethanol, teaching signal, or learning occurred. These clamps bypass the connectome decoder. The saved zero initial/final memory vectors document a disabled state; they are not evidence of learned synaptic activity.

| Diagnostic mode | Trials / unique seeds | B target choices | Other choices | No choice by 2 s |
|---|---:|---:|---:|---:|
| Approach | 16 / 8 | 14 | 0 | 2 |
| Neutral | 16 / 8 | 8 | 8 | 0 |
| Avoid | 16 / 8 | 3 | 11 | 2 |

All 56 trials completed without the runner's physics-failure flag. Direct positive and negative turn controls moved to their corresponding lateral sides (mean final y +6.25 and -5.14 mm); neither direct-turn group reached the forward choice criterion. This checks actuator sign, not successful odor navigation.

For continuous movement, target-signed final lateral position was averaged across both placements within each seed. Relative to neutral, approach shifted +9.20 mm (seed-bootstrap 95% interval 8.56 to 9.92), and avoidance shifted -13.16 mm (-14.43 to -11.52). Approach minus avoidance was +22.36 mm (20.41 to 24.09).

Each of the three prespecified comparisons had two-sided exact paired sign-flip p=0.0078125, Holm-adjusted p=0.0234375 across three tests; n=8 seed blocks, not 16 independent animals. The sign-flip test assumes a symmetric null distribution of paired differences. Bootstrap intervals use 10,000 seed-block resamples, seed 92071, and are not multiplicity-adjusted. These exploratory numerical checks do not support biological population inference.

**Interpretation:** Strong diagnostic values can drive directional approach and avoidance in this body/controller. Avoidance is not perfect, and a final position is not the same endpoint as first choice. This experiment does not validate the connectome-derived decoder or learned behavior.

### Full trajectories and actual-result statistics

![Complete motor trajectories](results/connectome_diagnostics_v1/figures/motor_trajectories.png)

All recorded trajectories continue after choice; orange endpoints are actual final positions. Negative-target placements are reflected to align B on the positive lateral side. No path is trimmed at a choice boundary. The dotted forward line is only one component of the choice rule (x >= 12 mm AND absolute y >= 2.5 mm), not a wall. The arena's low walls are not impenetrable: recorded flies can walk beyond them. Neither plots nor statistics conceal those excursions.

![Seed-block violin and dots](results/connectome_diagnostics_v1/figures/motor_violin_dots.png)

Dots are eight seed averages over the two placements. Thin lines pair seeds. Violin shapes are exploratory smoothing with a very small sample, not additional observations.

## 3. Memory without movement

Both previously calibrated configurations were frozen. Each model received the same paired training and fixed full-strength odor probes during retention; no arena movement or changing sensory input influenced the probe. Training used the same online reward-modulated rule as before.

| Frozen model | First positive internal value sample | Independent retention ODE maximum absolute error |
|---|---:|---:|
| Consolidating | 1.50 h | 2.93e-13 |
| Decay-only | 2.50 h | 2.51e-13 |

Sampling resolution was 0.25 h. An independent adaptive ODE solver checked the mean retention traces from the post-training state; it did not independently validate acquisition, sensory encoding, plasticity assumptions, or biology.

Kaun et al. reported a behavioral aversion-to-preference transition between 12 and 15 hours after conditioning. Neither frozen model matches that held-out timing target. Also, an internal value crossing is not itself a behavioral choice transition. The 0.5/24-hour value signs were calibration targets, not successful independent reproduction. [Paper, results and Supplementary Figure 1c reference](https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/).

![Frozen memory timing](results/connectome_diagnostics_v1/figures/memory_timing.png)

**Interpretation:** The checked retention equations are being integrated correctly, but the chosen kinetics fail this timing comparison. This is evidence against the current parameterized mechanism, not evidence that adding all remaining neurons would solve it.

## Recommended next implementation — not performed here

1. Replace the assumed bilateral MBON averaging with a justified, sign-checked behavioral readout. Diagnose normalization and concentration robustness before adjusting gain; establish a working signal range with matched approach/avoidance controls. Do not call a gain increase biological validation.
2. Compare explicit alternative aversive/appetitive retention or consolidation mechanisms against independently specified time-course targets. If the 12–15-hour transition becomes a fitting target, reserve other delays or manipulations for validation.
3. Repeat the frozen closed-loop experiment only after those components meet declared criteria. Address the short observation window and escapable arena walls explicitly. Expand the connectome only to test a specific missing-circuit hypothesis, not because the current result is negative.

These recommendations do not authorize or implement a new model revision. No whole-brain simulation, refitting, or new video was performed in this diagnostic run.

## Reproducibility and files

Run: results/connectome_diagnostics_v1. Hardware: Intel Core i7-9700F, 8 cores / 8 logical processors, 8 worker processes. MuJoCo physics ran on CPU; no GPU acceleration was requested.

- metadata.json: exact two model configurations, package versions, seeds, source hashes and prior-config hash.
- source_snapshot.zip: execution source for the diagnostic runner and model.
- perception_prechoice.parquet, perception_identity_swaps.csv, perception_concentration_probes.csv, perception_summary.csv.
- memory_history.parquet, memory_timecourse.csv, memory_checks.json and per-model naive/post-training/final NPZ states with KC root IDs.
- motor_choices.csv, motor_summary.csv, motor_statistics.json; per-trial control/physics Parquet, replay poses, unchanged initial/final states and summary JSON.
- audit.json: 56 physical trials, 1,120,000 physics rows and 11,200 control rows; time ordering, state joins, finite positions and disabled learning checked.
- Thirty regression tests passed, including five new diagnostic-control tests.

The sensory analysis reuses prior result histories; it is not 240 new trials. The memory experiment is a deterministic model check, not a new animal cohort. All simulations share one reconstructed anatomy. New outputs remain local.

Reproduce in an installed project environment:

    python experiments/diagnose_connectome.py --output results/NEW_DIAGNOSTIC_RUN --workers 8
    python experiments/analyze_connectome_diagnostics.py results/NEW_DIAGNOSTIC_RUN
    python -m pytest -q

The runner refuses an existing output directory. The analysis regenerates only its derived summaries/figures.

Related: [connectome methods](CONNECTOME_VERSION.md), [original corrected pilot results](CONNECTOME_RESULTS.md), [FlyWire v783 data](https://zenodo.org/records/10676866).

