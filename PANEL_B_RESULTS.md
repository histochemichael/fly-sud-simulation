# Panel-b attempt: physical learned choices, V3

The revised model expresses early avoidance and late approach only to the extent shown by the actual-choice results below. This is a calibrated model's behavioral test, not independent biological replication. Quantitative effect size, time dependence and the remaining figure panels are not reproduced.

## Actual results

| Model | Condition | Delay (h) | Target / other / none | CPI | Reciprocal replicate pairs |
|---|---|---:|---:|---:|---:|
| cascade | paired | 0.5 | 0/8/0 | -1.000 | 2 |
| cascade | paired | 24 | 8/0/0 | +1.000 | 2 |
| cascade | unpaired | 0.5 | 4/3/1 | +0.125 | 2 |
| cascade | unpaired | 24 | 4/4/0 | +0.000 | 2 |
| cascade | untrained | 0.5 | 3/4/1 | -0.125 | 2 |
| cascade | untrained | 24 | 3/4/1 | -0.125 | 2 |
| consolidating | paired | 0.5 | 0/8/0 | -1.000 | 2 |
| consolidating | paired | 24 | 8/0/0 | +1.000 | 2 |
| consolidating | unpaired | 0.5 | 4/4/0 | +0.000 | 2 |
| consolidating | unpaired | 24 | 4/4/0 | +0.000 | 2 |
| consolidating | untrained | 0.5 | 3/4/1 | -0.125 | 2 |
| consolidating | untrained | 24 | 3/4/1 | -0.125 | 2 |

Each cell contains 8 numerical flies: two reciprocal pairs, each containing two odor groups with two side-counterbalanced flies. CPI includes nonresponders. This does not match the paper's group sizes. All flies share one female anatomical reconstruction and generic odor encoding.

Qualitative direction checks: cascade: passed; consolidating: passed. Passing this check is not quantitative or mechanistic replication.

## What this does and does not recreate

- The paper's panel b shows early negative and late positive conditioned preference. Our 0.5/24-hour internal-value signs were calibration targets, so matching their behavioral direction is not a new prediction.
- CPI near -1 or +1 means essentially deterministic choices in this small pilot. Those magnitudes are much stronger than the roughly -0.28 and +0.23 bars visible in the supplied paper figure; those paper values are approximate visual readings, not raw data or fit targets.
- The paper's error bars and significant group comparisons are not reproduced. We have only two numerical replicate pairs per cell; increasing that number alone would not validate our biological assumptions.
- One-stage and cascade memory crossings are consolidating: 3.75 h, cascade: 7 h. Both precede the published 12–15-hour behavioral transition. Internal values and behavior are different endpoints.
- Seven-day persistence (panel c) and shock-resistant seeking with sugar/ethanol/odor/naive controls (panel d) were not run.

[Paper and methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/).

## Statistics

Exact two-sided paired sign-flip tests use reciprocal-replicate CPI differences (symmetric-null assumption), with Holm correction across all 8 reported comparisons. They do not treat individual flies or timesteps as independent animals. Full test results: results/panel_b_pilot_v3/statistics.json.

Minimum raw p=0.5; minimum Holm-adjusted p=1.0. N=2 pairs is insufficient for a convincing statistical reproduction. Zero-variance/small-N cells are shown as actual dots rather than an invented smooth violin.

![Actual reciprocal-pair CPI](results/panel_b_pilot_v3/figures/paired_cpi.png)

![One-stage trajectories](results/panel_b_pilot_v3/figures/consolidating_trajectories.png)

![Cascade trajectories](results/panel_b_pilot_v3/figures/cascade_trajectories.png)

Trajectories are complete moving paths until entry into collection regions, reflected to align the assigned target arm. Collection is an absorbing-vial approximation, explicitly logged, not a plot clipping boundary. Blue is early; orange is late. There is no claim of physical movement during collection holds.

![Memory timing](results/panel_b_pilot_v3/figures/memory_validation.png)

## What changed

The original hemisphere-based steering readout was replaced with concentration-normalized functional odor-value comparison using the measured FlyWire feedforward graph. The same graph is evaluated for each antenna; this is an assumed functional decoder, not two reconstructed brains or a measured descending motor pathway.
Dataset: FlyWire FAFB v783, using 8,570 selected neurons (2,281 ORNs, 685 PNs, 5,177 KCs, 96 MBONs and 331 DANs). The source is the FlyWire Consortium connectivity release, not a synthetic replacement or the full brain: https://zenodo.org/records/10676866.
Dopamine limitation: measured DAN-to-MBON connections contribute to an assumed tonic on/off retrieval gate. DAN-to-KC connections are exported but inactive in the plasticity rule. Ethanol/internal reward drives the associative teaching term directly. No realistic dopamine release, cell-specific DAN dynamics or compartment-specific dopamine-dependent synaptic learning is implemented. This pilot's three conditions do not add a new DAN-silencing cohort.

A second, two-stage consolidation hypothesis was added. Both models learn online from odor/ethanol experience. Neither controller contains a rule that changes the target choice at a specified time. No deep learning or complete addiction model is claimed.

The body walks in an actual collision-enabled flat Y-maze. A reward-independent geometric wall reflex and prescribed upwind-walking reflex support navigation. These reflexes were developed using separate clamps, not tuned to this cohort's CPI. They are engineering approximations, not measured neural circuitry or fluid dynamics.

Important correction to earlier arena interpretations: direct inspection confirmed that the older walls had no explicit body contact pairs and body collision masks were zero. Earlier excursions were not evidence of flies climbing valid walls. Preserve those results as open-arena choice-zone experiments, not confined Y-maze trials. This revision fixes the contacts and separately audits body-center containment.

## Verification and data

Audit: 96 trials, 11,256,249 actual physics rows, 576,000 control/collection rows; 0 physics failures; 0 outside-arena body-center samples with 0.5-mm boundary tolerance.
Nonresponders with less than 1 mm of movement during their final 10 seconds: 5. These are flagged as possible gait/navigation stalls, not interpreted as learned aversion. They remain in CPI denominators. Trial details are in audit.json.

Archived execution hashes verified. Memory means reconstructed from pre-test per-KC states agree within 1.66e-14. Numerical training-step sensitivity (60 versus 10 seconds): consolidating 0.0070%, cascade 0.0040%. Numerical consistency does not establish biological correctness.
Forty regression tests passed, including physical history schema, contact masks, memory retention consistency, and memory-dependent readout reversal. The steering replay audit reproduces actual commands and separately measures their value-dependent contribution; it is not a new closed-loop memory-erasure experiment.

The assay deadline is 120 seconds. MuJoCo integrates moving intervals at 0.1 ms; absorbed flies are held in the collecting state until the assay ends. Metadata distinguishes actual physics time from assay time. Training and retention are stationary state evolution, not simulated hours of walking.

Per fly: learning_history.parquet, state_history.parquet, physics_history.parquet, learning_states.npz with naive/post-training/pre-test/final states and KC IDs, replay.npz, summary.json. Batch metadata stores configuration, software versions, seeds and source/data hashes. CPG initialization uses the library's fixed seed 0 for all flies.

Data folder: results/panel_b_pilot_v3. Separate navigation gate: results/panel_b_motor_gate_v3e. Earlier development failures remain preserved. No upload or new public video was made.

## Next scientific step

Do not tune this completed run. A new, explicitly labeled calibration should address excessive behavioral gain/insufficient individual variability and compare memory mechanisms against independently specified time-course data. Any timing point used for fitting loses held-out status. Then run an independently seeded, larger reciprocal-cohort validation with declared success criteria. A whole-brain expansion is not justified by this pilot alone.

See [frozen protocol and limitations](PANEL_B_PROTOCOL.md) and [previous diagnostics](CONNECTOME_DIAGNOSTICS.md).