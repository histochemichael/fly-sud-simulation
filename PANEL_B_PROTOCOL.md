# Panel-b V3 protocol and interpretation rules

Scope: a local exploratory physical pilot of early aversion versus late preference. Not panels c/d, not whole-brain modeling, and not a public release.

## Frozen design

Two memory hypotheses: one-stage consolidating and two-stage cascade. Each was selected from a fixed grid using only internal-value sign margins at 0.5 and 24 hours. Those are **calibration targets**, not held-out biological validation. The candidate table and selection policy are in data/connectome_v783/panel_b_v3.

The motor controller was developed with separate value clamps, not the learned-condition outcome counts. Development runs are retained as panel_b_motor_gate_v3 through v3e. V3 failed on a logging schema error; v3b had disabled wall-body contact masks; v3c/d exposed collision-induced navigation stalls. V3e uses explicit contacts, early obstacle avoidance and a symmetric prescribed upstream-walking reflex. Do not pool these runs.

The upstream field follows the stem and each arm. It is a geometry-based controller assumption, not fluid simulation or measured neural circuitry. Neither this reflex nor wall avoidance knows the rewarded odor or retention delay. Memory-dependent steering still comes from sensory input and learned KC traces.

## Physical pilot

Run: results/panel_b_pilot_v3

- 2 models × 3 conditions (paired, unpaired, untrained) × 2 delays (0.5 and 24 h).
- 2 numerical replicate pairs per cell.
- Each replicate has two reciprocal odor groups; each odor group contains two side-counterbalanced flies.
- Total: 96 trials, 8 flies per model/condition/delay. This is **not** the paper's 50 flies per odor group or its N=8 reciprocal pairs.
- Master seed 194901; trial seeds depend on replicate and paired odor. Delays, models, conditions and spatial reversals are matched. The two spatial placements are counterbalanced repeats, not independent biological samples.
- Learning gain is lognormally varied; all flies share the same reconstructed female anatomy and generic odor encoding.
- Three 10-minute odor/10-minute odor+ethanol sessions with 50-minute gaps. Unpaired exposure ordering remains our documented dose-matched adaptation.
- A physical Y-maze replaces the open-arena choice line: 30-mm stem, 60-mm arms, 10-mm corridors and 10-mm-high collision-enabled walls. These dimensions/flat orientation are adaptations, not an exact reconstruction of the paper's apparatus.
- Full assay deadline: 120 seconds. Entry into the final 8 mm of an arm is an **absorbing collection state**. Moving intervals use MuJoCo at 0.1 ms; after collection, only memory and the logical assay clock continue. Collection-hold rows are explicitly marked and are not fabricated body movement.
- The absorbing-vial assumption simplifies real vial entry/exit. The assay should not be called an exact behavioral replication.

## Endpoints and decision rules

For each reciprocal odor group: PI=(paired choices - other choices)/all flies, including nonresponders. CPI is the mean of the two reciprocal group PIs. Physics failures are reported explicitly and invalidate successful-assay claims; they must not be silently removed.

A qualitative panel-b direction match requires paired CPI < 0 at 0.5 h, paired CPI > 0 at 24 h, the corresponding signed differences versus the unpaired control, and valid containment/physics. This is a check of a calibrated model's behavioral expression, not independent confirmation of the memory mechanism.

Report both models, all three conditions, counts, nonresponses and actual latencies regardless of outcome. No selection by favorable p-values. Compare paired/control CPI using seed-matched replicate differences and exact two-sided sign flips, with Holm correction across all reported comparisons. N=2 pairs is very underpowered; a direction match is not statistical reproduction.

Memory-only predictions at 3, 6, 12, 15 and 48 h were excluded from parameter selection. The published timing was already known, so these are held-out-from-fitting checks, not blind discovery. Frozen predictions already show crossings at 3.75 h and 7 h, missing the paper's 12–15 h behavioral transition. An internal value is not a behavioral CPI. Do not retune this run after seeing those values.

## Data and model limitations

Initial, post-training, pre-test and final KC memory arrays are saved with exact root IDs. Training/retention history retains per-neuron memory arrays. Test history stores the pre-test state reference, elapsed time and memory means: full per-KC test states can be reconstructed by the deterministic reward-free evolution equations. This avoids repeating unchanged learning structure thousands of times. Sensors, actions, separate wall/wind turns, random seeds, choices and latencies are stored per control timestep. Every integrated physics step is retained separately.

Replays contain actual moving body poses only; collection holds must not be presented as locomotion. Per-trial summaries identify actual physics duration versus 120-second assay duration. Metadata archives configurations, package versions, source and data hashes.

The readout normalizes sensory concentration, evaluates the shared connectome on each antenna's mixture, and compares learned scalar values. These are counterfactual functional evaluations, **not two anatomically separate fly brains** or reconstructed MBON-to-motor wiring. Plasticity is online associative/reward-modulated learning, not deep learning or a complete model of addiction.

Dopamine scope: the selected graph includes 331 FlyWire DANs. Measured DAN-to-MBON edges contribute to a simplified tonic retrieval gate; exported DAN-to-KC edges do not drive plasticity. The teaching term is directly based on internal ethanol/reward. This is not realistic dopamine-release dynamics or compartment-specific dopamine-dependent learning. The new pilot has paired, unpaired and untrained conditions, not an additional DAN-silencing cohort.

Scientific rationale, not parameter validation: [Kaun et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/); [MBON valence and action selection](https://elifesciences.org/articles/4580); [learned olfactory valence and upwind movement](https://elifesciences.org/articles/85756).

## Reproduction

    python experiments/calibrate_panel_b.py
    python experiments/run_panel_b.py --output results/NEW_PANEL_B --duration 120 --replicates 2 --workers 8
    python experiments/analyze_panel_b.py results/NEW_PANEL_B
    python -m pytest -q

Calibration refuses an existing calibration directory; the saved frozen configuration can be used directly instead. Trial runners refuse existing result directories. Earlier results are not overwritten.
