# Physical experiment: local release candidate

This is a separate experiment from `results/first_experiment_seed42`. Do not pool their trials. No data have been uploaded to Hugging Face.

## Scope

All 400 test trials use the FlyGym/NeuroMechFly body and MuJoCo contact dynamics. Circuit output modulates the six-leg gait controller. The three representative flies in `results/physical_pilot` also walk during training; batch conditioning uses stationary uniform odor exposures with the same online circuit updates. The representatives were selected as fly 000, not selected for successful choices. The untrained representative is a nonresponse.

The circuit is an explicit, compact ORN/AL-PN/KC/MBON/DAN model. It uses **online associative/reward-modulated learning during experience**, not deep learning, FlyWire connectivity, whole-brain simulation, biological brain imaging, or a complete model of addiction. AL and PN are collapsed rate transforms. There are 24 synthetic KC identities with bilateral rates and shared learned weights. Ethanol is a scheduled scalar reward, not pharmacokinetics. Retrieval dopamine gating is an explicit modeling assumption.

## Layout and joins

`results/physical_batch/choices.csv` contains one outcome per fly. `summary.csv` retains nonresponses and reports preference both among responders and over all flies. `statistics.json` contains unadjusted exploratory tests. `config_snapshot.json` and `metadata.json` preserve configuration, software versions, source hashes and seed information. `validation.json` records invariant checks.

Each condition/fly directory contains:

- `state_history.parquet`: each 0.01-second controller update, phase/session/trial/event, bilateral A/B odor readings, pose, ethanol/teaching signal, KC weights and activity, MBON/DAN rates, motor commands, seed, final choice and latency.
- `physics_history.parquet`: every 0.0001-second MuJoCo tick during physical motion, with actual pose and `state_index`. Join to that fly's state table on `state_index` for the held sensor/neural/action values. Values are not freshly sampled between control updates. Condition/fly identity comes from the directory and joined state table.
- `learning_states.npz`: initial `naive`, post-training `learned`, post-test `final` weights and fixed `projection` matrix. Arrays are readable with NumPy without pickle.
- `summary.json`: outcome, seeds, source side, latency, movement, counts and whether training was physical.
- `replay.npz` when recorded: body qpos and linked state indices for faithful offline rendering.

Coordinates are millimeters; heading is radians; time is simulated seconds. Rates, odor and reward are dimensionless model units. A/B identities are not specified biological odorants. `choice_latency_s` is null for nonresponders, not a zero-time choice. `final_choice` is a retrospective outcome and must be excluded from predictive training features to avoid label leakage. `training_gap_s` in the frozen config is reserved/unused: each of the four schedule slots actually uses `training_exposure_s` (0.25 s). This is a compressed protocol, not a time-calibrated paper replication.

## Suggested future dataset packaging

Keep one immutable experiment/version per release, with config, source hashes and provenance. Flatten condition/fly IDs into physics shards or retain explicit foreign keys. Partition by experiment/condition/phase; never randomly split timesteps from one fly across train/test. Keep paired and matched dopamine-silenced seeds in the same split. Add a dataset card, license review and checksums before any public release. These files are a local candidate, not an already published dataset.

## Reproduction

Run from the repository with its virtual environment: `python experiments/run_physical.py --n-flies 100 --workers 8 --output results/NEW_RUN --perturbation`. Use a fresh output directory to preserve this batch. Analysis/rendering scripts currently target the saved `physical_batch` and `physical_pilot` directories. `python experiments/audit_physical.py` checks this saved batch. Existing and circuit unit tests: `python -m pytest tests` (13 passed).

## Interpretation

### Supplemental post-choice observation revision

`results/continuation_batch` (400 flies) and `results/continuation_pilot`
(four filmed flies) repeat the same seeds and rules with
`continue_after_choice=true`. The first choice and latency are latched;
physics continues to the existing two-second test deadline. No missing
trajectory is extrapolated, and no original result is overwritten.
`continuation_validation.json` verifies exact equality of each original
control/physics prefix and learning snapshots, plus unchanged choices and
latencies. The added segment is not part of the original preference score.

`post_choice` flags rows after the first-choice event. `observation_end_s`
and `termination_reason` distinguish deadline from physics failure.
`distance_mm` preserves distance to first choice when one occurs;
`observation_distance_mm` covers the full recording. The last x/y and row
counts describe the new observation endpoint. Configuration snapshots,
seeds and package/source metadata accompany each supplemental batch.
Retain originals and continuations as linked versions, not independent
experimental replicates or separate train/test examples.

The latest polished movie and eight clips are in `output/sleek_revision`.
This visual-only revision adds a genuinely transparent logo, a lighter
header, static-card fade-ins and a subtle progress line. Scientific content,
chapter durations and all measurements are unchanged.
The main film includes a 26-second experiment introduction, paired-versus-DAN
training/testing, all four trajectory groups, and evidence/next-step chapters.
This editorial update reused recordings and did not rerun any experiment.
The corner map shows all recorded movement with no choice-zone mask.
The old planar Y-maze figure is not relabeled as physical body data.
No accelerated-time, free-ranging or whole-brain experiment was added.

Paired: 68 B, 23 A, 9 nonresponses. Untrained: 40 B, 46 A, 14 nonresponses. Unpaired: 42 B, 43 A, 15 nonresponses. Retrieval DAN-silenced: 39 B, 47 A, 14 nonresponses. Each condition has 100 simulated individuals. These outcomes demonstrate the configured mechanism, not an independent biological fit. There is one master-seed batch; no parameter-search uncertainty or biological population variability is estimated.

Kaun et al. (2011), Nature Neuroscience 14:612-619, doi:10.1038/nn.2805: https://pubmed.ncbi.nlm.nih.gov/21499254/ . The video/report compare published qualitative effects and protocol, not invented or digitized paper bar heights.
