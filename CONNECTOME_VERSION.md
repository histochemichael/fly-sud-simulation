# Connectome-based temporal experiment — development record

This is a new local version, not a replacement for the published V1 baseline. It is a **connectome-constrained olfactory/mushroom-body model**, not a whole-brain simulation. No new GitHub or Hugging Face publication is performed.

## Current follow-up and preserved pilot

The current follow-up is `results/connectome_time_paper_v2`, using frozen configs in `data/connectome_v783/calibration_paper`. It uses the paper's detailed paired-conditioning timing: three 10-min/10-min odor sequences, with 50-min intervening gaps (160 min total). The unpaired control moves three 10-min ethanol exposures into those gaps; that particular unpaired ordering is our dose-matched design, not a claimed reproduction of the paper's control schedule.

The controller now subtracts the predicted equal-bilateral-input response before converting learned values to steering. This removes a fixed left/right bias from unequal anatomical populations. It is an explicit motor-decoder calibration, not a reconstructed motor pathway. The follow-up uses a new master seed (9301) and retains all five delays, six conditions and both models.

The first pilot, `results/connectome_time_v2`, remains intact with 240 trials and its exact source snapshot. Its data-integrity audit passed, but its motor decoder had a persistent anatomical-population bias and its schedule was adapted. Do not pool the two runs or treat the first pilot as confirmatory evidence.

For the current protocol, 24 consolidating and seven decay-only candidate parameter sets satisfy the two endpoint sign constraints. That non-uniqueness is evidence of underdetermination, not confidence in either mechanism. The rest of this document's first-pilot notes are retained for provenance; current results and failure checks are in `CONNECTOME_RESULTS.md` after analysis.

## What is measured

Source: FlyWire FAFB materialization 783, official connectivity DOI [10.5281/zenodo.10676866](https://zenodo.org/records/10676866), and annotation release v2.1.0, commit `ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0`.

The importer checks the published MD5 of the connectivity archive, saves SHA-256 hashes, preserves int64 root IDs, sums neuropil rows by directed neuron pair, and retains pairs with at least five synapses. It joins 139,255 annotations to select:

| Role | Selected neurons |
|---|---:|
| Olfactory sensory neurons | 2,281 |
| Antennal-lobe projection neurons | 685 |
| Kenyon cells | 5,177 |
| Mushroom-body output neurons | 96 |
| Dopaminergic neurons | 331 |
| Total selected | 8,570 |

There are 60,731 retained directed pairs within this selection. 51,369 pairs drive the implemented ORN→PN, PN→KC, KC→MBON and DAN→MBON blocks. DAN→KC connections are exported for inspection but are **not active** in the current memory rule. Other recurrent and cross-class edges are retained but not simulated. Some selected neurons have no usable connection in a particular block; selected count is not a claim that every neuron is activated.

The endpoints of active edges comprise 7,999 selected neurons (2,110 ORNs, 390 PNs, 5,147 KCs, 84 MBONs and 268 DANs). This connected-endpoint count is also not a claim that all cells have nonzero activity in each trial.

This is the female adult brain reconstruction, not a dataset of 160,000 neurons and not a male or whole-CNS reconstruction. All simulated individuals share this anatomical specimen. Their random streams and learning gains vary; anatomy does not.

## What is assumed

- Generic odor A/B tuning at annotated ORN types, seeded and reproducible; not measured isoamyl-alcohol/ethyl-acetate responses.
- Nonnegative, row-normalized synapse-count propagation through a feedforward rate model. Synapse counts are not measured conductances. Neurotransmitter predictions are retained as metadata, not used to infer every synapse's sign.
- A sparsifying threshold is a functional approximation, **not a reconstructed APL inhibitory circuit**.
- Per-KC aversive, precursor and long-term traces modulate effective transmission over real KC→MBON edges. These are assumed plasticity factors, not experimentally observed synaptic changes.
- The three trace types are carried per KC; they are not a reconstruction of the paper's sequential gamma/alpha-prime-beta-prime/alpha-beta memory stages.
- Population MBON values are decoded to left/right gait drive. This is **not a connectome-derived descending-neuron or ventral-nerve-cord motor pathway**.
- Uniform tonic DAN drive and its removal gate appetitive expression through real DAN→MBON edges. Aversion has a separate output. Acquisition/consolidation do not depend on DAN drive by default.
- Broad class-DAN silencing is not an exact reconstruction of a TH-GAL4 or Ddc-GAL4 manipulation.
- Dynamics are dimensionless steady-state rates and analytic memory evolution, not membrane-voltage or spike simulations.
- Annotation-era side labels and positions are retained; current Codex corrected the historical handedness. No claim of anatomically registered movement or imagery is made.

## Steps 1–4

1. **Competing memories:** aversion decays with a fitted time constant. Compare an immediately formed persistent appetitive trace (`decay_only`) with a precursor that transfers into long-term memory (`consolidating`). Learning is online during odor/internal-ethanol overlap.
2. **Biological time:** exposure, clearance, learning, consolidation and retention use seconds/hours. Quiet intervals use analytic state evolution, without inventing physical locomotion during the skipped interval. Ethanol is an uncalibrated uptake/clearance state; its units are not blood concentration.
3. **Delay matrix:** 0.5, 3, 6, 12 and 24 h; untrained, paired, unpaired, retrieval-DAN-silenced, acquisition-DAN-silenced, consolidation-DAN-silenced. Each delay has independent numerical seeds; models and conditions share matched seeds. Odor identity is reciprocally counterbalanced. The first physical pilot has four numerical flies per cell (two reciprocal pairs), totaling 240 physical test rollouts. It is not a high-powered biological replication.
4. **Calibration and falsification:** only paired 0.5/24-h internal-value signs enter calibration. No paper numeric means, error bars, p-values, intermediate delays or perturbation results are fit. The ±0.001 fitting margin is an arbitrary model-unit constraint, not a paper effect size. There are 19 feasible consolidating candidates and six feasible decay-only candidates: the endpoints do not identify the mechanism. Intermediate times and physical outcomes must be reported even when wrong. DAN effects that follow directly from our gating assumption are not independent validation.

## Protocol mismatch that must remain visible

The current frozen pilot uses matched-duration control slots, not the paper's complete protocol. Each of three sessions has a 10-min slot, a 60-min gap, then two 10-min odor slots; a further 60-min interval separates sessions. Paired training presents ethanol with the target odor. Unpaired training presents ethanol alone before the long within-session gap. Total simulated conditioning is 6.5 h. All conditions share this elapsed time and paired/unpaired groups receive equal nominal odor and ethanol exposures.

The full paper methods specify three repetitions of 10-min odor 1 then 10-min odor 2 plus ethanol, spaced by **50-min intervals**, and a **2-min** behavioral test. The paper's overview calls sessions one-hour-spaced. Our pilot's physical test is still V1's **2-second** arena assay. Do not label its training schedule or test duration a protocol replication.

The paper reports the aversion-to-preference transition at **12–15 h**. This is an important held-out qualitative challenge: producing the two endpoint signs does not establish the correct transition time. The current models must be judged against it without secretly retuning on the intermediate results.

## Output schema

Each trial directory contains:
- `state_history.parquet`: one row per biological or control update, with fly/condition/model/delay/phase/session, times, bilateral odors, ethanol, teaching signal, actions, final choice and latency, and list columns holding all ORN/PN/KC/DAN rates, MBON channel outputs, and all three KC memory vectors.
- `physics_history.parquet`: every MuJoCo tick during the test, including pose and a `state_index` join to neural/controller state.
- `learning_states.npz`: naive, post-training, pre-test and final states with exact KC root-ID order.
- `replay.npz`: actual body poses for later rendering; it is not a finished video.
- `summary.json`: outcomes, seeds, paired odor, delay and completion.

Null pose/action fields during stationary conditioning/retention mean not simulated, not zero motion. Training/delay rows timestamp the end of their state-update interval; test rows timestamp the beginning of the held motor-control interval. `interval_mean_teaching_signal` is the integration signal; `instantaneous_teaching_signal` is the endpoint internal state during training. Test reward/teaching signals are zero, while stored memories can continue to decay/consolidate.

List ordering is the ascending `root_id` order within each role in `data/connectome_v783/neurons.parquet`. Memory arrays follow KC ordering, not 24 synthetic units. Raw measured connectivity remains immutable; effective memory-dependent transmission can be reconstructed from counts, normalizations and per-step traces.

Batch files preserve frozen configs, package versions, source/data hashes and numerical seed rules. No old result folder is overwritten.

## Statistics and interpretation

Report numerical sample size, responders and nonresponses. Paired-odor choice probability and latency are computed from actual recorded choices. The paired condition is compared to each control with exact matched McNemar tests on paired-odor choice versus all other outcomes, with Holm correction across 50 tests. These are exploratory tests of simulator variability, not tests of biological validity. Conditional-on-response probabilities are shown separately from completion. No missing latency is imputed as the test deadline. With four numerical flies per cell, nonsignificance cannot establish equivalence.

The paper uses reciprocal-group conditioned preference indexes; our raw neural value is **not** that index. A small simulation p-value would not prove a biological mechanism. Topology provenance, numerical correctness, behavioral fit and mechanistic identification are separate checks.

## Reproduce

For the current follow-up, use `experiments/calibrate_connectome.py --paper` once, then add `--calibration data/connectome_v783/calibration_paper` to the cohort command below. Use a new output directory. Historical configs remain in `calibration`; do not silently substitute them.

```powershell
.\.venv\Scripts\python.exe experiments/fetch_connectome.py
.\.venv\Scripts\python.exe experiments/build_connectome.py
.\.venv\Scripts\python.exe experiments/calibrate_connectome.py
.\.venv\Scripts\python.exe experiments/run_connectome.py --output results/NEW_CONNECTOME_RUN --physical --batches 2 --per-batch 2 --workers 8
.\.venv\Scripts\python.exe experiments/audit_connectome.py results/NEW_CONNECTOME_RUN
.\.venv\Scripts\python.exe experiments/analyze_connectome.py results/NEW_CONNECTOME_RUN
```

Calibration refuses to overwrite its frozen output. Use the already saved calibrated configs to rerun the same model. The complete scientific audit is specific to the documented 240-trial matrix.

## Sources and reuse

- Dorkenwald et al. (2024), [Neuronal wiring diagram of an adult brain](https://doi.org/10.1038/s41586-024-07558-y).
- Schlegel et al. (2024), [Whole-brain annotation and multi-connectome cell typing of Drosophila](https://doi.org/10.1038/s41586-024-07686-5).
- [FlyWire official connectivity archive](https://zenodo.org/records/10676866), CC BY 4.0. Preserve attribution for reused or derived connectivity.
- [Publication-era annotation release](https://github.com/flyconnectome/flywire_annotations/tree/v2.1.0). Source provenance is preserved; no new blanket license is asserted.
- Kaun et al. (2011), [A Drosophila model for alcohol reward](https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/), DOI 10.1038/nn.2805.
