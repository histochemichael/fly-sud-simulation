# Connectome version: results and limitations

The connectome-based implementation and both memory hypotheses have been built and tested. **This is not a successful reproduction of the paper's aversion-to-preference behavior.**

## What ran

- 240 corrected physical test rollouts: two models × six conditions × five delays × four numerical flies.
- 8,570 selected real FlyWire neurons; 51,369 measured directed edges drive the active feedforward blocks. 7,999 selected neurons are endpoints of those active edges.
- Separate 240-test diagnostic pilot and four-test smoke run preserved, not pooled.
- Corrected paired training: three 10-min/10-min odor sequences with 50-min gaps. Unpaired ethanol ordering is our dose-matched control adaptation.
- Tests at 0.5, 3, 6, 12 and 24 biological hours, with independent delay cohorts and reciprocal odor identity.
- Two-second MuJoCo tests, not the paper's two-minute assay. Training/retention have internal-state histories, not invented motor trajectories.
- Not whole-brain dynamics, deep learning, a complete addiction model, or a connectome-derived descending motor pathway.

## Actual paired-condition endpoints

| Model | Delay (h) | Paired odor choices | Other choices | Nonresponses | Total |
|---|---:|---:|---:|---:|---:|
| consolidating | 0.5 | 2 | 2 | 0 | 4 |
| consolidating | 24 | 2 | 2 | 0 | 4 |
| decay_only | 0.5 | 2 | 2 | 0 | 4 |
| decay_only | 24 | 2 | 2 | 0 | 4 |

Counts come from recorded fly trajectories, not internal neural values.

## Held-out timing and identifiability

- consolidating: first positive internal value approximately 1.5 h (0.25-h sampling); paper behavioral switch 12–15 h. Timing match: **False**.
  Numerical training-step sensitivity: relative early-value change between 60-s and 10-s steps = 0.024%.
- decay_only: first positive internal value approximately 2.5 h (0.25-h sampling); paper behavioral switch 12–15 h. Timing match: **False**.
  Numerical training-step sensitivity: relative early-value change between 60-s and 10-s steps = 0.012%.

The endpoint signs were calibration targets, not validation. Twenty-four consolidating and seven decay-only candidates met them. Intermediate timing was not fit. Model-generated DAN effects following directly from the assumed gate are not independent biological evidence.

## Statistics and verification

- 50 exact matched McNemar tests, four numerical pairs each, with Holm correction. Minimum raw p = 1; minimum corrected p = 1.
- Tests concern numerical trajectories sharing one anatomical specimen. Tiny cohorts cannot establish equivalence or biological validity.
- Complete-data audit: 240 trials, 76,824 neural/control records, 4,800,000 physics records, 0 physics failures.
- Initial/post-training/pre-test/final learning states saved; per-neuron arrays checked for dimensions and finite values.
- Dataset hashes and archived source verified. All 25 regression tests passed before this run.

## Implemented versus unproven

| Requirement | Status |
|---|---|
| Real connectome-based network | Implemented; bounded olfactory/mushroom-body subnetwork |
| Competing aversive/appetitive memories | Implemented hypotheses, not inferred kinetics |
| Biological-time retention and clearance | Implemented; pharmacokinetics not empirically calibrated |
| Delay-by-condition experiment | Completed numerical pilot |
| Held-out validation | Performed; successful biological replication not established |
| Paper's sequential KC subtype roles | Not reconstructed; three traces per KC are not that circuit |
| 12–15 h switch and long-delay behavioral preference | Not established |
| New public release | Not uploaded; original V1 release unchanged |

## Next evidence-driven step

**Diagnostic follow-up completed:** see [CONNECTOME_DIAGNOSTICS.md](CONNECTOME_DIAGNOSTICS.md) for the requested perception, motor-decoding and isolated-memory checks. The original results below and their source snapshots are preserved.

Validate the sensory-to-motor interface with controlled odor gradients and measured MBON behavioral signs. Extend the physical observation window with an explicit compute budget. Compare additional consolidation mechanisms using independent time-course data. Map the paper's genetic manipulations to specific identified cells, rather than silencing all DANs. Do not present a fit to 12–15 h as independent validation afterward.

## Files

Run folder: results/connectome_time_paper_v2

- choices.csv, summary.csv, statistics.json: actual outcomes and tests.
- validation_report.json and frozen_neural_timecourse.csv: frozen predictions and timing checks.
- figures/: real annotation positions/connectivity, complete trajectories, actual latency violins/dots, time curves.
- source_snapshot.zip, metadata.json, config_snapshot.json: reproduction records.
- Per-fly Parquet histories and NPZ learning states; see [methods and schema](CONNECTOME_VERSION.md).

## Sources

[Kaun et al. (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/); [FlyWire data](https://zenodo.org/records/10676866); [Dorkenwald et al. (2024)](https://doi.org/10.1038/s41586-024-07558-y); [Schlegel et al. (2024)](https://doi.org/10.1038/s41586-024-07686-5).
