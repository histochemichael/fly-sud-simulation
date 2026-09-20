# Drosophila ethanol-cue learning simulation

## Latest published results: individual-behavior V4b

[Paper-versus-simulation results and figures](docs/results/individual_behavior_v4b/RESULTS.md) · [Full Hugging Face dataset](https://huggingface.co/datasets/Histochemichael/fly-sud-simulation)

The 256-fly independently seeded validation showed mixed individual choices and early avoidance / late approach, but **FAILED the full declared validation**: early Holm p=0.03125; late p=0.09375. This is not quantitative biological replication. Both timing points informed calibration, and the memory transition remains too early. Failed V4 calibration and all V4b trials are preserved in the dataset with source snapshots, configurations, seeds, learning states, Parquet histories and checksums. The original experiments were not tuned or overwritten for publication.

![Original paper beside actual simulation results](docs/results/individual_behavior_v4b/paper_vs_sim_v4b.png)

Only paper panel b is directly compared. Sample sizes, statistical tests and axis ranges differ; see the detailed results and [paper figure attribution](docs/results/individual_behavior_v4b/PAPER_ATTRIBUTION.md). The historical V1 film below is not a V4b recording.

## Connectome-based temporal version

See [CONNECTOME_VERSION.md](CONNECTOME_VERSION.md) for the new FlyWire FAFB v783 implementation: 8,570 selected olfactory/mushroom-body neurons, measured neuron-to-neuron connectivity, competing memory hypotheses and biological-time delay tests. It is **not a whole-brain simulation**; dynamics, plasticity and motor decoding remain explicit assumptions. The published V1 below is preserved and does not use connectome-derived connectivity.

Read [CONNECTOME_RESULTS.md](CONNECTOME_RESULTS.md) for the completed 240-test corrected pilot, actual outcome counts, and failed timing/behavioral validation targets. Internal value signs are not behavioral replication. Source code is now published; the Hugging Face release contains V4/V4b records only.

## Historical development and V1 media

The [V4 protocol](INDIVIDUAL_BEHAVIOR_V4_PROTOCOL.md) and [V4b follow-up protocol](INDIVIDUAL_BEHAVIOR_V4B_PROTOCOL.md) preserve the failed gain calibration and separate it from independently seeded validation. The controller retains memory magnitude and gives each fly independent gait, heading, sensory and motor seeds. Variability distributions are assumed; memory timing is unchanged. These records are now published on Hugging Face under raw/v4 and raw/v4b.

The [completed V4b report and individual-fly figures](docs/results/individual_behavior_v4b/RESULTS.md) cover 256 independently seeded validation flies. Mixed choices and early avoidance/late approach were observed. The early paired-versus-unpaired comparison passed (Holm p=0.03125), but the late comparison did not (p=0.09375), so the full declared validation failed. All physical/data audits passed; this is not a quantitative paper replication. The earlier pilot was not tuned or overwritten.

Local panel-b revision: [PANEL_B_RESULTS.md](PANEL_B_RESULTS.md) and [PANEL_B_PROTOCOL.md](PANEL_B_PROTOCOL.md) document 96 physical trials using FlyWire-derived connectivity. Both calibrated models express early avoidance and late approach, but their effect is too strong and memory timing still mismatches the paper. Dopamine learning remains simplified. This is not a quantitative replication or a new public release.

Local follow-up: [CONNECTOME_DIAGNOSTICS.md](CONNECTOME_DIAGNOSTICS.md) reports sensory checks, 56 learning-disabled physical motor diagnostics, and isolated memory timing tests. The body responds to strong approach/avoidance clamps, but the learned readout is weak and frozen memory timing fails the paper comparison. These are diagnostics, not a successful replication or new public release.

The latest silent experiment film is [fly_ethanol_reward_sleek.mp4](output/sleek_revision/fly_ethanol_reward_sleek.mp4). See the [representative figures and detailed report](output/representative_figures.pdf), [actual-result plots and statistics](output/individual_results), and [data guide](output/DATA_GUIDE.md). The four conditions are untrained, paired, unpaired, and retrieval-DAN-silenced.

Historical V1 per-fly histories and states remain in the [V1 GitHub release](https://github.com/histochemichael/fly-sud-simulation/releases/tag/v1.0.0), preserving their results paths. Download all 24 fly-sud-results-part-XX.zip archives and extract each into this repository for V1 saved-data analyses. These are standalone ZIP files, not binary split volumes. The new Hugging Face dataset contains V4/V4b only; do not pool those experiments with V1. Temporary renders, credentials and the virtual environment are excluded.

The physical result is a qualitative conditioned-preference demonstration, not a quantitative replication of Kaun et al. (2011), DOI: 10.1038/nn.2805. It does not reproduce the paper's early aversion, 24-hour consolidation or ethanol pharmacology. Learning is online associative/reward-modulated, not deep learning or a biologically complete addiction model. Continued trajectories reuse the same flies and are not new independent replicates.

The latest film uses a transparent Impulse Neuro logo. Its brain backdrop is reference anatomy; circuit overlays are illustrative simulated activity, not measured or anatomically registered activation. Reference image attribution: Virtual Fly Brain / JFRC2; Jenett et al. (2012), Ito et al. (2014), CC BY 4.0. See the report for full credits. No blanket license is asserted for third-party assets or branding.

The sections below retain historical implementation notes; the current deliverable links above supersede older video filenames and statements about upload status.

## New physical-body experiment and video

The requested follow-up now connects a compact explicit ORN/AL-PN/KC/MBON/DAN circuit to actual FlyGym locomotion. `results/physical_batch` contains **400 physical test trials**, including 100 matched retrieval-DAN-silenced controls. The paired condition made 68 B choices, 23 A choices and 9 nonresponses. This is a separate experiment; do not pool it with the original planar batch described below.

Start with `output/fly_ethanol_reward.mp4`, `output/representative_figures.pdf`, and `output/DATA_GUIDE.md`. `output/video_examples.zip` contains training and test clips for all four reported conditions. Brain activity is a schematic visualization of computed circuit rates, not anatomical imaging. No whole-brain or FlyWire model was built, and nothing was uploaded.

All physical-batch testing uses MuJoCo dynamics. Batch conditioning is stationary; four predetermined representatives additionally walk during training. This remains online associative/reward-modulated learning, not deep learning or a complete addiction model. Kaun et al. comparisons explicitly distinguish timing, metrics and replication units.

Run a new physical batch with `python experiments/run_physical.py --n-flies 100 --workers 8 --output results/NEW_RUN --perturbation`. Preserve existing result directories. See `experiments/analyze_physical.py`, `audit_physical.py`, `render_replays.py`, `build_film.py`, `export_clips.py`, `build_report.py`, and `check_media.py` for the saved-run pipeline. Rendering uses Windows Calibri fonts; narration uses local Windows SAPI. Thirteen unit tests and all 400 saved-data invariant checks passed.

## Original planar V1 notes (historical)

This project is a runnable computational reproduction of the *logic* of the ethanol-cue conditioning paradigm in Kaun et al. (2011), "A Drosophila model for alcohol reward." It asks whether reward-modulated odor memory can bias a sensor-driven embodied fly's later choice when reward is absent.

## What is implemented

- A verified FlyGym 2.1.0 / MuJoCo 3.9.0 NeuroMechFly backend. The smoke test compiles a 69-body-segment fly with leg joints, actuators, and adhesion.
- A fast planar embodied backend for statistically useful batch experiments. Each fly has a pose, bilateral antennae, noisy sensors and motor dynamics, and moves through a spatially varying two-channel odor field in a Y-maze.
- Three conditioning sessions and a compact reward-prediction-error learning rule with independent memory per fly.
- Untrained, paired, and unpaired-reward controls; randomized odor side; no reward during testing.
- Raw choices, trajectories, memory histories, summary statistics, confidence intervals, and four figures.
- A release-oriented Parquet table with per-step state, sensors, reward, memory, actions, outcome, and per-fly seed, plus explicit naive/final learning states.

## Biological mapping and abstractions

| Biological experiment | V1 implementation |
|---|---|
| Odor identity and gradients | Two distance-decaying odor channels |
| Bilateral antennae | Two body-fixed concentration sensors |
| Ethanol exposure | Explicit positive reinforcing state during training |
| Dopamine-dependent plasticity | Reward-prediction-error weight update |
| Walking and turning | Noisy sensor-driven planar body dynamics |
| Y-maze preference test | Reward-free embodied navigation |

V1 uses **online associative, reward-modulated learning during each simulated fly's conditioning experience**. It does not use deep learning. The ethanol state is **not ethanol pharmacology**. The associative weights are not a mushroom-body circuit, the fast planar backend is not the complete NeuroMechFly musculoskeletal body, and simulated time is not a biological 24-hour consolidation model. Results can support claims about the sufficiency of this specified learning-and-navigation mechanism, not claims about real-fly neurobiology, addiction, molecular action, or quantitative effect size.

## Install and run (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe experiments\run_ethanol_conditioning.py --n-flies 100 --seed 42
```

Verify the physical NeuroMechFly/MuJoCo backend and run tests:

```powershell
.\.venv\Scripts\python.exe experiments\check_flygym.py
.\.venv\Scripts\python.exe -m pytest
```

Each run creates a timestamped folder in `results/` containing:

- `state_history.parquet`: per-step/discrete-update training and test records, including pose, heading, bilateral odor readings, ethanol/reward state, teaching signal, memory, action, outcome, and random seed.
- `learning_states.parquet`: naive and final learned state for every fly.
- `choices.csv`, `summary.csv`, `memory_history.csv`, and `trajectories.csv`.
- `config_snapshot.json`, `metadata.json`, and `dataset_manifest.json`, including exact parameters, software versions, config hash, seed provenance, and schema.
- Four PNG figures.

The outputs are structured for possible later conversion into a Hugging Face Dataset, but this project performs no upload and contains no Hugging Face credentials or publishing code. Use `--output-dir PATH` for an exact destination. All scientific and movement parameters live in `config/default.json`; command-line fly count and seed override that snapshot without modifying it.

## Architecture for V2

`OdorField -> bilateral_readout -> AssociativeMemory -> movement controller -> YMaze` is deliberately modular. V2 can replace `AssociativeMemory` with receptor/PN/Kenyon-cell/MBON/DAN circuitry and replace the fast movement layer with a FlyGym turning controller, while retaining the protocol, randomization, analysis, and controls.

The next scientific step is to add explicit mushroom-body and dopaminergic populations, then test whether simulated DAN silencing removes learned odor-B preference without destroying locomotion or innate odor sensing.
