# Drosophila ethanol-cue reward simulation (V1)

## Current public release

The latest silent experiment film is [fly_ethanol_reward_sleek.mp4](output/sleek_revision/fly_ethanol_reward_sleek.mp4). See the [representative figures and detailed report](output/representative_figures.pdf), [actual-result plots and statistics](output/individual_results), and [data guide](output/DATA_GUIDE.md). The four conditions are untrained, paired, unpaired, and retrieval-DAN-silenced.

Full per-fly Parquet histories, initial/final learning states, seeds, configurations, metadata and validation records are distributed in the GitHub release archive, preserving their `results/` paths. Extract it in this repository to run saved-data analyses. Summaries are also tracked in `results/`. No data are uploaded to Hugging Face. Temporary renders, the virtual environment and obsolete video versions are excluded.

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
