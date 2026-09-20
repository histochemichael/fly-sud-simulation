"""Prepare an additive, auditable public release without changing saved experiments."""
from pathlib import Path
import argparse, hashlib, json, shutil
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-image", type=Path, required=True)
    a = ap.parse_args()
    report = ROOT / "results/individual_behavior_v4b/report"
    docs = ROOT / "docs/results/individual_behavior_v4b"
    stage = ROOT / "publication/hf-fly-sud-v4b"
    docs.mkdir(parents=True, exist_ok=False)
    stage.mkdir(parents=True, exist_ok=False)
    for p in report.iterdir():
        if p.is_file() and p.suffix in {".png", ".csv", ".json", ".md", ".zip"}:
            shutil.copy2(p, docs / p.name)
    paper = docs / "kaun_2011_figure1_user_supplied.png"
    shutil.copy2(a.paper_image, paper)
    data = pd.read_csv(report / "validation_reciprocal_cpi.csv")
    fig = plt.figure(figsize=(15, 8), facecolor="white")
    left = fig.add_axes([.035,.23,.43,.65])
    left.imshow(plt.imread(paper))
    left.axis("off")
    left.set_title("PAPER: original Figure 1, unchanged", fontsize=14)
    right = fig.add_axes([.57,.30,.39,.53])
    for x, delay in enumerate([.5,24]):
        values = data[(data.condition == "paired") & (data.delay_h == delay)].cpi.to_numpy()
        right.scatter(x + np.linspace(-.09,.09,len(values)), values, color="#e99b00", alpha=.65)
        right.errorbar(x, values.mean(), yerr=values.std(ddof=1)/np.sqrt(len(values)),
                       fmt="D", color="#c77f00", capsize=6, markersize=8)
    right.axhline(0,color=".6",lw=1)
    right.set(xticks=[0,1],xticklabels=["30 min","24 h"],ylim=(-1.12,1.12),
              xlim=(-.4,1.4),ylabel="Actual-choice CPI",xlabel="Retention delay")
    right.set_title("SIMULATION: independent validation seeds\nMean ± SEM; dots = reciprocal pairs",fontsize=13)
    fig.suptitle("Kaun et al. (2011) versus FlyWire-informed V4b simulation",fontsize=18,y=.97)
    fig.text(.05,.18,"Paper panel b: N=8 reciprocal pairs; 50 flies per odor group.\nWilcoxon two-sample: p=0.006 (30 min), p=0.02 (24 h).",fontsize=10)
    fig.text(.55,.18,"Simulation: N=8 pairs; 32 flies per condition/delay.\nExact paired sign-flip, Holm p=0.03125 / 0.09375.",fontsize=10)
    fig.text(.05,.09,"Overall simulation validation FAILED: the late contrast did not pass. Timing points informed calibration.\nCompare panel b only; paper panels c/d are NOT reproduced. Original paper axes retained; axis ranges differ.\nPaper image supplied by user; Kaun et al., Nature Neuroscience 14, 612–619, DOI:10.1038/nn.2805.",fontsize=10)
    fig.savefig(docs / "paper_vs_sim_v4b.png", dpi=170)
    plt.close(fig)
    attribution = """# Paper figure attribution

Kaun KR, Azanchi R, Maung Z, Hirsh J, Heberlein U (2011).
A Drosophila model for alcohol reward. Nature Neuroscience 14, 612–619.
DOI: https://doi.org/10.1038/nn.2805
Primary source: https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/

The user supplied the original Figure 1 image for this research comparison.
Its pixel content is retained unchanged in the standalone asset and displayed
beside newly plotted simulation data. Only panel b is the direct comparison.
No paper data were digitized or invented. Paper panels c/d are not reproduced
by this simulation. The paper figure is not a simulation record or training sample.
Original author/publisher rights remain; no blanket open license is asserted
for this figure, third-party assets, or branding.
"""
    (docs / "PAPER_ATTRIBUTION.md").write_text(attribution,encoding="utf-8")
    comparison = """## Paper versus simulation

![Original paper beside actual simulation results](paper_vs_sim_v4b.png)

Compare the paper's panel b with our paired-condition CPI plot. Panels c/d
(persistence and shock resistance) have not been reproduced. Original paper
axes are preserved and differ from the simulation axes.

| Detail | Kaun et al. Figure 1b | Current simulation |
|---|---|---|
| Outcome direction | Early aversion, late preference | Same directions, not quantitative replication |
| Replication | N=8 reciprocal pairs; 50 flies per odor group | N=8 pairs; 2 flies per odor group, 32 per condition/delay |
| Error bars | Mean ± SEM across replicates | Mean ± SEM across reciprocal pairs, not individual movement spread |
| Tests versus unpaired | Wilcoxon two-sample, p=0.006 / 0.02 | Exact paired sign-flip, Holm p=0.03125 / 0.09375 |
| Overall validation | Biological experiment | FAILED declared gate: late contrast did not pass |
| Time dependence | Independently observed biology | Both times informed calibration; new seeds are not new biological evidence |

The assumed memory mechanism still changes sign too early. Larger N can improve
precision but cannot repair incorrect mechanisms or establish biological validity.
No post-validation tuning or selective exclusions were made.

[Paper source](https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/) · [Figure attribution](PAPER_ATTRIBUTION.md)

## Public data

Full per-fly records and failed calibration history:
[Hugging Face dataset](https://huggingface.co/datasets/Histochemichael/fly-sud-simulation).
Code and protocols: [GitHub](https://github.com/histochemichael/fly-sud-simulation).
Dataset raw paths are under raw/v4/ and raw/v4b/. Earlier V1 GitHub releases
remain separate and must not be pooled with these runs.

"""
    original = (report / "RESULTS.md").read_text(encoding="utf-8")
    (docs / "RESULTS.md").write_text(original.replace("## Actual CPI results",comparison+"## Actual CPI results"),encoding="utf-8")
    provenance = {"prepared_utc":datetime.now(timezone.utc).isoformat(),
      "input_report_sha256":digest(report/"RESULTS.md"),"paper_image_sha256":digest(paper),
      "preparation_script_sha256":digest(Path(__file__)),
      "original_experiments_unchanged":True,"comparison":"Original paper image beside original simulation data; no digitization"}
    (docs/"PUBLICATION_PROVENANCE.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    tables = stage / "tables"
    tables.mkdir()
    for version in ["v4","v4b"]:
        source = ROOT / ("results/individual_behavior_"+version)
        target = stage / "raw" / version
        target.mkdir(parents=True)
        collected = {}
        for p in source.iterdir():
            if p.is_file() and p.suffix in {".json",".zip"}:
                shutil.copy2(p,target/p.name)
            elif p.is_dir() and (p.name.startswith("calibration_gain_") or p.name=="validation"):
                shutil.copytree(p,target/p.name)
                choice = p/"choices.csv"
                if choice.exists():
                    role = "validation" if p.name=="validation" else "calibration"
                    df = pd.read_csv(choice)
                    df["run_version"]=version
                    df["experimental_role"]=role
                    df["source_batch"]=p.name
                    df["individual_key"]=version+"/"+p.name+"/"+df.fly_id.astype(str)
                    collected.setdefault(role,[]).append(df)
        for role, frames in collected.items():
            pd.concat(frames,ignore_index=True).to_parquet(tables/(version+"_"+role+"_choices.parquet"),index=False)
    shutil.copytree(docs,stage/"results/v4b")
    (stage/"protocols").mkdir()
    for name in ["INDIVIDUAL_BEHAVIOR_V4_PROTOCOL.md","INDIVIDUAL_BEHAVIOR_V4B_PROTOCOL.md"]:
        shutil.copy2(ROOT/name,stage/"protocols"/name)
    card = """---
language:
- en
pretty_name: FlyWire-informed fly odor-reward simulation — V4b
tags:
- neuroscience
- drosophila
- flywire
- mujoco
- simulation
- associative-learning
configs:
- config_name: v4b_choices
  default: true
  data_files:
  - split: calibration
    path: tables/v4b_calibration_choices.parquet
  - split: validation
    path: tables/v4b_validation_choices.parquet
- config_name: v4_failed_calibration
  data_files:
  - split: calibration
    path: tables/v4_calibration_choices.parquet
---

# FlyWire-informed odor-reward simulation: individual-behavior V4b

**The full predeclared validation FAILED. This is synthetic simulation data,
not measured fly behavior or a quantitative reproduction of Kaun et al. (2011).**

[Detailed results](results/v4b/RESULTS.md) ·
[Code and protocols](https://github.com/histochemichael/fly-sud-simulation)

![Paper versus simulation](results/v4b/paper_vs_sim_v4b.png)

## Findings and limitations

256 independently seeded validation flies, four conditions (paired, unpaired,
untrained, retrieval-DAN-silenced), two delays (30 min, 24 h), 32 flies per cell:
8 reciprocal replicate pairs with 2 flies per odor group.
Paired CPI was -0.59375 ± 0.13311 SEM early and +0.46875 ± 0.15264 late.
Paired-versus-unpaired exact two-sided paired sign-flip tests gave Holm-adjusted
p=0.03125 and p=0.09375. The late contrast failed the declared statistical gate.
Ten stationary nonresponders remain in denominators. No selective exclusions
or post-validation tuning were performed.

Both timing points informed calibration: independent validation seeds assess
model repeatability, not held-out time-course prediction or biological validity.
Individual variation is modeled, not fitted to measured individual flies.
The memory transition remains too early; 7-day persistence and shock resistance
are not reproduced. The paper and simulation use different tests and group sizes.
CPI SEM is across reciprocal pairs, not a standard deviation of individual paths.

## Included runs

- V4 failed calibration: 48 trials at gains 5, 20, 80, using the same 16 seed
  profiles across gains (common random numbers). No candidate passed.
- V4b calibration: 32 new flies at predeclared gain 50.
- V4b validation: 256 new flies after freezing the selected configuration.

Failed calibration records are preserved. Earlier V1/V2/V3 experiments are not
pooled into these tables. Original V1 archives remain in the GitHub release.

## Data layout and reuse

Dataset Viewer exposes only explicit individual-choice tables with stable
individual_key = version/batch/fly_id. Full records are under raw/v4/ and raw/v4b/.
Each trial includes Parquet state/physics/learning histories, summary, individual
configuration, replay and NPZ naive/post-training/pre-test/final learning states,
including exact FlyWire root IDs. Configuration, seeds, software versions and
source snapshots are preserved alongside each run. MANIFEST.json provides
SHA-256 checksums for all uploaded files except the manifest itself.

~~~python
from datasets import load_dataset
choices = load_dataset("Histochemichael/fly-sud-simulation", "v4b_choices")
from huggingface_hub import snapshot_download
folder = snapshot_download("Histochemichael/fly-sud-simulation", repo_type="dataset")
~~~

Physics state indices join within the same trial, not globally. Nonresponse
latencies are censored at the 120-second assay limit, never zero-latency choices.
At an absorbing collection zone physics stops; subsequent hold rows are not
walking. Do not invent trajectories beyond collection. Test memory states can
be reconstructed from initial state, elapsed time and archived update equations.

## Model and compute

Real FlyWire FAFB v783 connectivity informs an 8,570-neuron olfactory/mushroom-body
subset. All simulated flies share that reference anatomy; individual connectomes
were not measured. Assumed dynamics, noise, memory kinetics and motor decoding
are not determined by anatomy. Dopamine is a simplified DAN-to-MBON retrieval
gate; this is not mechanistic DAN-to-KC plasticity or a whole-brain simulation.
Learning is online associative/reward-modulated, not deep learning or a complete
biological addiction model.

Testing uses actual MuJoCo/NeuroMechFly physical dynamics. Conditioning is
stationary. Local Intel Core i7-9700F, eight CPU workers; no GPU acceleration.
Publishing dependencies may differ from archived experiment environments.

## Sources and rights

Kaun et al. (2011), A Drosophila model for alcohol reward, Nature Neuroscience,
DOI:10.1038/nn.2805. [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/).
The user-supplied paper figure is retained unchanged for research comparison,
not a training example. See [attribution](results/v4b/PAPER_ATTRIBUTION.md).
Original author/publisher rights remain; no blanket license is asserted.
FlyWire source connectivity is attributed under its CC BY 4.0 source terms:
https://zenodo.org/records/10676866 . Branding and other third-party assets
retain their own rights. Do not infer an unrestricted license from public access.
"""
    (stage/"README.md").write_text(card,encoding="utf-8")
    entries = [{"path":p.relative_to(stage).as_posix(),"bytes":p.stat().st_size,"sha256":digest(p)}
               for p in sorted(stage.rglob("*")) if p.is_file()]
    (stage/"MANIFEST.json").write_text(json.dumps({"files":entries,"provenance":provenance},indent=2),encoding="utf-8")
    print(json.dumps({"docs":str(docs),"stage":str(stage),"files":len(entries)+1,"bytes":sum(e["bytes"] for e in entries)}))
if __name__=="__main__":
    main()

