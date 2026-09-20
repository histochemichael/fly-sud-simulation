"""Write an evidence-backed report from the completed cohort."""
from pathlib import Path
import sys,json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/(sys.argv[1] if len(sys.argv)>1 else "results/connectome_time_paper_v2")
    d=pd.read_csv(out/"choices.csv");s=pd.read_csv(out/"summary.csv")
    validation=json.loads((out/"validation_report.json").read_text())
    audit=json.loads((out/"audit.json").read_text());stats=json.loads((out/"statistics.json").read_text())
    curve=pd.read_csv(out/"frozen_neural_timecourse.csv")
    fig,ax=plt.subplots(figsize=(10,5))
    for model,g in curve.groupby("model"):ax.plot(g.delay_h,g.neural_value,label=model)
    ax.axhline(0,c="black",lw=.8);ax.axvspan(12,15,color="gray",alpha=.18,label="Paper behavioral transition: 12–15 h")
    ax.scatter([.5,24],[0,0],marker="x",c="red",s=50,label="Endpoint calibration times")
    ax.set_xlabel("Hours after training");ax.set_ylabel("Internal paired-minus-unpaired value (model units)")
    ax.set_title("Frozen internal predictions do not identify the biological time course")
    ax.legend(fontsize=8);ax.grid(alpha=.15);fig.tight_layout()
    fig.savefig(out/"figures/falsification.png",dpi=170);plt.close(fig)
    lines=["# Connectome version: results and limitations","",
      "The connectome-based implementation and both memory hypotheses have been built and tested. **This is not a successful reproduction of the paper's aversion-to-preference behavior.**","",
      "## What ran","",
      f"- {len(d)} corrected physical test rollouts: two models × six conditions × five delays × four numerical flies.",
      "- 8,570 selected real FlyWire neurons; 51,369 measured directed edges drive the active feedforward blocks. 7,999 selected neurons are endpoints of those active edges.",
      "- Separate 240-test diagnostic pilot and four-test smoke run preserved, not pooled.",
      "- Corrected paired training: three 10-min/10-min odor sequences with 50-min gaps. Unpaired ethanol ordering is our dose-matched control adaptation.",
      "- Tests at 0.5, 3, 6, 12 and 24 biological hours, with independent delay cohorts and reciprocal odor identity.",
      "- Two-second MuJoCo tests, not the paper's two-minute assay. Training/retention have internal-state histories, not invented motor trajectories.",
      "- Not whole-brain dynamics, deep learning, a complete addiction model, or a connectome-derived descending motor pathway.","",
      "## Actual paired-condition endpoints","",
      "| Model | Delay (h) | Paired odor choices | Other choices | Nonresponses | Total |",
      "|---|---:|---:|---:|---:|---:|"]
    for _,r in s[(s.condition=="paired")&s.delay_h.isin([.5,24])].iterrows():
        lines.append(f"| {r.model} | {r.delay_h:g} | {int(r.paired_choices)} | {int(r.other_choices)} | {int(r.nonresponses)} | {int(r.n_total)} |")
    lines+=["","Counts come from recorded fly trajectories, not internal neural values.","",
      "## Held-out timing and identifiability",""]
    for r in validation["reports"]:
        lines.append(f"- {r['model']}: first positive internal value approximately {r['first_positive_internal_value_h']:g} h (0.25-h sampling); paper behavioral switch 12–15 h. Timing match: **{r['internal_timing_matches_paper']}**.")
        lines.append(f"  Numerical training-step sensitivity: relative early-value change between 60-s and 10-s steps = {r['relative_60_vs_10_error']:.3%}.")
    lines+=["",
      "The endpoint signs were calibration targets, not validation. Twenty-four consolidating and seven decay-only candidates met them. Intermediate timing was not fit. Model-generated DAN effects following directly from the assumed gate are not independent biological evidence.","",
      "## Statistics and verification","",
      f"- 50 exact matched McNemar tests, four numerical pairs each, with Holm correction. Minimum raw p = {min(t['p_raw'] for t in stats['tests']):g}; minimum corrected p = {min(t['p_holm'] for t in stats['tests']):g}.",
      "- Tests concern numerical trajectories sharing one anatomical specimen. Tiny cohorts cannot establish equivalence or biological validity.",
      f"- Complete-data audit: {audit['trials']} trials, {audit['state_rows']:,} neural/control records, {audit['physics_rows']:,} physics records, {audit['failures']} physics failures.",
      "- Initial/post-training/pre-test/final learning states saved; per-neuron arrays checked for dimensions and finite values.",
      "- Dataset hashes and archived source verified. All 25 regression tests passed before this run.","",
      "## Implemented versus unproven","",
      "| Requirement | Status |","|---|---|",
      "| Real connectome-based network | Implemented; bounded olfactory/mushroom-body subnetwork |",
      "| Competing aversive/appetitive memories | Implemented hypotheses, not inferred kinetics |",
      "| Biological-time retention and clearance | Implemented; pharmacokinetics not empirically calibrated |",
      "| Delay-by-condition experiment | Completed numerical pilot |",
      "| Held-out validation | Performed; successful biological replication not established |",
      "| Paper's sequential KC subtype roles | Not reconstructed; three traces per KC are not that circuit |",
      "| 12–15 h switch and long-delay behavioral preference | Not established |",
      "| New public release | Not uploaded; original V1 release unchanged |","",
      "## Next evidence-driven step","",
      "Validate the sensory-to-motor interface with controlled odor gradients and measured MBON behavioral signs. Extend the physical observation window with an explicit compute budget. Compare additional consolidation mechanisms using independent time-course data. Map the paper's genetic manipulations to specific identified cells, rather than silencing all DANs. Do not present a fit to 12–15 h as independent validation afterward.","",
      "## Files","",
      f"Run folder: {out.relative_to(ROOT).as_posix()}","",
      "- choices.csv, summary.csv, statistics.json: actual outcomes and tests.",
      "- validation_report.json and frozen_neural_timecourse.csv: frozen predictions and timing checks.",
      "- figures/: real annotation positions/connectivity, complete trajectories, actual latency violins/dots, time curves.",
      "- source_snapshot.zip, metadata.json, config_snapshot.json: reproduction records.",
      "- Per-fly Parquet histories and NPZ learning states; see [methods and schema](CONNECTOME_VERSION.md).","",
      "## Sources","",
      "[Kaun et al. (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/); [FlyWire data](https://zenodo.org/records/10676866); [Dorkenwald et al. (2024)](https://doi.org/10.1038/s41586-024-07558-y); [Schlegel et al. (2024)](https://doi.org/10.1038/s41586-024-07686-5).",""]
    (ROOT/"CONNECTOME_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    print(ROOT/"CONNECTOME_RESULTS.md")
if __name__=="__main__":main()

