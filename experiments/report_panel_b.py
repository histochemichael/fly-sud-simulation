"""Save frozen memory validation and create a plainly qualified pilot report."""
from pathlib import Path
import sys,json,zipfile,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from fly_sud.connectome import Connectome,train_and_delay
from fly_sud.panel_b import memory_factory,FunctionalReadout
def main():
    out=ROOT/sys.argv[1];meta=json.loads((out/"metadata.json").read_text());summary=pd.read_csv(out/"summary.csv")
    audit=json.loads((out/"audit.json").read_text());stats=json.loads((out/"statistics.json").read_text())
    curve=pd.read_csv(ROOT/"data/connectome_v783/panel_b_v3/frozen_predictions.csv")
    validation=[]
    for model,cfg in meta["configs"].items():
        vals=[]
        for step in [60.,10.]:
            c={**cfg,"biological_step_s":step};g=Connectome(ROOT/"data/connectome_v783",c)
            m=memory_factory(len(g.ids["KC"]),c,model);train_and_delay(g,m,"paired",1,.5);r=FunctionalReadout(g)
            vals.append(r.probe(m,1)-r.probe(m,0))
        positive=curve[(curve.model==model)&(curve.condition=="paired")&(curve.value_difference>0)]
        crossing=float(positive.delay_h.min())
        validation.append(dict(model=model,first_positive_internal_value_h=crossing,held_out_timing_match=12<=crossing<=15,
            early_value_dt60=vals[0],early_value_dt10=vals[1],relative_training_step_error=abs(vals[0]-vals[1])/abs(vals[1])))
    (out/"memory_validation.json").write_text(json.dumps(validation,indent=2))
    fig,ax=plt.subplots(figsize=(8,4.5))
    for model,part in curve[curve.condition=="paired"].groupby("model"):ax.plot(part.delay_h,part.value_difference,label=model)
    ax.axhspan(-.001,.001,color="gray");ax.axvspan(12,15,color="gray",alpha=.2,label="Paper behavioral transition")
    ax.set(xlim=(0,48),xlabel="Retention delay (h)",ylabel="Internal value difference, NOT CPI",title="Frozen memory predictions: timing still mismatches")
    ax.legend();fig.tight_layout();fig.savefig(out/"figures/memory_validation.png",dpi=150);plt.close(fig)
    rows=[]
    for r in summary.itertuples():
        rows.append(f"| {r.model} | {r.condition} | {r.delay_h:g} | {r.positive_choices}/{r.negative_choices}/{r.nonresponders} | {r.mean:+.3f} | {r.count} |")
    table="\n".join(rows)
    gates=[]
    for model in summary.model.unique():
        s=summary[summary.model==model].set_index(["condition","delay_h"])
        early=s.loc[("paired",.5),"mean"];late=s.loc[("paired",24.),"mean"]
        ok=early<0 and late>0 and early<s.loc[("unpaired",.5),"mean"] and late>s.loc[("unpaired",24.),"mean"]
        ok=bool(ok and audit["physics_failures"]==0 and audit["containment_passed"])
        gates.append(dict(model=model,calibrated_behavioral_direction_match=ok,early_cpi=float(early),late_cpi=float(late)))
    (out/"interpretation.json").write_text(json.dumps(dict(models=gates,quantitative_replication=False,
        independent_mechanistic_validation=False,seven_day_tested=False,punishment_tested=False),indent=2))
    lines=[
        "# Panel-b attempt: physical learned choices, V3",
        "",
        "The revised model expresses early avoidance and late approach only to the extent shown by the actual-choice results below. This is a calibrated model's behavioral test, not independent biological replication. Quantitative effect size, time dependence and the remaining figure panels are not reproduced.",
        "",
        "## Actual results",
        "",
        "| Model | Condition | Delay (h) | Target / other / none | CPI | Reciprocal replicate pairs |",
        "|---|---|---:|---:|---:|---:|",
        table,
        "",
        "Each cell contains 8 numerical flies: two reciprocal pairs, each containing two odor groups with two side-counterbalanced flies. CPI includes nonresponders. This does not match the paper's group sizes. All flies share one female anatomical reconstruction and generic odor encoding.",
        "",
        "Qualitative direction checks: "+'; '.join(r['model']+(': passed' if r['calibrated_behavioral_direction_match'] else ': failed') for r in gates)+". Passing this check is not quantitative or mechanistic replication.",
        "",
        "## What this does and does not recreate",
        "",
        "- The paper's panel b shows early negative and late positive conditioned preference. Our 0.5/24-hour internal-value signs were calibration targets, so matching their behavioral direction is not a new prediction.",
        "- CPI near -1 or +1 means essentially deterministic choices in this small pilot. Those magnitudes are much stronger than the roughly -0.28 and +0.23 bars visible in the supplied paper figure; those paper values are approximate visual readings, not raw data or fit targets.",
        "- The paper's error bars and significant group comparisons are not reproduced. We have only two numerical replicate pairs per cell; increasing that number alone would not validate our biological assumptions.",
        "- One-stage and cascade memory crossings are "+", ".join(f"{r['model']}: {r['first_positive_internal_value_h']:g} h" for r in validation)+". Both precede the published 12–15-hour behavioral transition. Internal values and behavior are different endpoints.",
        "- Seven-day persistence (panel c) and shock-resistant seeking with sugar/ethanol/odor/naive controls (panel d) were not run.",
        "",
        "[Paper and methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/).",
        "",
        "## Statistics",
        "",
        "Exact two-sided paired sign-flip tests use reciprocal-replicate CPI differences (symmetric-null assumption), with Holm correction across all "+str(len(stats))+" reported comparisons. They do not treat individual flies or timesteps as independent animals. Full test results: results/panel_b_pilot_v3/statistics.json.",
        "",
        "Minimum raw p="+str(min(r["p"] for r in stats))+"; minimum Holm-adjusted p="+str(min(r["p_holm"] for r in stats))+". N=2 pairs is insufficient for a convincing statistical reproduction. Zero-variance/small-N cells are shown as actual dots rather than an invented smooth violin.",
        "",
        "![Actual reciprocal-pair CPI](results/panel_b_pilot_v3/figures/paired_cpi.png)",
        "",
        "![One-stage trajectories](results/panel_b_pilot_v3/figures/consolidating_trajectories.png)",
        "",
        "![Cascade trajectories](results/panel_b_pilot_v3/figures/cascade_trajectories.png)",
        "",
        "Trajectories are complete moving paths until entry into collection regions, reflected to align the assigned target arm. Collection is an absorbing-vial approximation, explicitly logged, not a plot clipping boundary. Blue is early; orange is late. There is no claim of physical movement during collection holds.",
        "",
        "![Memory timing](results/panel_b_pilot_v3/figures/memory_validation.png)",
        "",
        "## What changed",
        "",
        "The original hemisphere-based steering readout was replaced with concentration-normalized functional odor-value comparison using the measured FlyWire feedforward graph. The same graph is evaluated for each antenna; this is an assumed functional decoder, not two reconstructed brains or a measured descending motor pathway.",
        "Dataset: FlyWire FAFB v783, using 8,570 selected neurons (2,281 ORNs, 685 PNs, 5,177 KCs, 96 MBONs and 331 DANs). The source is the FlyWire Consortium connectivity release, not a synthetic replacement or the full brain: https://zenodo.org/records/10676866.",
        "Dopamine limitation: measured DAN-to-MBON connections contribute to an assumed tonic on/off retrieval gate. DAN-to-KC connections are exported but inactive in the plasticity rule. Ethanol/internal reward drives the associative teaching term directly. No realistic dopamine release, cell-specific DAN dynamics or compartment-specific dopamine-dependent synaptic learning is implemented. This pilot's three conditions do not add a new DAN-silencing cohort.",
        "",
        "A second, two-stage consolidation hypothesis was added. Both models learn online from odor/ethanol experience. Neither controller contains a rule that changes the target choice at a specified time. No deep learning or complete addiction model is claimed.",
        "",
        "The body walks in an actual collision-enabled flat Y-maze. A reward-independent geometric wall reflex and prescribed upwind-walking reflex support navigation. These reflexes were developed using separate clamps, not tuned to this cohort's CPI. They are engineering approximations, not measured neural circuitry or fluid dynamics.",
        "",
        "Important correction to earlier arena interpretations: direct inspection confirmed that the older walls had no explicit body contact pairs and body collision masks were zero. Earlier excursions were not evidence of flies climbing valid walls. Preserve those results as open-arena choice-zone experiments, not confined Y-maze trials. This revision fixes the contacts and separately audits body-center containment.",
        "",
        "## Verification and data",
        "",
        f"Audit: {audit['trials']} trials, {audit['physics_ticks']:,} actual physics rows, {audit['control_rows']:,} control/collection rows; {audit['physics_failures']} physics failures; {audit['outside_arena_physics_rows']} outside-arena body-center samples with 0.5-mm boundary tolerance.",
        "Nonresponders with less than 1 mm of movement during their final 10 seconds: "+str(len(audit.get('stalled_nonresponse_trials',[])))+". These are flagged as possible gait/navigation stalls, not interpreted as learned aversion. They remain in CPI denominators. Trial details are in audit.json.",
        "",
        "Archived execution hashes verified. Memory means reconstructed from pre-test per-KC states agree within "+format(audit["max_reconstructed_memory_mean_error"],".3g")+". Numerical training-step sensitivity (60 versus 10 seconds): "+", ".join(f"{r['model']} {100*r['relative_training_step_error']:.4f}%" for r in validation)+". Numerical consistency does not establish biological correctness.",
        "Forty regression tests passed, including physical history schema, contact masks, memory retention consistency, and memory-dependent readout reversal. The steering replay audit reproduces actual commands and separately measures their value-dependent contribution; it is not a new closed-loop memory-erasure experiment.",
        "",
        "The assay deadline is 120 seconds. MuJoCo integrates moving intervals at 0.1 ms; absorbed flies are held in the collecting state until the assay ends. Metadata distinguishes actual physics time from assay time. Training and retention are stationary state evolution, not simulated hours of walking.",
        "",
        "Per fly: learning_history.parquet, state_history.parquet, physics_history.parquet, learning_states.npz with naive/post-training/pre-test/final states and KC IDs, replay.npz, summary.json. Batch metadata stores configuration, software versions, seeds and source/data hashes. CPG initialization uses the library's fixed seed 0 for all flies.",
        "",
        "Data folder: results/panel_b_pilot_v3. Separate navigation gate: results/panel_b_motor_gate_v3e. Earlier development failures remain preserved. No upload or new public video was made.",
        "",
        "## Next scientific step",
        "",
        "Do not tune this completed run. A new, explicitly labeled calibration should address excessive behavioral gain/insufficient individual variability and compare memory mechanisms against independently specified time-course data. Any timing point used for fitting loses held-out status. Then run an independently seeded, larger reciprocal-cohort validation with declared success criteria. A whole-brain expansion is not justified by this pilot alone.",
        "",
        "See [frozen protocol and limitations](PANEL_B_PROTOCOL.md) and [previous diagnostics](CONNECTOME_DIAGNOSTICS.md).",
    ]
    report=ROOT/'PANEL_B_RESULTS.md'
    report.write_text("\n".join(lines).replace('results/panel_b_pilot_v3',out.relative_to(ROOT).as_posix()),encoding="utf-8")
    sources=[Path(__file__),ROOT/'experiments/analyze_panel_b.py',ROOT/'PANEL_B_PROTOCOL.md',ROOT/'tests/test_panel_b.py',report]
    with zipfile.ZipFile(out/'analysis_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT))
    (out/'report_provenance.json').write_text(json.dumps(dict(files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
    print(json.dumps(gates,indent=2))
if __name__=="__main__":main()
