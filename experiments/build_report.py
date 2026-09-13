"""Representative-figures PDF: readable portrait pages, verified by rendering."""
from pathlib import Path
import csv,json
from io import BytesIO
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output'
pdfmetrics.registerFont(TTFont('Calibri','C:/Windows/Fonts/calibri.ttf'))
pdfmetrics.registerFont(TTFont('Calibri-Bold','C:/Windows/Fonts/calibrib.ttf'))
c=canvas.Canvas(str(OUT/'representative_figures.pdf'),pagesize=(612,792))
c.setTitle('Fly ethanol reward: silent video summary, evidence and proposed next steps')
c.setAuthor('Local simulation experiment')
style=ParagraphStyle('body',fontName='Calibri',fontSize=11,leading=15,textColor=HexColor('#24344A'))
small=ParagraphStyle('small',parent=style,fontSize=9,leading=12)
def para(text,y,smalltext=False,x=54,w=504):
    p=Paragraph(text,small if smalltext else style);_,h=p.wrap(w,700);p.drawOn(c,x,y-h);return y-h-12
def start(n,title,sub):
    c.drawImage(str(OUT/'assets/impulse_neuro_logo.png'),54,720,width=60,height=50,preserveAspectRatio=True,anchor='c')
    c.setFillColor(HexColor('#172A3F'))
    c.setFont('Calibri-Bold',16);c.drawString(128,750,'Impulse Neuro')
    c.setFont('Calibri',8);c.drawString(128,733,'FLY ETHANOL REWARD / REPRESENTATIVE FIGURES')
    c.setFillColor(HexColor('#172A3F'));c.setFont('Calibri-Bold',24);c.drawString(54,694,title)
    para(sub,675,True)
    c.setStrokeColor(HexColor('#D4DDE4'));c.line(54,49,558,49)
    c.setFont('Calibri',9);c.drawString(54,33,'Impulse Neuro | September 2026 | Simulation, not biological validation')
    c.drawRightString(558,33,str(n))
def table(data,y,widths):
    t=Table(data,colWidths=widths)
    t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Calibri'),('FONTNAME',(0,0),(-1,0),'Calibri-Bold'),('FONTSIZE',(0,0),(-1,-1),10),('BACKGROUND',(0,0),(-1,0),HexColor('#EAF0F4')),('TEXTCOLOR',(0,0),(-1,-1),HexColor('#24344A')),('BOTTOMPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),9),('LINEBELOW',(0,0),(-1,-1),.4,HexColor('#D4DDE4'))]))
    _,h=t.wrap(504,700);t.drawOn(c,54,y-h);return y-h-20
def picture(im,x,y,w,h):
    c.drawImage(ImageReader(im),x,y,width=w,height=h,mask='auto')
rows={r['condition']:r for r in csv.DictReader((ROOT/'results/physical_batch/summary.csv').open())}
start(1,'A learned cue moves a physical fly','Outcome summary | 400 physical test trials; 100 individuals per condition')
y=para('The learned olfactory signal now drives the six-leg NeuroMechFly gait in MuJoCo. A compact synthetic KC/MBON/DAN circuit replaces the earlier scalar memory in this separate experiment. Original planar results remain unchanged.',626)
data=[['Condition','A','B','No response','PI (responders)']]
for key,name in [('untrained','Untrained'),('paired','Paired'),('unpaired','Unpaired'),('dan_silenced','DAN silenced')]:
    r=rows[key];data.append([name,r['a'],r['b'],r['nonresponse'],f"{float(r['pi_responders']):.3f}"])
y=table(data,y,[154,50,50,100,150])
y=para('<b>Main result.</b> Paired flies made 68 B choices among 91 completed choices (74.7%; PI 0.495, 95% Wilson interval 0.298 to 0.651). Including all 100 flies and scoring nonresponses as zero gives PI 0.45. Nonresponses are never reassigned to a preferred arm.',y)
y=para('<b>Exploratory comparisons.</b> Paired versus untrained: Fisher exact p = 0.000193; paired versus unpaired: p = 0.000615. Matched retrieval-DAN silencing caused 29 B-choice losses and no gains (exact paired p = 3.73 × 10<super>-9</super>). These tests are unadjusted and conditional on this model configuration.',y)
y=para('<b>What this demonstrates.</b> Online associative/reward-modulated learning can alter subsequent physical navigation under the configured assumptions. It is not deep learning, a whole-brain model, or a biologically complete model of addiction.',y)
para('The three examples on the following pages are predetermined fly 000 representatives, not outcome-selected successes. Batch training uses stationary cue exposures; these examples additionally include physical walking during training. All batch tests are physical.',y,True)
c.showPage()
train=Image.open(OUT/'branded_revision/training.png');test=Image.open(OUT/'branded_revision/test.png')
captions={
'Untrained':'No cue-reward pairing is supplied. Initial and final learned weights remain zero. Innate odor attraction and motor noise still produce movement. This representative makes no choice within the 2-second test window; the film retains that nonresponse.',
'Paired':'Odor B coincides with the teaching reward in each conditioning session. B-sensitive KC weights increase online; A-sensitive weights remain near zero. At the training snapshot, active KC and MBON populations accompany the DAN signal. Test reward is zero while learned weights remain fixed.',
'Unpaired':'Ethanol/reward occurs in a separate air slot before the odor exposures. Because odor-active KCs and reward do not coincide, the learning rule leaves both odor memories at zero. This models temporal pairing, not ethanol pharmacology.'}
for i,name in enumerate(['Untrained','Paired','Unpaired']):
    start(i+2,name+' / training and test','Figure '+str(i+1)+' | Actual body replay beside recorded circuit state')
    c.setFont('Calibri-Bold',12);c.drawString(54,617,'Training: session 3');c.drawString(312,617,'Reward-free test: 1.60 s')
    box=(32+624*i,200,640+624*i,1028)
    picture(train.crop(box),54,266,246,335);picture(test.crop(box),312,266,246,335)
    y=para(captions[name],252)
    y=para('Illustrative circuit paths now animate over JFRC2 reference anatomy. Their brightness uses logged KC activity, KC weights and DAN state. Moving dots depict flow, not individual spikes. Path positions and shapes are not anatomically registered. Numeric bars remain available as a check.',y,True)
    para('The large overhead view follows the body. A corner map shows complete gray group paths and the live yellow fly path, with odor sides aligned. A white ring marks the original first choice, but recording continues to 2 seconds. Shading marks choice areas, never reward delivery. Training shows a training path against test-group context.',y,True)
    c.showPage()
start(5,'Dopamine-silenced control','Figure 4 | Matched paired training, then synthetic retrieval gating disabled')
c.setFont('Calibri-Bold',12);c.drawString(54,617,'Training: normal teaching signal');c.drawString(312,617,'Test: DAN retrieval gate off')
def clip_still(phase,frame):
    return Image.open(OUT/'branded_revision'/f'dan_{phase}.png').crop((1132,200,1740,1028))
picture(clip_still('training',650),54,266,246,335)
picture(clip_still('test',162),312,266,246,335)
y=para('Training is identical to paired conditioning and preserves the learned weights. Only the retrieval gate is silenced in the test. The saved MBON rates still represent weighted KC input, but their contribution to steering is disabled; innate attraction and motor noise remain.',252)
y=para('In the 100-fly matched batch, B choices fall from 68 to 39, with 47 A choices and 14 nonresponses after silencing. This is an intervention in a programmed mechanism, not an independently inferred dopamine function.',y)
para('The reference anatomy stays unchanged. DAN paths dim and its rate bar drops to zero during testing. This represents a disabled model gate, not biological dopamine depletion. Weighted MBON input can remain visible even though its contribution to steering is disabled.',y,True)
c.showPage()
start(6,'Simulation beside Kaun et al.','Figure 5 | Qualitative comparison, not a numerical replication or fitted paper curve')
c.setFont('Calibri-Bold',13);c.drawString(54,618,'Our physical simulation');c.drawString(312,618,'Kaun et al. (2011)')
left=[('Conditioned preference','Paired: PI 0.495 among responders. Unpaired: PI −0.012. The effect is measured immediately after compressed training.'),('Protocol','Three sessions; four 0.25-second slots each. Two-choice physical arena; reward-free test up to 2 seconds.'),('Replication and metric','100 individuals per condition. PI = (B − A)/(A + B); nonresponses also reported separately and in all-fly PI.'),('Dopamine perturbation','Synthetic DAN retrieval gating is disabled in a seed-matched batch. Acquisition requires a teaching signal by construction.')]
right=[('Time-dependent effect','Figure 1b: aversion at 30 minutes (N = 8, p = 0.006); preference at 24 hours (N = 8, p = 0.02).'),('Protocol','Three spaced sessions with 10-minute odor exposures, ethanol paired to one odor, and one-hour spacing.'),('Replication and metric','N denotes replicate groups, not individual flies. CPI averages reciprocal odor-training groups; spatial randomization alone is not equivalent.'),('Dopamine perturbation','Figure 4 supports a retrieval requirement, but blocking acquisition or consolidation did not disrupt preference. Our acquisition rule is therefore not a faithful match.')]
for j,((a,b),(aa,bb)) in enumerate(zip(left,right)):
    yy=585-j*113
    para('<b>'+a+'</b><br/>'+b,yy,False,54,238)
    para('<b>'+aa+'</b><br/>'+bb,yy,False,312,238)
para('Paper bar heights are neither digitized nor invented. The simulation omits delayed aversion/preference transitions, consolidation, mushroom-body subtype specificity and punishment-resistant seeking. Behavioral agreement is limited to a paired-cue preference under different timing and measurement.',123,True)
c.showPage()
start(7,'Methods, provenance and files','Reproducibility notes | No upload or public release has occurred')
y=para('<b>Mechanism.</b> Bilateral A/B sensors feed saturating AL/PN rates and 24 synthetic KC identities. Reward modulates online KC-to-MBON weight changes during training. Bilateral output contrast and motor noise modulate gait amplitude. A/B identities are generic; ethanol is a scheduled scalar. No source coordinates are supplied to the circuit.',625)
y=para('<b>Time resolution.</b> MuJoCo uses 0.0001-second ticks; sensors and circuit update every 0.01 second. Each physics row joins a held control state. The original scored batch contains 4,818,271 physical rows and 168,346 controller rows. Supplemental records continue after choice to 2 seconds. Batch training is stationary; filmed training and all tests include body dynamics.',y)
y=para('<b>Body and hardware.</b> The installed body has 69 segments and 42 actuated joint degrees of freedom (7 per leg), driven by six coupled tripod-gait oscillators. These are not 42 behaviors. Local host: Intel i7-9700F, 8 cores, approximately 64 GB RAM; RTX 5060 Ti installed. Eight CPU workers ran MuJoCo physics; no GPU-accelerated physics is claimed.',y)
y=para('<b>Verification.</b> Thirteen unit tests passed. Across all 400 saved flies, the audit verified zero naive weights, frozen test weights, zero test reward, valid state joins, monotonic physics time and finite poses. Representative body recordings are rendered from saved qpos. This is engineering validation, not validation against biological ground truth.',y)
y=para('<b>Release-oriented records.</b> Per-fly Parquet tables retain positions, headings, sensors, rewards, neural activity, weights, actions, phase/trial/session and outcomes. NPZ snapshots retain naive, learned and final weights plus fixed projections. JSON/CSV files preserve configurations, versions, seeds, source hashes and statistics. See DATA_GUIDE.md for joins, units, reserved parameters and leakage warnings.',y)
y=para('<b>Limits.</b> One master-seed batch; no biological population fit or parameter-sensitivity study. Brain paths are illustrative, not anatomically registered. Some training walks cross the low perimeter; the replay preserves these poses rather than implying hard confinement. No FlyWire or whole-brain model was built.',y)
y=para('<b>Source.</b> Kaun KR, Azanchi R, Maung Z, Hirsh J, Heberlein U. A Drosophila model for alcohol reward. <i>Nature Neuroscience</i> 14, 612–619 (2011). DOI: 10.1038/nn.2805.<br/><link href="https://pubmed.ncbi.nlm.nih.gov/21499254/" color="#225C87">PubMed: article and published figure descriptions</link>',y,True)
para('Current deliverables: branded_revision/fly_ethanol_reward_impulse_neuro.mp4 (silent, 3:20); eight condition/phase clips; this report; DATA_GUIDE.md. Earlier supplemental runs observed post-choice motion while preserving original outcomes and exact state prefixes. This branding/editorial revision reran no experiments.',y,True)
c.showPage()
start(8,'A guide to the new silent video','Detailed viewing summary | All nine revised video files have zero audio streams')
data=[['Video time','What is shown'],['0:00-0:26','Experiment, paper, four conditions, model and hypothesis'],['0:26-0:56','Untrained, paired and unpaired training'],['0:56-1:16','Their reward-free tests, including post-choice movement'],['1:16-1:46','Paired versus DAN-control training: identical protocol'],['1:46-2:06','Paired versus retrieval-DAN-silenced test'],['2:06-2:20','Complete trajectories for all four groups'],['2:20-2:40','Actual choice dots, then latency violins, with statistics'],['2:40-2:52','What the configured model demonstrated'],['2:52-3:04','Paper agreement and explicit mismatches'],['3:04-3:20','Proposed validation before free-ranging studies']]
y=table(data,624,[103,401])
y=para('<b>Intro and hypothesis.</b> The first 26 seconds identify Kaun et al. (2011), all four conditions, 400 original tests, body/gait dimensions, hardware and circuit roles. The hypothesis is increased B preference after pairing and reduced bias when retrieval DAN is silenced. The logo remains at the top right.',y,True)
y=para('<b>Read the views.</b> Reward is uniform during scheduled training, never in a test arm. Gray is the full group; yellow follows fly 000. White marks first choice, not the end. The added DAN comparison shows intact learning during training and a disabled retrieval gate during test. The four-group plot uses complete recorded paths.',y,True)
para('<b>Read the circuit animation.</b> Reference anatomy is static; illustrative paths use model activity and weights. Flow dots are not spikes. Batch training is stationary; filmed training walks are representative examples. No biological activity recording or anatomical registration is implied.',y,True)
c.showPage()
start(9,'What matched the paper?','Evidence categories | Qualitative recapitulation is not quantitative replication')
y=para('<b>1. A qualitative behavioral pattern was recapitulated.</b> The configured model produces attraction to the previously rewarded odor in a reward-free test. The paired group favors B; untrained and unpaired controls are near even. This supports the sufficiency of this specified learning-and-navigation mechanism for a cue preference. It does not establish that the biological mechanism or effect size was recovered.',624)
y=para('<b>2. A pairing control was implemented.</b> Separating odor from reward removes the learned bias in this model. This is evidence that temporal coincidence matters under our learning rule, not proof that all alternative explanations have been excluded. In particular, no ethanol pharmacokinetics, intoxication or innate odor-identity asymmetry was fitted.',y)
y=para('<b>3. Retrieval-DAN dependence is only a partial analogy.</b> Turning off a programmed retrieval gate reduces B choice while retaining weights. The effect is expected from the implemented gate, so it is a useful intervention check but not an independent discovery of dopamine biology. Our acquisition rule also needs the teaching signal; the paper comparison on page 6 identifies the mismatch.',y)
y=para('<b>4. The central time course was NOT reproduced.</b> There is no modeled short-delay aversion, delayed preference transition, consolidation process or multi-day persistence. Running the existing weights for longer would not create those processes. The current immediate post-training test must not be described as a 24-hour test.',y)
y=para('<b>5. Quantitative and mechanistic replication remain untested.</b> The protocol timing, reward representation, arena and unit of replication differ. No paper means or raw data were fitted; error bars use a different statistical construction. Randomized spatial sides are not reciprocal odor-identity training. Molecular mechanisms, mushroom-body subtype functions and punishment-resistant seeking were not reproduced.',y)
para('<b>Recommended claim:</b> A compact online associative-learning model recapitulates paired-cue preference and a temporal-pairing control during reward-free physical navigation. It does not replicate the full Kaun et al. experiment or its delayed biological mechanisms. Source: Kaun et al. (2011), Fig. 1 and Fig. 4; link on page 7.',y,True)
c.showPage()
start(10,'Could we accelerate free-ranging flies?','Discussion only | A feasible modeling question, not a completed or authorized experiment')
y=para('<b>Yes, in principle - but we would simulate a hypothesis.</b> Free-ranging virtual flies could encounter odor and ethanol patches, acquire different exposure histories, and later be tested in a controlled choice assay. That would ask whether the proposed learning mechanism survives self-selected experience. It would not by itself reproduce addiction or show that real flies behave identically.',624)
y=para('<b>Keep three clocks separate.</b> Biological/model time controls absorption, recovery and memory. Wall-clock time is the computer runtime. Playback time is the movie speed. Faster rendering or relabelling three seconds as three hours does not simulate consolidation. Faster computation is legitimate only when the equations and relevant event timing retain their intended meaning.',y)
y=para('<b>The current model is missing the key slow states.</b> Candidate extensions might include an internal ethanol state, a short-lived aversive component and slower reward-memory dynamics. These are proposed hypotheses, not verified components. Their parameters and interactions would need independent calibration and held-out tests; simply inserting an aversion-to-preference switch at the target times would be circular.',y)
y=para('<b>Continuous full-body physics is expensive.</b> At the existing 0.0001-second timestep, one simulated day requires 864 million body steps per fly, or 86.4 billion for 100 flies. These are arithmetic workload estimates, not runtime benchmarks. Parallelism helps throughput but does not remove the work or justify increasing the timestep without validation.',y)
y=para('<b>A plausible acceleration strategy is multiscale.</b> Use the detailed body during encounters and tests, and an explicitly validated reduced movement model over long intervals. Slow internal states could use larger adaptive steps or exact updates where their equations permit. Check this approximation against full physics over overlapping intervals, particularly odor-boundary crossings and reinforcement timing.',y)
para('<b>Important boundary:</b> changing locomotion speed, odor diffusion or learning rates to make the video progress faster changes the experiment unless justified by a consistent model. No speedup factor, new parameter, simulation result or biological timescale has been established here.',y,True)
c.showPage()
start(11,'Proposed next step and decision gates','Not implemented | First establish a controlled time course, then test free-ranging exposure')
y=para('<b>Step 1 - Specify the claim before the arena.</b> Target the sign and magnitude of delayed cue preference under a matched conditioning protocol. Choose reciprocal odor assignments, exposure schedules, test delays, nonresponse rules and replicate-group statistics in advance. Use new independent seeds and reserve conditions or delays for validation.',624)
y=para('<b>Step 2 - Test a minimal time-dependent hypothesis.</b> Compare the existing associative baseline with candidate slow-state models under controlled exposures. Fit only a declared subset of observations. Demand predictions for held-out delays and perturbations; fitting the same points used to score success is not replication. An acquisition-DAN mismatch must be addressed explicitly, not hidden by the retrieval result.',y)
y=para('<b>Step 3 - Validate accelerated execution.</b> Run the same short scenarios with full-resolution and accelerated methods, comparing encounter durations, absorbed-dose histories, memory trajectories, choice distributions and nonresponses. Set tolerances before comparing. A faster method fails if it changes the scientific conclusion, even if it produces attractive trajectories.',y)
y=para('<b>Step 4 - Only then allow self-selected exposure.</b> Start with independent free-moving flies before adding social interactions. Local odor fields and ethanol availability would determine each fly\'s experience. Control for total dose and timing using matched or yoked exposure controls; otherwise a learned preference could merely reflect greater exposure or pre-existing location bias.',y)
y=para('<b>Step 5 - Return to a standardized reward-free assay.</b> Test after defined delays, with location and odor identity counterbalanced. Report individuals within replicate groups, uncertainty, nonresponses and every encounter history. A successful result would generalize beyond fitted conditions and remain stable across plausible parameters and independent batches.',y)
para('<b>Our proposed first decision:</b> a small controlled time-course model is more informative than immediately simulating a large free-ranging population. If it cannot predict the delayed transition without a built-in time switch, expanding the environment will not fix the missing mechanism. This page is a discussion agenda only; no new run, upload or whole-brain work was initiated.',y,True)
c.showPage()
start(12,'Visual provenance and reading notes','Reference images, original data and clear separation of evidence from proposals')
picture(Image.open(OUT/'assets/jfrc2_brain.png'),54,450,300,150)
y=para('<b>Reference anatomy plus an illustrative overlay.</b> JFRC2 is supplied by Virtual Fly Brain and derived from confocal anatomy. The background colors are not our activity. Added red circuit paths use logged KC rates; KC-to-MBON brightness additionally uses learned weights. DAN paths use the dopamine state. Locations are illustrative, not registered axons. Flow dots are not spikes. Adaptations: resizing and the identified circuit overlay.',425)
y=para('<b>Credit and reuse.</b> Arnim Jenett and colleagues, Janelia/FlyLight; Jenett et al. (2012), <i>Cell Reports</i> 2:991-1001, doi:10.1016/j.celrep.2012.09.011. BrainName annotation: Jenett, Shinomiya and Ito; Ito et al. (2014), <i>Neuron</i> 81:755-765, doi:10.1016/j.neuron.2013.12.017. Virtual Fly Brain; CC BY 4.0 with attribution as recorded in the source licensing notes.',y,True)
y=para('<link href="https://virtualflybrain.org/docs/data/templates/" color="#225C87">Virtual Fly Brain template description</link><br/><link href="https://github.com/VirtualFlyBrain/DrosAdultBRAINdomains/blob/master/LICENSE-NOTES.md" color="#225C87">Source image licensing and attribution notes</link><br/><link href="https://virtualflybrain.org/data/VFB/i/0001/7894/VFB_00017894/thumbnail.png" color="#225C87">Anatomical reference image</link>',y,True)
y=para('<b>Audit trail.</b> Original scores and replays remain in physical_batch and physical_pilot. Supplemental runs are in results/continuation_batch and results/continuation_pilot, with exact-prefix validation files. Latest media, credits and decode checks are in output/branded_revision. The body is rendered from recorded qpos; corner maps use recorded x/y. No original outcome was changed.',y)
para('<b>Status of this update.</b> Pages 1-9 summarize existing artifacts and evidence. Pages 10-11 are explicitly prospective discussion. No accelerated-time or free-ranging experiment was run. No new biological claim, fitted parameter, upload or whole-brain implementation is implied.',y,True)
c.showPage()
start(13,'Complete physical-test trajectories','Figure 6 | First choice is a marked event, not the end of the new recording')
picture(Image.open(OUT/'branded_revision/trajectories.png'),54,330,504,283.5)
y=para('Gray lines show all 100 supplemental test trajectories per displayed condition. Yellow shows the predetermined filmed fly 000, whose training includes walking. A white ring marks its original first-choice time; subsequent yellow motion is post-choice observation. Group paths are mirrored only to align odor sides with the filmed example.',308)
y=para('The observation window is 2 seconds for every fly unless physics fails. Recording continues beyond the original choice threshold, but choices and latencies stay latched at their first values. Later movement is excluded from the original preference statistics. Endpoints now mean the observation deadline, not arrival at an odor source.',y)
para('This physical arena is not a walled Y-maze. Choice-zone shading is not a wall, and no trajectory is masked at its edge. The earlier planar Y-maze data are preserved separately, but that stopped-at-choice figure is no longer used in this film. No paths were extrapolated or invented.',y,True)
c.showPage()
start(14,'Actual individual choice results','Figure 7 | One dot per original simulated fly; 100 per condition, 400 total')
picture(Image.open(OUT/'individual_results/choice.png'),54,280,504,324)
y=para('Choice is categorical: A, B, or no choice. The former bars and whiskers summarized group preference with Wilson confidence intervals; they were not box plots. One aggregate PI per condition cannot supply an individual-level violin. These dots expose every recorded outcome without inventing replicate preference indexes.',260)
y=para('Fisher exact tests compare B versus A among responders: paired n = 91, untrained n = 86, unpaired n = 85. For seed-matched paired versus DAN-off, exact McNemar compares B versus not-B in all 100 pairs, including nonresponses. There were 29 B losses and 0 gains. All tests are two-sided.',y)
para('The plot reports raw p and Holm-adjusted p across the six choice/latency comparisons on pages 14-15. These are exploratory analyses of one configured master-seed batch, not biological replication. Original p-values on page 1 remain raw and unchanged. Data: results/physical_batch/choices.csv; full precision and test statistics: output/individual_results/statistics.json.',y,True)
c.showPage()
start(15,'Actual first-choice latencies','Figure 8 | Violin density plus every responding individual; center line is the median')
picture(Image.open(OUT/'individual_results/latency.png'),54,280,504,324)
y=para('Responders: untrained n = 86, paired n = 91, unpaired n = 85, DAN-off n = 86, out of 100 each. The violin is a Scott-bandwidth density estimate trimmed to observed extrema, not additional data. Dots retain the recorded latency; horizontal jitter is visual only. Nonresponders are not assigned a latency of 2 seconds.',260)
y=para('Paired versus untrained/unpaired uses two-sided Mann-Whitney U with asymptotic tie correction and continuity correction. Paired versus DAN-off uses Wilcoxon signed-rank on 78 complete matched pairs, with differences rounded to the 0.0001-second simulation resolution; normal approximation, no continuity correction. This assumes symmetric paired differences.',y)
para('No latency comparison is significant at 0.05, before or after the six-test Holm adjustment. This does not establish equivalence. Latency tests are conditional on responding, especially the 78-pair DAN subset; they are not a censoring-aware analysis of the whole population. Sources: SciPy mannwhitneyu and wilcoxon documentation; original saved choices, not supplemental continuation endpoints.',y,True)
c.showPage()
start(16,'Statistical results and sample sizes','Table 2 | Two-sided exploratory tests; Holm correction over the six displayed comparisons')
stats=json.loads((OUT/'individual_results/statistics.json').read_text())
data=[['Endpoint / comparison','Test','Sample','Statistic','Raw p','Holm p']]
for t in stats['tests']:
    paired='matched_pairs' in t['details'] or 'matched_complete_pairs' in t['details']
    sample=f"{t['n1']} pairs" if paired else f"{t['n1']} / {t['n2']}"
    stat=('OR=' if t['test'].startswith('Fisher') else 'min=' if 'McNemar' in t['test'] else 'U=' if 'Mann' in t['test'] else 'W=')+f"{t['statistic']:g}"
    vals=[t['endpoint'].title()+'<br/>'+t['comparison'],t['test'].replace(' (two-sided)',''),sample,stat,f"{t['p']:.4g}",f"{t['p_holm_six_tests']:.4g}"]
    data.append([Paragraph(v,small) for v in vals])
y=table(data,622,[115,126,61,64,69,69])
y=para('<b>Read n correctly.</b> Every condition contains 100 original simulated individuals. Choice Fisher tests use responders; exact McNemar uses all 100 matched paired/DAN individuals and treats nonresponse as not-B. Latency uses observed responses only; the DAN test is restricted to the 78 individuals responding in both matched conditions.',y)
y=para('<b>Statistics.</b> Fisher reports an odds ratio (OR). Exact McNemar uses the smaller discordant count (min = 0; 29 B losses, 0 gains). U and W are rank-test statistics. Holm-adjusted values control the six-test family defined for this exploratory display, not every analysis ever run on this model. None of these results constitutes a biological replication.',y)
y=para('<b>Traceability.</b> Full-precision values, exact sample definitions and the SHA-256 of the original choices file are saved in output/individual_results/statistics.json. The CSV is a convenient summary. Original observations and learning states were not modified. No p-value was selected to make the model appear more successful.',y,True)
para('<b>Method documentation:</b> <link href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html" color="#225C87">SciPy Mann-Whitney U</link>; <link href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html" color="#225C87">SciPy Wilcoxon signed-rank</link>. Both latency analyses are conditional on response; they do not estimate a population-wide latency distribution with censoring.',y,True)
c.save();print(OUT/'representative_figures.pdf')
