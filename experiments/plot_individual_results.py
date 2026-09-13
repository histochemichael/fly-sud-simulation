"""Actual individual outcomes: categorical dots and continuous latency violins."""
from pathlib import Path
import json,hashlib
import sys,platform,scipy
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact,binomtest,mannwhitneyu,wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/individual_results';OUT.mkdir(exist_ok=True)
ORDER=['untrained','paired','unpaired','dan_silenced'];LABELS=['Untrained','Paired','Unpaired','DAN off']
COLORS=['#7497B1','#D99222','#34AAA7','#9570BA']
df=pd.read_csv(ROOT/'results/physical_batch/choices.csv').sort_values(['condition','fly_id'])
assert len(df)==400 and not df.duplicated(['condition','fly_id']).any()
groups={c:df[df.condition==c].set_index('fly_id') for c in ORDER}
assert all(len(g)==100 for g in groups.values())
tests=[]
for c in ['untrained','unpaired']:
    a=groups['paired'];b=groups[c];tab=[[int((g.choice=='B').sum()),int((g.choice=='A').sum())] for g in [a,b]]
    r=fisher_exact(tab,alternative='two-sided')
    tests.append(dict(endpoint='choice',comparison='Paired vs '+LABELS[ORDER.index(c)],test='Fisher exact (two-sided)',n1=sum(tab[0]),n2=sum(tab[1]),statistic=float(r.statistic),p=float(r.pvalue),details={'table_B_A':tab,'excludes':'nonresponses'}))
a=groups['paired'];b=groups['dan_silenced'];assert np.array_equal(a.random_seed,b.random_seed)
loss=int(((a.choice=='B')&(b.choice!='B')).sum());gain=int(((a.choice!='B')&(b.choice=='B')).sum())
tests.append(dict(endpoint='choice',comparison='Paired vs DAN off',test='Exact McNemar (two-sided)',n1=100,n2=100,statistic=min(loss,gain),p=float(binomtest(loss,loss+gain).pvalue),details={'B_loss':loss,'B_gain':gain,'endpoint':'B vs not-B, including nonresponse','matched_pairs':100}))
for c in ['untrained','unpaired']:
    a=groups['paired'].latency_s.dropna();b=groups[c].latency_s.dropna();r=mannwhitneyu(a,b,alternative='two-sided',method='asymptotic',use_continuity=True)
    tests.append(dict(endpoint='latency',comparison='Paired vs '+LABELS[ORDER.index(c)],test='Mann-Whitney U (two-sided)',n1=len(a),n2=len(b),statistic=float(r.statistic),p=float(r.pvalue),details={'method':'asymptotic, tie-corrected, continuity correction','population':'responders only'}))
pair=groups['paired'][['latency_s']].join(groups['dan_silenced'][['latency_s']],lsuffix='_paired',rsuffix='_dan').dropna()
diff=np.round(pair.latency_s_paired-pair.latency_s_dan,4);r=wilcoxon(diff,zero_method='wilcox',alternative='two-sided',method='approx',correction=False)
tests.append(dict(endpoint='latency',comparison='Paired vs DAN off',test='Wilcoxon signed-rank (two-sided)',n1=len(pair),n2=len(pair),statistic=float(r.statistic),p=float(r.pvalue),details={'matched_complete_pairs':len(pair),'nonzero_differences':int((diff!=0).sum()),'method':'normal approximation; zero differences dropped; no continuity correction','difference_rounding_s':0.0001,'population':'both conditions responded; symmetry of differences assumed'}))
ps=np.array([t['p'] for t in tests]);ix=np.argsort(ps);adj=np.empty(len(ps));adj[ix]=np.minimum(1,np.maximum.accumulate(ps[ix]*(len(ps)-np.arange(len(ps)))))
for t,q in zip(tests,adj):t['p_holm_six_tests']=float(q)
summary=[]
for c,g in groups.items():
    counts={k:int((g.choice==k).sum()) for k in ['A','B','none']};v=g.latency_s.dropna()
    summary.append(dict(condition=c,n=len(g),**counts,n_latency=len(v),median_latency_s=float(v.median()),pi_responders=(counts['B']-counts['A'])/(counts['B']+counts['A'])))
(OUT/'statistics.json').write_text(json.dumps({'source':'results/physical_batch/choices.csv','source_sha256':hashlib.sha256((ROOT/'results/physical_batch/choices.csv').read_bytes()).hexdigest(),'summary':summary,'tests':tests,'adjustment':'Holm across all six exploratory comparisons; no new independent batches','violin':'Scott KDE, trimmed to observed extrema; each dot is one original fly','latency_caveat':'Conditional on response; nonresponses are not assigned a latency of 2 seconds. Not a censoring-aware population latency analysis.'},indent=2))
pd.DataFrame([{k:v for k,v in t.items() if k!='details'} for t in tests]).to_csv(OUT/'statistics.csv',index=False)
def pf(x):return f'{x:.3g}' if x>=.001 else f'{x:.2e}'
def plot(kind):
    plt.rcParams.update({'font.size':14,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(14,9),facecolor='white');ax=fig.add_axes([.105,.36,.86,.47])
    fig.text(.045,.947,'ACTUAL SIMULATION RESULTS',fontsize=23,fontweight='bold',color='#172A3F')
    fig.text(.045,.90,'Individual choices (one dot per fly)' if kind=='choice' else 'Choice latency: violin + every responding fly',fontsize=19,color='#172A3F')
    rng=np.random.default_rng(3042)
    for i,c in enumerate(ORDER):
        g=groups[c];s=summary[i]
        if kind=='choice':
            for y,k in [(-1,'A'),(0,'none'),(1,'B')]:
                n=int((g.choice==k).sum());cols=min(10,n);xs=(np.arange(n)%max(1,cols)-(cols-1)/2)*.055
                ys=y+(np.arange(n)//max(1,cols)-(int(np.ceil(n/max(cols,1)))-1)/2)*.043
                ax.scatter(i+xs,ys,s=26,color=COLORS[i],edgecolors='white',linewidths=.25,zorder=3)
            label=f'{LABELS[i]}\nn=100 | A {s["A"]}, B {s["B"]}\nNo choice {s["none"]}'
        else:
            v=g.latency_s.dropna().to_numpy();parts=ax.violinplot(v,positions=[i],widths=.78,showextrema=False,bw_method='scott')
            for body in parts['bodies']:body.set_facecolor(COLORS[i]);body.set_alpha(.24);body.set_edgecolor(COLORS[i])
            ax.scatter(i+rng.uniform(-.23,.23,len(v)),v,s=15,color=COLORS[i],alpha=.82,edgecolors='white',linewidths=.2,zorder=3)
            med=np.median(v);ax.plot([i-.16,i+.16],[med,med],color='#172A3F',lw=2,zorder=4)
            label=f'{LABELS[i]}\nn={len(v)} / 100\nNo choice {s["none"]}'
        ax.text(i,-.08,label,ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=12)
    ax.set_xlim(-.55,3.55);ax.set_xticks([]);ax.grid(axis='y',alpha=.15)
    if kind=='choice':ax.set_yticks([-1,0,1],['A','No choice','B']);ax.set_ylim(-1.35,1.35);ax.set_ylabel('Recorded categorical outcome')
    else:ax.set_ylim(0,2.08);ax.set_ylabel('First-choice latency (s)')
    fig.text(.045,.225,'COMPARISON / SAMPLE / TEST / p (RAW) / p (HOLM, 6 TESTS)',fontsize=12,fontweight='bold')
    for j,t in enumerate([t for t in tests if t['endpoint']==kind]):
        n=f'{t["n1"]} matched pairs' if 'matched_pairs' in t['details'] or 'matched_complete_pairs' in t['details'] else f'n={t["n1"]}, {t["n2"]}'
        fig.text(.045,.186-j*.035,f'{t["comparison"]} | {n} | {t["test"].replace(" (two-sided)","")} | {pf(t["p"])} | {pf(t["p_holm_six_tests"])}',fontsize=12)
    foot='All tests two-sided and exploratory. Fisher: responders only. McNemar: B vs not-B in all 100 matched pairs.' if kind=='choice' else 'Responders only; DAN comparison uses complete matched pairs. Nonresponses are not imputed as 2 s.'
    fig.text(.045,.050,foot,fontsize=11,color='#506278')
    fig.text(.045,.025,'Dots are original flies, not timesteps, bootstraps or biological replicates. Jitter is visual only.' if kind=='choice' else 'Violin = smoothed observed distribution (Scott KDE); dots = flies; line = median. One master-seed batch.',fontsize=11,color='#506278')
    fig.savefig(OUT/f'{kind}.png',dpi=160);plt.close(fig)
plot('choice');plot('latency')
(OUT/'plot_metadata.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':{'scipy':scipy.__version__,'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},'dot_jitter_seed':3042,'violin_bandwidth':'Scott','data_source':'results/physical_batch/choices.csv','uploaded':False},indent=2))
print(json.dumps(tests,indent=2))
