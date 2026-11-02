#!/usr/bin/env python3
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();paper=Path(__file__).resolve().parents[3]/'paper'
original=json.loads((root/'output/summary.json').read_text());grounded=json.loads((root/'output_grounded/summary.json').read_text())
text=(paper/'vlm_task_conditioned_manuscript.tex').read_text()
hits=sum(r['instruction_met'] for r in grounded['aggregate'] if r['method']=='grounded')
abstract=r'''\begin{abstract}
Laboratory delivery can require different clearance preferences despite unchanged
geometric navigation goals. We implement a task-object-grounded interface above
a shared multi-mode MPPI controller. DeepSeek returns the visible image box of
the instructed person or equipment trolley. Measured depth supplies a surface
anchor; deterministic code scores shared bypass guides by target-proxy gap.
Two mirrored layouts, two instructions and three planning seeds yield 48 unique
completed motion records across four selectors. The grounded interface satisfies
the registered route preference in HIT/12 runs; direct visual route selection
does so in 11/12, and geometric selection and a fixed person policy in 6/12 each.
An original semantic failure is retained and examined in the follow-up.
Thirty-six matched-input API calls separately test the direct selector's image
and instruction dependence. This evidence concerns the bounded interface and
tested service preferences, not superiority over task-aware rules, a new MPPI
optimizer, steering-actuator validity or completed document handover.
\end{abstract}'''.replace('HIT',str(hits))
text=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:abstract,text,flags=re.S)
text=re.sub(r'\\title\{.*?\}\n\\author',lambda m:r'\title{Task-Object-Grounded MPPI Route Selection\protect\\for Laboratory Document Delivery}'+'\n'+r'\author',text,count=1,flags=re.S)
text=text.replace(r'\subsection{Visual Candidate Projection and Selection}',r'\subsection{Direct Route-ID Selection Reference}')
method=r'''\subsection{Task-Object Grounding and Metric Preference}
The grounded variant separates object interpretation from metric route choice.
Its supported taxonomy is person/trolley, explicitly bound to the two registered
instructions; it is not an arbitrary-language parser. DeepSeek receives raw RGB
and the requested category, and returns only target kind, normalized visible
box and reason. No route, metric goal or velocity is requested. Invalid or
absent boxes and insufficient depth stop the episode without a hidden fallback.

Valid box depth pixels are unprojected with the camera model. Points at or below
the 20th depth percentile are retained; coordinate-wise median planar position
gives a visible-surface anchor $\hat p_T$, not a calibrated object centre.
No simulator target pose is read. With $Q_k$ the dense points of shared guide $k$,
\begin{align}
 G_k&=\min_{q\in Q_k}\max\{0,\operatorname{dist}(\hat p_T,\mathcal R(q,0))-r_T\},\\
 k^*&=\arg\max_{k\in\mathcal F_t}G_k .
\end{align}
Guide spacing is approximately $0.05$ m; $r_T$ is $0.18$ m for people and
$0.41$ m for trolleys. These are preference proxies around a measured surface,
not certified object bounding sets. Scoring assumes zero guide heading;
actual MPPI uses current scans and its executing footprint. $\mathcal F_t$
contains current non-STOP prefixes, not globally certified complete guides.
The same motion-mode bank and execution controller are used by all selectors.

This interface exposes object evidence before route scoring. It follows the
broader semantic-conditioning idea of VLM-MPPI while replacing model-returned
route IDs with object localization and a metric preference. The contribution
is the bounded object-to-metric implementation and crossed-task evaluation,
not a new optimizer or a generally superior perception algorithm.

'''
text=text.replace(r'\subsection{Selection-to-Motion Execution}',method+r'\subsection{Selection-to-Motion Execution}')
table=[]
for name,src,selector in [('Geometric',original,'geometric'),('Person policy',original,'fixed'),('Direct visual',original,'vlm'),('Grounded target',grounded,'grounded')]:
 rs=[r for r in src['aggregate'] if r['method']==selector]
 table.append(f"{name} & {sum(r['arrived'] for r in rs)}/12 & {sum(r['instruction_met'] for r in rs)}/12 & {sum(r['gap_met'] for r in rs)}/12 "+chr(92)*2)
results=r'''\subsection{Grounded Follow-Up and Retained Failure}
In south/trolley/seed31, direct selection returns LEFT and calls the right-side
person an equipment area, although the wheeled trolley is visible on the left.
The robot arrives but misses the instructed preference. This original episode
is retained. A follow-up protocol registered after that observation evaluates
the grounded interface on the same two layouts, two tasks and three seeds:
12 additional physical episodes. The 24 geometric/person-policy episodes are
reused, not counted as new trials; the crossed comparison has 48 unique
completed records.

\begin{table}[t]\centering\footnotesize
\caption{Unique crossed-task motion records. Target gap uses independent
evaluation geometry and the registered $0.60$ m criterion.}
\label{tab:grounded}
\begin{tabular}{lccc}\toprule
Selector & Arrival & Preference & Target gap\\\midrule
ROWS
\bottomrule\end{tabular}\end{table}

The grounded interface localizes the visible trolley in the original failure
condition and selects RIGHT from its measured anchor and route scores.
Boxes, depth ROIs, anchors, scores and actual trajectories are retained.
This compares the complete interface change, including explicit target-category
prompting; it does not isolate box formatting or depth as the sole cause.
The small static suite does not establish a population-level advantage.
The matched-input controls concern direct selection, not a grounding ablation.
Two transport timeouts and an initial JSON-format request error are retained
separately from valid returns; repairs do not remove the semantic failure.

'''.replace('ROWS','\n'.join(table))
text=text.replace(r'\section{Discussion}',results+r'\section{Discussion}')
for target in ['person','trolley']:
 text=text.replace(str(root/'output')+f'/north_{target}_en.pdf',str(root/'output_grounded')+f'/north_{target}_en.pdf')
text=text.replace('Top: actual camera and\ncandidate projections.','Top: actual camera, candidate projections and target evidence box.')
text=text.replace('all seeds appear in Table~\\ref{tab:crossed}.','all-seed method totals appear in Table~\\ref{tab:grounded}.')
text=text.replace('The initial 18 runs establish the person-policy reference; the\nadditional 36 runs and 36 matched-input calls evaluate object-specific task\nadaptation and the dependence of observed choices on the supplied inputs.','The initial 18 runs establish the person-policy reference; the 48 unique\ncrossed-task motion records and 36 matched-input calls evaluate object-specific\nadaptation, a retained semantic failure and a grounded-interface follow-up.')
text=text.replace(r'\input{vlm_route_results.tex}','')
text=re.sub(r'\\subsection\{Observed Route Choices\}.*?(?=\\subsection\{Crossed Task and Layout Experiment\})','',text,flags=re.S)
text=text.replace('The matrix has two placement conditions, three selectors, and three planning\nseeds per cell: 18 runs. The seeds repeat planner sampling in the same static\ncondition; they are not independent semantic scenes. No completed run is\nexcluded. Earlier projection-development recordings are retained separately\nand are not included in this matrix.',
'''The crossed comparison has two layouts, two instructions and three seeds
per selector: 12 runs each for geometric, fixed person-policy, direct visual,
and grounded target selection, or 48 unique runs. The 24 baseline episodes are
reused in the follow-up comparison, not counted as new trials. Seeds repeat
sampling in the same static condition, not independent semantic scenes.
No valid completed decision is excluded; transport/interface failures are
retained separately. The earlier person-only development matrix is historical
evidence and is not included in these 48 records.''')
text=text.replace('Six actual calls used the requested\nand returned model identifier', 'The direct reference and grounded follow-up use the requested\nand returned model identifier')
text=text.replace(r'\caption{Visual selection with shared geometric execution}',r'\caption{Task-object-grounded route preference and shared execution}')
text=text.replace(r'\STATE Project visible prefixes onto the robot image'+'\n'+r'\STATE Request a bounded route identifier from DeepSeek'+'\n'+r'\STATE Validate the returned identifier and response fields',
r'''\STATE Request a bounded task-object box from current RGB
\STATE Validate target kind, box, and measured depth
\STATE Unproject the near-depth subset to a surface anchor
\STATE Score feasible guides by conservative target gap
\STATE Select the guide with the largest preference score''')
text=text.replace('The initial person-only matrix shows that a detector-and-depth rule is\nsufficient for one clearance instruction. The crossed protocol changes the',
'A detector-and-depth rule is sufficient for the person instruction in this\nsetting. The crossed protocol changes the')
text=text.replace('The initial 18 runs establish the person-policy reference; the 48 unique','The 48 unique')
path=paper/'vlm_grounded_delivery_manuscript.tex';path.write_text(text);print(path)

