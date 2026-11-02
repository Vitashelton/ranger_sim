#!/usr/bin/env python3
"""Update submission candidate from frozen 48-run evidence and held-out 9-run data."""
import json,re,shutil
from pathlib import Path
ws=Path(__file__).resolve().parents[3];paper=ws/'paper';root=ws/'artifacts/submission_readiness_20261013';bundle=root/'submission_bundle'
s=(paper/'vlm_grounded_submission_draft.tex').read_text()
s=s.replace('Task-Object-Grounded MPPI Route Selection','Visual Semantics for Multimode MPPI Navigation')
s=s.replace('The destination alone does not express this preference. The engineering\nquestion addressed here is how to select among available bypass behaviours\nusing the current visual scene while retaining a geometric motion controller.',
'''The destination alone does not express this preference. Conversely, a distant
person beyond the destination should not trigger a needless bypass near other
objects. The engineering question is how to respect the specified passing
preference while retaining geometric efficiency. We address it using current
visual evidence, measured range, and a shared local motion controller.''')
s=s.replace('This study implements a bounded semantic-to-motion interface on a shared\nplanar controller and tests it through a crossed task/layout protocol.\nThe protocol distinguishes route adaptation to a named object from the\nbehaviour of an instruction-independent person-clearance policy. Matched-image\ncontrols separately remove visual input and the target-specific instruction.\nThese are implementation and evaluation contributions; no new MPPI optimizer\nor general superiority over task-aware rule systems is claimed.',
'''The implemented method maps the specified task object to current visual
evidence, uses measured depth to score the shared bypass guides, and leaves
motion execution to the existing multimode MPPI controller. This provides
an inspectable interface from service preference to physically executed
navigation. The evaluation combines switched person/trolley preferences,
matched-input controls, and a distant-object case with a task-aware image
rule. Together, these tests examine preference fulfilment and unnecessary
detours while keeping the execution backend shared.''')
range_root=ws/'artifacts/delivery_range_validation_20261013'
data=json.loads((range_root/'output/summary.json').read_text());rs={r['method']:r for r in data['aggregate']}
# Preserve the detailed crossed-condition table as a source appendix.
pattern=r'\\begin\{table\}\[t\].*?\\label\{tab:crossed\}.*?\\end\{table\}'
match=re.search(pattern,s,flags=re.S)
if match:
 (bundle/'paper/crossed_condition_table.tex').write_text(match.group())
 s=s[:match.start()]+s[match.end():]
s=s.replace('all-seed method totals appear in Table~\\ref{tab:grounded}.','all-seed method totals appear in Table~\\ref{tab:grounded}.')
s=s.replace('tested service preferences, not superiority over task-aware rules, a new MPPI\noptimizer, steering-actuator validity or completed document handover.',
'tested service preferences. In a held-out nine-run scene with the person beyond\nthe navigation goal, all selectors meet the person-gap criterion. Mean motion\ntimes are 27.41 s for a task-aware image rule and 13.91 s for metric preference.\nThis result concerns the complete metric selection policy, including geometric\ntie handling, and does not establish general depth or speed superiority.')
s=s.replace('Two mirrored layouts, two instructions and three planning seeds yield 48 unique\ncompleted motion records across four selectors.',
'Two mirrored layouts, two instructions and three planning seeds yield 48 unique\nmotion records across four selectors. A third held-out range scene adds nine\nrecords, giving 57 completed navigation runs.')
s=s.replace('k^*&=\\arg\\max_{k\\in\\mathcal F_t}G_k .','k^*&=\\arg\\max_{k\\in\\mathcal F_t}G_k .')
s=s.replace('The same motion-mode bank and execution controller are used by all selectors.',
'In the range extension, scores differing by at most $10^{-9}$ m are treated\nas numerical ties and the shared MPPI objective breaks them. This does not\nchange the distinct-score choices in the earlier 12 grounded runs.\nThe same motion-mode bank and execution controller are used by all selectors.',1)
s=s.replace('The initial experiment\nalso supplied predicted geometric clearances; the crossed follow-up omits\nthese numeric values to avoid attributing a geometry-only cue to vision.','The crossed comparison supplies no numeric target positions or predicted\ngeometric clearances to the direct route-ID client.')
rows=[]
for method,name in [('geometric','Geometric'),('image_rule','Image rule'),('grounded','Metric')]:
 r=rs[method];rows.append(f"{name} & 3/3 & {r['gap_mean']:.3f} & {r['time_mean']:.2f} $\\pm$ {r['time_sd']:.2f} & {r['path_mean']:.3f} "+chr(92)*2)
extension=r'''\subsection{Held-Out Range Case with a Task-Aware Rule}
The post-audit protocol places the person at $(8.8,-0.95)$ m, beyond the
navigation goal, while retaining the nearby trolley on the other side.
The instruction requests space for waiting people along the delivery route.
Three planning seeds each compare geometric selection, a task-aware image-side
rule, and metric preference, giving nine new physical runs. These are not
retuned or substituted records from the earlier matrix.

The image rule uses the same target-localization model, prompt and requested
category as metric preference, choosing the bypass opposite the detected box
centre. It has task awareness and visual object evidence but no measured range.
Both methods make fresh localization calls, so their returned boxes need not be
identical. All methods share the same world, guides and executing controller.

\begin{table}[t]\centering\footnotesize
\caption{Held-out range case. All methods arrive and pass the registered
person-gap threshold 3/3. Gap is mean minimum person-proxy clearance (m);
time is mean $\pm$ sample SD (s); path is mean sampled length (m).}
\label{tab:range}
\begin{tabular}{lcccc}\toprule
Selector & Gap pass & Gap & Motion time & Path\\\midrule
ROWS
\bottomrule\end{tabular}\end{table}

The distant target gives equal nominal minimum gap on the two merging guides.
Metric preference therefore uses the common geometric objective as tie breaker
and executes LEFT. The image-side rule chooses RIGHT near the trolley, where
the same MPPI backend incurs longer travel and local-planning stops.
Both satisfy the person requirement; a different bypass is not labelled a
task failure. This case supports avoiding an unnecessary semantic detour in
the stated setting, not general superiority of depth, population-level speed
gains, or a comparison against every task-aware geometric rule.

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{RANGEFIG}
\caption{Actual RGB, candidates, same-episode overhead frames and measured
routes for the held-out remote-person case. Representative seed17; the table
reports all three seeds. Motion time excludes initial API waiting.}
\label{fig:range}\end{figure*}

'''.replace('ROWS','\n'.join(rows)).replace('RANGEFIG',str(range_root/'output/comparison_en.pdf'))
s=s.replace('\\section{Discussion}',extension+'\\section{Discussion}')
s=s.replace('The 48 unique\ncrossed-task motion records and 36 matched-input calls evaluate object-specific',
'The 48 crossed-task records, nine held-out range records and 36 matched-input\ncalls evaluate object-specific')
out=paper/'vlm_submission_candidate.tex';out.write_text(s)
packet=bundle/'paper'
for target in ['person','trolley']:
 name=f'north_{target}_en.pdf';shutil.copy2(ws/'artifacts/delivery_semantic_grounding_20261013/output_grounded'/name,packet/'figures'/name)
shutil.copy2(range_root/'output/comparison_en.pdf',packet/'figures/range_en.pdf')
s=s.replace(str(ws/'artifacts/delivery_semantic_grounding_20261013/output_grounded')+'/','figures/').replace(str(range_root/'output/comparison_en.pdf'),'figures/range_en.pdf')
(packet/'main.tex').write_text(s)
print(out)

