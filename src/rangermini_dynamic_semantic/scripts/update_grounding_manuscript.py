#!/usr/bin/env python3
"""Create a new working draft from measured crossed-task data, preserving v1."""
import argparse,json,re
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();ws=Path(__file__).resolve().parents[3];paper=ws/'paper';data=json.loads((root/'output/summary.json').read_text())
 old=(paper/'vlm_route_manuscript.tex').read_text();rows=data['aggregate'];cs={c['condition']:c for c in data['controls']}
 totals={m:sum(r['instruction_met'] for r in rows if r['method']==m) for m in ['geometric','fixed','vlm']}
 body=old
 abstract=r'''\begin{abstract}
Indoor document-delivery robots can encounter several geometrically feasible
bypasses whose suitability depends on the current delivery instruction.
We implement instruction-conditioned visual route selection above a shared
multi-mode MPPI controller for a Ranger-style mobile robot. Feasible motion
prefixes are projected into the actual robot image. DeepSeek selects a bounded
route identifier using the image and instruction, while current scan and pose
feedback remain responsible for local motion. A crossed Gazebo experiment
places a waiting person and an equipment trolley on opposite sides of an
obstacle, swaps their positions, and requests clearance for either object.
The 36 recorded runs all reach the navigation destination. Route preferences
are satisfied in %V%/12 visual-selection runs, %F%/12 fixed person-policy runs,
and %G%/12 geometric-selection runs. On identical image/candidate packets,
full input matches the registered route in %CF%/12 controlled calls and
image removal matches it in %CN%/12 calls. The results evaluate task adaptation
relative to a fixed person policy, not superiority over task-aware rules.
The candidate-selection approach follows VLM-MPPI; the present evidence concerns
its planar service-navigation implementation, not a new steering controller
or complete physical document handover.
\end{abstract}'''
 abstract=abstract.replace('%V%',str(totals['vlm'])).replace('%F%',str(totals['fixed'])).replace('%G%',str(totals['geometric'])).replace('%CF%',str(cs['full']['matches_registered_route'])).replace('%CN%',str(cs['no_image']['matches_registered_route']))
 body=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:abstract,body,flags=re.S)
 body=body.replace('Visual Route Selection for Document-Delivery Navigation','Instruction-Conditioned Visual Route Selection for Delivery')
 body=body.replace('Only this instruction\nand the en-route navigation stage are tested.','The initial placement experiment tests this instruction. The crossed follow-up\nalso requests space for equipment handling beside a trolley; both evaluate\nonly the en-route navigation stage.')
 body=body.replace('identifiers, their feasibility flags, colours, and predicted geometric\nclearances.','identifiers, their feasibility flags, and colours. The initial experiment\nalso supplied predicted geometric clearances; the crossed follow-up omits\nthese numeric values to avoid attributing a geometry-only cue to vision.')
 body=re.sub(r'\\begin\{figure\*\}.*?\\end\{figure\*\}', '',body,flags=re.S)
 body=body.replace('(Fig.~\\ref{fig:north})','').replace('(Fig.~\\ref{fig:south})','')
 intro='''This study implements a bounded semantic-to-motion interface on a shared
planar controller and tests it through a crossed task/layout protocol.
The protocol distinguishes route adaptation to a named object from the
behaviour of an instruction-independent person-clearance policy. Matched-image
controls separately remove visual input and the target-specific instruction.
These are implementation and evaluation contributions; no new MPPI optimizer
or general superiority over task-aware rule systems is claimed.
'''
 begin=body.index('This study documents');end=body.index('\\section{Problem',begin);body=body[:begin]+intro+'\n'+body[end:]
 table='\n'.join(f"{r['layout'][0].upper()} & {'Person' if r['target']=='person' else 'Trolley'} & {'Geom.' if r['method']=='geometric' else 'Person rule' if r['method']=='fixed' else 'Visual'} & {r['instruction_met']}/3 & ${r['gap_mean']:.3f}\\pm{r['gap_sd']:.3f}$ \\\\" for r in rows)
 controls='\n'.join(f"{c['condition'].replace('_',' ')} & {c['matches_registered_route']}/12 & {c['task_pair_route_flips']}/6 \\\\" for c in data['controls'])
 extension=r'''\subsection{Crossed Task and Layout Experiment}
The follow-up preserves the furnished world and relocates its existing trolley
assembly to the opposite side from the person. The person is at $(4.9,-0.95)$ m
and trolley at $(4.9,-3.30)$ m in layout N, with these locations swapped in S.
All methods, instructions, and seeds within a layout use byte-identical world
copies. One instruction requests distance from the waiting student; the other
requests distance from the equipment trolley to leave room for equipment
access and handling. These are route preferences, not a claim that either
object may violate geometric collision constraints.

For each layout, the instructed target determines the independently registered
preferred bypass. Two layouts, two instructions, three selectors and three
seeds give 36 physical runs. The fixed baseline retains its RGB-D person
policy for both instructions. It is therefore a specified static service
policy, not a task-aware detector-and-rule system. This distinction is essential
to interpreting the comparison.

\begin{table}[t]
\centering\caption{Crossed-task motion results. All cells arrive 3/3.
Preference is route agreement; gap is minimum target-proxy clearance (m),
mean $\pm$ sample SD over three planning seeds.}
\label{tab:crossed}\footnotesize
\begin{tabular}{lllcc}\toprule
Layout & Target & Selector & Preference & Gap\\\midrule
%TABLE%
\bottomrule\end{tabular}\end{table}

The evaluator uses the same person proxy as before and a $0.70\times0.42$ m
planar trolley bounding rectangle. The trolley metric is a conservative
bounding proxy, not a full mesh/contact measurement. Each candidate package
sent to the model contains only ID, colour and feasibility; no target location
or correct route label is supplied. The model is queried once before departure,
and the same geometric controller performs all subsequent motion.

\subsection{Matched-Image Input Controls}
For each layout, one actual recorded robot image and one candidate packet are
held identical while the two target-specific instructions are alternated.
Three calls per layout/instruction/input condition give 36 additional actual
API calls: full image and instruction, instruction without image, and image
with a neutral instruction that names no preferred object. No-image requests
contain no object locations. The protocol and expected sides were stored
before these calls, after the initial geometric scene check.

\begin{table}[t]\centering
\caption{Matched-input controls. Route matching is descriptive agreement with
the registered target side. Neutral instructions have no target-specific
correctness label; their matching count is diagnostic only. Flip counts refer
to six paired target instructions within the same image.}
\label{tab:controls}\footnotesize
\begin{tabular}{lcc}\toprule
Input & Route matching & Paired flips\\\midrule
%CONTROLS%
\bottomrule\end{tabular}\end{table}

These controls test this two-layout setting, not general visual reasoning.
Repeated calls and planning seeds do not create independent semantic scenes.

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{%FIGROOT%/north_person_en.pdf}
\caption{Layout N, person-clearance instruction. Top: actual camera and
candidate projections. Bottom: actual same-episode overhead frames and measured
trajectories. Representative seed17; all seeds appear in Table~\ref{tab:crossed}.}
\label{fig:taskperson}\end{figure*}
\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{%FIGROOT%/north_trolley_en.pdf}
\caption{Same layout and destination, trolley-clearance instruction. The
fixed person policy does not change with this instruction. Candidate geometry
and the motion backend remain shared. This comparison concerns a fixed
policy, not all possible task-aware rules.}
\label{fig:tasktrolley}\end{figure*}
'''.replace('%TABLE%',table).replace('%CONTROLS%',controls).replace('%FIGROOT%',str(root/'output'))
 body=body.replace('\\section{Discussion}',extension+'\n\\section{Discussion}')
 b=body.index('The fixed-rule result limits');e=body.index('\\begin{thebibliography}',b)
 endtext=r'''The initial person-only matrix shows that a detector-and-depth rule is
sufficient for one clearance instruction. The crossed protocol changes the
preferred object while holding physical geometry fixed, and thus tests a
separate requirement: adapting the route to the current service instruction.
The fixed person policy does not encode that change. A task-aware rule supplied
with reliable object detections and instruction branches could make the same
registered choices as the visual model; the experiment does not exclude that
alternative. The input controls further restrict conclusions to the tested
image/instruction combinations rather than an unconstrained model capability.

The simulator uses a single-body VelocityControl proxy and visual wheels,
not a validated steering actuator or wheel--ground dynamics model. Localization
uses Gazebo pose rather than an independent estimator. Two static layouts and
two instructions do not establish operation around moving pedestrians, across
laboratories, or during complete document handover. Numerical clearance
assessments concern specified evaluation proxies and sampled trajectories.

\section{Conclusion}
Instruction-conditioned route selection has been executed in the furnished
Gazebo laboratory with actual DeepSeek image calls and continuous geometric
MPPI motion. The initial 18 runs establish the person-policy reference; the
additional 36 runs and 36 matched-input calls evaluate object-specific task
adaptation and the dependence of observed choices on the supplied inputs.
This evidence supports a bounded planar service-navigation implementation and
its measured task behaviour. It does not establish a new chassis controller,
complete delivery success, or superiority over task-aware rules.

'''
 body=body[:b]+endtext+body[e:]
 (paper/'vlm_task_conditioned_manuscript.tex').write_text(body)
 print(paper/'vlm_task_conditioned_manuscript.tex')
if __name__=='__main__':main()
