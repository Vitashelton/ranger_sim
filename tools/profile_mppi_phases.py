#!/usr/bin/env python3
"""Exclusive phase timings from recorded-state replay; production code unchanged.

In-memory AST scopes time original statements, preserving their order and RNG.
This is NOT a retrospective breakdown of an uninstrumented online callback.
"""
import argparse, ast, contextlib, dataclasses, functools, gzip, hashlib, inspect
import json, math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic import observation_mppi as om, interaction_mppi as im
from rangermini_dynamic_semantic.stable_interaction_mppi import StableInteractionMPPI
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph,Track
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig

PHASES=('sampling','rollout','collision','cost','postprocess')

class Timer:
    def __init__(self):self.reset()
    def reset(self):
        self.totals=defaultdict(float);self.counts=defaultdict(int)
        self.current=None;self.stack=[];self.last=time.perf_counter()
    def enter(self,name):
        now=time.perf_counter()
        if self.current:self.totals[self.current]+=now-self.last
        self.stack.append(self.current);self.current=name;self.counts[name]+=1
        self.last=time.perf_counter()
    def leave(self):
        now=time.perf_counter();self.totals[self.current]+=now-self.last
        self.current=self.stack.pop();self.last=time.perf_counter()
    @contextlib.contextmanager
    def scope(self,name):
        self.enter(name)
        try:yield
        finally:self.leave()

T=Timer()

def scope(statements,name):
    enter=ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='_phase_timer',ctx=ast.Load()),attr='enter',ctx=ast.Load()),args=[ast.Constant(value=name)],keywords=[]))
    leave=ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='_phase_timer',ctx=ast.Load()),attr='leave',ctx=ast.Load()),args=[],keywords=[]))
    # A single AST statement packages entry + try/finally without generators.
    node=ast.If(test=ast.Constant(value=True),body=[enter,ast.Try(body=statements,handlers=[],orelse=[],finalbody=[leave])],orelse=[])
    return ast.copy_location(node,statements[0])

def blocks(items):
    result=[];pending=[];phase=None
    for node,key in items:
        if key!=phase and pending:
            result.append(scope(pending,phase));pending=[]
        phase=key;pending.append(node)
    if pending:result.append(scope(pending,phase))
    return result

def wrap(fn,default,transform=None):
    # Read original class method source; no generated source is persisted over it.
    import textwrap
    tree=ast.parse(textwrap.dedent(inspect.getsource(fn)))
    function=tree.body[0]
    if transform:transform(function)
    function.body=[scope(function.body,default)]
    namespace=dict(fn.__globals__,_phase_timer=T)
    exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(fn),'exec'),namespace)
    return namespace[fn.__name__]

def base_evaluate(function):
    loop=next(n for n in function.body if isinstance(n,ast.For) and 'range(self.cfg.horizon)' in ast.unparse(n.iter))
    inner=[]
    for node in loop.body:
        text=ast.unparse(node)
        if isinstance(node,ast.If) and text.startswith('if sequence is not None'):
            inner.append((node,'sampling'))
        elif isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='fraction':
            modified=[]
            for sub in node.body:
                txt=ast.unparse(sub)
                phase='collision' if ('_clearance_cost' in txt or txt.startswith('clearance_min =') or txt.startswith('if clearance <')) else 'rollout'
                modified.append((sub,phase))
            node.body=blocks(modified);inner.append((node,'rollout'))
        elif text.startswith(('clearance_cost +=','if reference_trajectory:','visibility =','view_cost +=')):
            inner.append((node,'cost'))
        else:inner.append((node,'rollout'))
    loop.body=blocks(inner)
    index=function.body.index(loop)
    function.body[index+1:]=[scope(function.body[index+1:],'cost')]

def interaction_evaluate(function):
    out=[]
    for node in function.body:
        txt=ast.unparse(node)
        if isinstance(node,ast.If) and txt.startswith('if len(self.static_boxes)'):
            out.append(scope([node],'collision'))
        elif isinstance(node,ast.For) and 'self.tracks'==ast.unparse(node.iter):
            node.body=blocks([(sub,'cost' if ast.unparse(sub).startswith('dynamic_cost +=') else 'collision') for sub in node.body])
            out.append(node)
        else:out.append(node)
    function.body=out

def behavior_prepare(function):
    out=[]
    for node in function.body:
        txt=ast.unparse(node)
        phase='collision' if (txt.startswith('points = self._scan_points_world') or txt.startswith('if tracks:') or
            (isinstance(node,ast.If) and 'support is not None' in txt)) else 'sampling'
        # The call into plan has its own nested exclusive timing.
        out.append((node,phase))
    function.body=blocks(out)

def nested(x):return tuple(nested(v) for v in x) if isinstance(x,list) else x

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if 'time_ms' not in k}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return x.tolist()
    return x

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=ROOT/'artifacts/executor_audit_20261021/process_verified')
    parser.add_argument('--output',type=Path,default=ROOT/'post/group_meeting_20261021/mppi_runtime')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    ep=args.run/'episode';manifest=json.loads((ep/'manifest.json').read_text())
    with gzip.open(ep/'planner_observations.jsonl.gz','rt') as f:observations=[json.loads(l) for l in f]
    routes=[json.loads(l) for l in (ep/'route_guidance.jsonl').read_text().splitlines()]
    traces=[json.loads(l) for l in (ep/'trace.jsonl').read_text().splitlines()]
    graph=SpatialGraph(args.run/'world.sdf');cfg=MPPIConfig(**manifest['mppi_config'])
    originals={(om.ObservationMPPI,'_evaluate'):om.ObservationMPPI._evaluate,
               (om.ObservationMPPI,'plan'):om.ObservationMPPI.plan,
               (im.InteractionMPPI,'_evaluate'):im.InteractionMPPI._evaluate,
               (im.InteractionMPPI,'behavior_plan'):im.InteractionMPPI.behavior_plan,
               (im.InteractionMPPI,'bank'):im.InteractionMPPI.bank}
    patched={(om.ObservationMPPI,'_evaluate'):wrap(om.ObservationMPPI._evaluate,'rollout',base_evaluate),
             (om.ObservationMPPI,'plan'):wrap(om.ObservationMPPI.plan,'postprocess'),
             (im.InteractionMPPI,'_evaluate'):wrap(im.InteractionMPPI._evaluate,'cost',interaction_evaluate),
             (im.InteractionMPPI,'behavior_plan'):wrap(im.InteractionMPPI.behavior_plan,'sampling',behavior_prepare),
             (im.InteractionMPPI,'bank'):wrap(im.InteractionMPPI.bank,'postprocess')}
    # Functions using zero-argument super() need the original __class__ cell.
    # Bind that cell by replacing super() with explicit super(original class,self).
    # wrap() above compiles standalone methods; interaction override handled below.
    def patch_super(fn,cls,default,transform):
        import textwrap
        tree=ast.parse(textwrap.dedent(inspect.getsource(fn)));f=tree.body[0]
        class ExplicitSuper(ast.NodeTransformer):
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name) and node.func.id=='super' and not node.args:
                    node.args=[ast.Name(id='_original_class',ctx=ast.Load()),ast.Name(id='self',ctx=ast.Load())]
                return node
        tree=ExplicitSuper().visit(tree)
        transform(f);f.body=[scope(f.body,default)]
        ns=dict(fn.__globals__,_phase_timer=T,_original_class=cls)
        exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(fn),'exec'),ns)
        return ns[f.name]
    patched[(im.InteractionMPPI,'_evaluate')]=patch_super(originals[(im.InteractionMPPI,'_evaluate')],im.InteractionMPPI,'cost',interaction_evaluate)
    def install(mapping):
        for (cls,name),fn in mapping.items():setattr(cls,name,fn)
    def planner_for(d):
        p=StableInteractionMPPI(cfg,prediction=True,static_boxes=graph.static_boxes)
        p.guide_support=functools.partial(graph.supports_pose,margin=.10)
        p.obs=dataclasses.replace(p.obs,arrival_radius=.24)
        p.wait_action_penalty=p.obs.stop_action_penalty
        p.rng.setstate(nested(d['rng_state']))
        # Recover previous executed-family commitments from the recorded trace.
        for row in traces:
            if row['timestamp']>=d['sim_time']:break
            plan=row.get('plan')
            if plan:p.note_execution(plan,row['timestamp'],tuple(row['command']))
        latest=[r for r in routes if r['sim_time']<=d['sim_time']+1e-8]
        if latest:
            r=latest[-1];p.route_goal=r['guide'];p.route_yaw=r['guide_yaw'];p.reference_path=r['path']
        return p
    def run_bank(d):
        p=planner_for(d)
        scan=([float('inf') if v is None else v for v in d['scan']['ranges']],d['scan']['angle_min'],d['scan']['angle_increment'])
        act=d['actuation'];tracks=[Track(**r) for r in d['tracks']]
        T.reset();start=time.perf_counter()
        with T.scope('postprocess'):
            result=p.bank(d['pose'],d['goal'][0],scan,tracks,d['sim_time'],
                current_mode=act['mode'],goal_yaw=d['goal'][1],current_steering_angles=act['wheel_steering_rad'],
                current_steering_transition=act['steering_transition_stop'])
        total=(time.perf_counter()-start)*1000
        return result,total,{k:T.totals[k]*1000 for k in PHASES},dict(T.counts)
    rows=[]
    try:
        # One warm-up, excluded, same recorded input.
        run_bank(observations[0])
        for i,d in enumerate(observations):
            install(originals);baseline,plain,_,_=run_bank(d)
            install(patched);profiled,instrumented,phases,counts=run_bank(d)
            if clean(baseline)!=clean(profiled):raise AssertionError(f'profiling changed output at snapshot {i}')
            rows.append(dict(index=i,sim_time=d['sim_time'],unprofiled_ms=plain,instrumented_ms=instrumented,
                phases_ms=phases,scope_entries=counts,semantic_behaviors=len(profiled),
                motion_candidates=sum(c['plan']['candidate_count'] for c in profiled),
                tracks=len(d['tracks']),obstacle_points=len(d['scan']['ranges']),outputs_identical=True))
            print(i,round(plain,1),round(instrumented,1),flush=True)
    finally:install(originals)
    summaries={k:dict(p50_ms=float(np.percentile([r['phases_ms'][k] for r in rows],50)),
                        p95_ms=float(np.percentile([r['phases_ms'][k] for r in rows],95)),
                        mean_ms=float(np.mean([r['phases_ms'][k] for r in rows]))) for k in PHASES}
    summary=dict(kind='Recorded-state replay with exclusive phase instrumentation; not online callback timing',
        n_snapshots=len(rows),input_run=str(args.run.relative_to(ROOT)),config=dataclasses.asdict(cfg),
        static_boxes=len(graph.static_boxes),phase_summary=summaries,
        unprofiled_total_ms={f'p{q}':float(np.percentile([r['unprofiled_ms'] for r in rows],q)) for q in [50,95]},
        instrumented_total_ms={f'p{q}':float(np.percentile([r['instrumented_ms'] for r in rows],q)) for q in [50,95]},
        instrumentation_overhead_p50_ms=float(np.median([r['instrumented_ms']-r['unprofiled_ms'] for r in rows])),
        instrumentation_accounting='Phase timers exclude time spent in timer bookkeeping; unprofiled baseline and instrumented wall time are both retained. Residual call/scaffolding overhead and cache effects remain.',
        phase_sum_ms={f'p{q}':float(np.percentile([sum(r['phases_ms'].values()) for r in rows],q)) for q in [50,95]},
        percentile_note='Each phase percentile is computed across snapshots; sums of percentiles need not equal total percentiles.',
        scope_note='Collision includes scan preprocessing, registered-map SAT and dynamic footprint-gap geometry. Cost includes objective arithmetic. Sampling includes behavior-reference preparation. Postprocess includes family weighting, selection and candidate packaging; excludes ROS, IPC, A*, VLM and recording.',
        replay_limitations='Sensor states, route references and RNG are recorded; motion-family commitments reconstructed from prior trace timestamps. Not a bit-exact replay of asynchronously updated online node state.',
        profiling_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={str(Path(m.__file__).relative_to(ROOT)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in [om,im]},
        observations_sha256=hashlib.sha256((ep/'planner_observations.jsonl.gz').read_bytes()).hexdigest(),rows=rows)
    (args.output/'runtime_profile.json').write_text(json.dumps(summary,indent=2))
    import csv
    with (args.output/'runtime_profile.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['phase','p50_ms','p95_ms','mean_ms'])
        for key,value in summaries.items():writer.writerow([key,value['p50_ms'],value['p95_ms'],value['mean_ms']])
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','source_sha256']},indent=2))

if __name__=='__main__':main()
