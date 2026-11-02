"""Detach a frozen campaign at an episode boundary and finish its analysis."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def process_rows():
    rows = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            tail = path.read_text().rsplit(')', 1)[1].split()
            rows[int(path.parent.name)] = (int(tail[1]), tail[0])
        except (OSError, ValueError, IndexError):
            continue
    return rows


def descendants(pid, rows):
    found = {pid}
    while True:
        newer = found | {p for p, (parent, _) in rows.items() if parent in found}
        if newer == found:
            return found - {pid}
        found = newer


def online():
    try:
        with socket.create_connection(('api.deepseek.com', 443), timeout=3):
            return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt', type=int, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    state_file = batch/'supervisor_state.json'

    def state(label, **extra):
        value = dict(state=label, supervisor_pid=os.getpid(),
                     updated_wall_time=time.time(), **extra)
        temporary = state_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(state_file)
        print(label, flush=True)

    # Reuse the explicitly authorized credential from this campaign's own
    # children. It is never printed, written to disk or put in command arguments.
    key = os.environ.get('DEEPSEEK_API_KEY')
    for pid in descendants(args.adopt, process_rows()):
        if key:
            break
        try:
            fields = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
            key = next((x.split(b'=', 1)[1].decode() for x in fields
                        if x.startswith(b'DEEPSEEK_API_KEY=')), None)
        except OSError:
            pass
    if not key:
        raise SystemExit('Campaign credential unavailable; original launcher unchanged')
    os.environ['DEEPSEEK_API_KEY'] = key
    os.kill(args.adopt, signal.SIGSTOP)
    state('draining_current_episodes', previous_launcher=args.adopt)
    adopted = False
    try:
        deadline = time.monotonic()+480
        while True:
            rows = process_rows()
            children = [p for p, (parent, status) in rows.items()
                        if parent == args.adopt and status != 'Z']
            if not children:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Episode drain exceeded existing campaign wall limit')
            time.sleep(2)
        # All currently active episodes have ended before ownership changes.
        os.kill(args.adopt, signal.SIGTERM)
        os.kill(args.adopt, signal.SIGCONT)
        adopted = True
        while args.adopt in process_rows() and process_rows()[args.adopt][1] != 'Z':
            time.sleep(.2)
        protocol = json.loads((batch/'protocol.json').read_text())
        partial = [t['name'] for t in protocol['trials']
                   if (batch/t['name']).exists()
                   and not (batch/t['name']/'episode/result.json').exists()]
        if partial:
            state('retained_incomplete_attempt_needs_review', attempts=partial)
            return
        while not online():
            state('waiting_for_visual_api_network')
            time.sleep(10)
        command = [sys.executable, str(ROOT/'tools/run_task_condition_benchmark.py'),
                   '--output', str(batch)]
        with (batch/'tmux_campaign.log').open('a') as log:
            worker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state('running_in_tmux', campaign_pid=worker.pid)
            paused = False
            while worker.poll() is None:
                connected = online()
                if not connected and not paused:
                    os.kill(worker.pid, signal.SIGSTOP)
                    paused = True
                    state('network_lost_new_episode_dispatch_paused', campaign_pid=worker.pid)
                elif connected and paused:
                    os.kill(worker.pid, signal.SIGCONT)
                    paused = False
                    state('running_in_tmux', campaign_pid=worker.pid)
                time.sleep(5)
        if worker.returncode:
            state('campaign_exit_requires_review', return_code=worker.returncode)
            return
        env = dict(os.environ, PYTHONNOUSERSITE='1')
        for script, rest in [
            ('summarize_task_condition_benchmark.py', [str(batch)]),
            ('render_task_condition_benchmark.py', [str(batch), '--output',
                 str(ROOT/'paper/figures/task_condition_20261023')])]:
            subprocess.run(['/usr/bin/python3', str(ROOT/'tools'/script), *rest],
                           cwd=ROOT, env=env, check=True)
        state('finished_analysis_and_figures')
    finally:
        if not adopted:
            try:
                os.kill(args.adopt, signal.SIGCONT)
            except ProcessLookupError:
                pass


if __name__ == '__main__':
    main()
