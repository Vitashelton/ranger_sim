"""Persistent spawn worker: the existing planner, RNG and commitments are retained.

ROS objects never enter the child. Only explicit planner configuration/state and
method arguments cross the pipe. A synchronous RPC releases the parent's GIL;
ROS sensor callbacks belong to separate callback groups in the parent executor.
"""
import multiprocessing as mp
import time
import traceback
import cloudpickle


def _worker(connection, serialized):
    planner = cloudpickle.loads(serialized)
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            name, args, kwargs, updates = request
            planner.__dict__.update(updates)
            start = time.monotonic()
            try:
                value = getattr(planner, name)(*args, **kwargs)
                state = {k: v for k, v in planner.__dict__.items() if k != 'guide_support' and not k.startswith('_runtime_')}
                connection.send((True, value, state, start, time.monotonic()))
            except Exception:
                connection.send((False, traceback.format_exc(), {}, start, time.monotonic()))
    finally:
        connection.close()


class PlannerProcess:
    METHODS = {'bank', 'behavior_plan', 'note_execution', '_scan_points_world'}

    def __init__(self, planner, emit=None, timeout=30.):
        object.__setattr__(self, '_state', dict(planner.__dict__))
        object.__setattr__(self, '_dirty', {})
        object.__setattr__(self, '_emit', emit or (lambda **kw: None))
        object.__setattr__(self, '_timeout', timeout)
        ctx = mp.get_context('spawn')
        parent, child = ctx.Pipe()
        process = ctx.Process(target=_worker, args=(child, cloudpickle.dumps(planner)),
                              name='ranger_mppi_worker', daemon=True)
        process.start()
        child.close()
        object.__setattr__(self, '_pipe', parent)
        object.__setattr__(self, '_process', process)

    def __getattr__(self, name):
        if name in self.METHODS:
            return lambda *a, **kw: self._call(name, a, kw)
        return self._state[name] if name in self._state else self._missing(name)

    def _missing(self, name):
        raise AttributeError(name)

    def __setattr__(self, name, value):
        self._state[name] = value
        self._dirty[name] = value

    def _call(self, name, args, kwargs):
        started = time.monotonic()
        self._emit(event='planner_rpc_begin', method=name, worker_pid=self._process.pid)
        self._pipe.send((name, args, kwargs, self._dirty.copy()))
        self._dirty.clear()
        if not self._pipe.poll(self._timeout):
            raise TimeoutError('MPPI worker response deadline exceeded')
        ok, value, state, begin, end = self._pipe.recv()
        self._emit(event='planner_rpc_end', method=name, worker_pid=self._process.pid,
                   worker_begin_monotonic_s=begin, worker_end_monotonic_s=end,
                   rpc_wall_s=time.monotonic()-started)
        if not ok:
            raise RuntimeError(value)
        self._state.update(state)
        return value

    def close(self):
        if self._process.is_alive():
            self._pipe.send(None)
            self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
        self._pipe.close()
