"""C ABI for MPPI numerical kernels. Handles stay process-local, never pickled."""
import ctypes as ct
import os
from pathlib import Path
import numpy as np

_lib = None
MODES = ('STOPPED', 'DUAL_ACKERMANN', 'PARALLEL', 'SPINNING')
Double7 = ct.c_double * 7
Double10 = ct.c_double * 10
Double2 = ct.c_double * 2

def library():
    global _lib
    if _lib is None:
        configured = os.environ.get('RANGER_MPPI_NATIVE_LIBRARY')
        path = Path(configured) if configured else Path(__file__).resolve().parents[3]/'integration/mppi_native_core/build/libranger_mppi_native.so'
        if not path.is_file():
            raise RuntimeError('Build native MPPI kernels with integration/mppi_native_core/build.py, or set RANGER_MPPI_NATIVE_LIBRARY')
        lib = ct.CDLL(str(path))
        ptr = ct.POINTER(ct.c_double)
        lib.ranger_native_abi.restype = ct.c_int
        if lib.ranger_native_abi() != 1:
            raise RuntimeError('Incompatible MPPI native ABI')
        lib.ranger_wheel_step.argtypes = [ptr, ptr, ptr]
        lib.ranger_wheel_step.restype = None
        lib.ranger_clearance.argtypes = [ptr, ct.c_size_t] + [ct.c_double]*5 + [ptr]
        lib.ranger_clearance.restype = None
        _lib = lib
    return _lib

def wheel_step(model, vx, vy, wz, dt):
    # ctypes arrays are not retained in planner state because spawn workers
    # serialize that state.
    params = Double7(model.wheelbase, model.track, model.max_speed,
                     model.max_steer_rate, model.max_ackermann_angle,
                     model.max_parallel_angle, model.deadband)
    mode = 'PARALLEL' if model.mode == 'SIDE_SLIP' else model.mode
    state = Double10(vx, vy, wz, dt, MODES.index(mode), model.transition_pending, *model.angles)
    out = Double10()
    library().ranger_wheel_step(params, state, out)
    model.mode = MODES[int(out[3])]
    model.transition_pending = bool(out[4])
    model.angles = list(out[6:10])
    return tuple(out[:3]), model.mode, tuple(model.angles), bool(out[5])

def prepare_points(points):
    return np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)

def clearance(x, y, yaw, points, length, width):
    pts = prepare_points(points)
    out = Double2()
    library().ranger_clearance(pts.ctypes.data_as(ct.POINTER(ct.c_double)),
                              len(pts), x, y, yaw, length, width, out)
    return out[0], out[1]
