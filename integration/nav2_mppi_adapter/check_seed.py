"""Check fixed-observation first-solve seeding in two fresh processes."""
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if '--child' in sys.argv:
    lib=C.CDLL(str(ROOT/'integration/nav2_mppi_adapter/build/libranger_nav2_mppi.so'))
    D=C.POINTER(C.c_double);U=C.POINTER(C.c_ubyte)
    lib.nav2_mppi_create.argtypes=[C.c_int,C.c_int,C.c_double,C.c_double,C.c_int]
    lib.nav2_mppi_create.restype=C.c_void_p
    lib.nav2_mppi_solve.argtypes=[C.c_void_p,D,D,U,C.c_int,C.c_int,C.c_double,C.c_double,C.c_double,D,C.c_int,D,D]
    p=lib.nav2_mppi_create(1000,40,.12,.35,17)
    if not p:raise RuntimeError('Cannot initialize official optimizer')
    state=np.zeros(3);velocity=np.zeros(3);grid=np.zeros((161,161),np.uint8)
    path=np.column_stack((np.linspace(0,2,41),np.zeros(41),np.zeros(41))).astype(float)
    command=np.zeros(4);trajectory=np.zeros((40,3))
    n=lib.nav2_mppi_solve(p,state.ctypes.data_as(D),velocity.ctypes.data_as(D),grid.ctypes.data_as(U),161,161,.05,-4.025,-4.025,path.ctypes.data_as(D),41,command.ctypes.data_as(D),trajectory.ctypes.data_as(D))
    print(json.dumps({'n':n,'command':command.tolist()}))
    lib.nav2_mppi_destroy.argtypes=[C.c_void_p];lib.nav2_mppi_destroy(p)
else:
    values=[]
    for _ in range(2):
        p=subprocess.run([sys.executable,__file__,'--child'],capture_output=True,text=True)
        if p.returncode:raise RuntimeError(p.stderr+p.stdout)
        values.append(next(json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')))
    report={'seed':17,'fresh_process_results':values,'same_first_solve':values[0]==values[1],
            'scope':'Fixed-observation first solve, not deterministic Gazebo closed-loop replay'}
    (ROOT/'artifacts/nav2_omni_qualification_20261021/seed_check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
    assert report['same_first_solve']
