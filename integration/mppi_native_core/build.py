"""Build the stateless native motion/geometry kernels and record provenance."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
subprocess.run(['cmake', '-S', str(root), '-B', str(root/'build'),
                '-DCMAKE_BUILD_TYPE=Release'], check=True)
subprocess.run(['cmake', '--build', str(root/'build'), '-j', '2'], check=True)
paths = [root/'core.cpp', root/'CMakeLists.txt', root/'build/libranger_mppi_native.so']
(root/'provenance.json').write_text(json.dumps({
    'abi': 1, 'compiler': subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
    'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
}, indent=2)+'\n')
