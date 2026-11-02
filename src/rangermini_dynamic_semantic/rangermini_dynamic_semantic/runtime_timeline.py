"""Clock domains are explicit; monotonic time correlates processes on this host."""
import json
import os
import threading
import time
from pathlib import Path


class RuntimeTimeline:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, 'w', buffering=1)
        self.lock = threading.Lock()

    def emit(self, **fields):
        row = dict(wall_monotonic_s=time.monotonic(), system_time_ns=time.time_ns(),
                   pid=os.getpid(), thread_id=threading.get_native_id(), **fields)
        with self.lock:
            self.file.write(json.dumps(row, allow_nan=True)+'\n')

    def close(self):
        with self.lock:
            self.file.close()
