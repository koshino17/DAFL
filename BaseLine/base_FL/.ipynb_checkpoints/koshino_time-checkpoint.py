import time
from functools import wraps
from collections import defaultdict

def timeit_step(name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            start = time.perf_counter()
            out = fn(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            # 建立或更新 step_times
            if not hasattr(self, "step_times"):
                self.step_times = defaultdict(list)
            self.step_times[name].append(elapsed)
            print(f"[Timing] {name}: {elapsed:.3f}s")
            return out
        return wrapped
    return decorator