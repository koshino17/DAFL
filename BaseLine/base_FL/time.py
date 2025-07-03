import time
from functools import wraps
from collections import defaultdict
import cProfile, pstats, io
GLOBAL_STEP_TIMES = defaultdict(list)

# def timeit_step(name: str):
#     def decorator(fn):
#         @wraps(fn)
#         def wrapper(self, *args, **kwargs):
#             start = time.perf_counter()
#             result = fn(self, *args, **kwargs)
#             elapsed = time.perf_counter() - start
#             # 將耗時存到 self.step_times
#             self.step_times.setdefault(name, []).append(elapsed)
#             print(f"[Timing] {name}: {elapsed:.4f}s")
#             return result
#         return wrapper
#     return decorator

def timeit_step(name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            out   = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start

            if args and hasattr(args[0], "step_times"):
                args[0].step_times.setdefault(name, []).append(elapsed)
            else:   # free function：存到全域或忽略
                GLOBAL_STEP_TIMES.setdefault(name, []).append(elapsed)

            return out
        return wrapper
    return decorator

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"[Timing] {func.__name__} took {t1-t0:.2f}s")
        return result
    return wrapper

def auto_timeit(cls):
    for name, attr in cls.__dict__.items():
        if callable(attr) and not name.startswith("_"):
            setattr(cls, name, timeit_step(name)(attr))
    return cls

def profile_fn(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = fn(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(5)
        print(f"[Profile] {fn.__name__}\n{s.getvalue()}")
        return result
    return wrapped