"""UTC event timestamps with monotonic sub-second ordering in one API process.

Some Windows clocks return equal datetime.now values for rapid consecutive
events. Anchor a high-resolution monotonic timer to UTC at process startup.
"""
from datetime import datetime, timezone, timedelta
from time import perf_counter_ns

_anchor_utc = datetime.now(timezone.utc)
_anchor_counter = perf_counter_ns()

def utc_now() -> str:
    return (_anchor_utc + timedelta(microseconds=(perf_counter_ns() - _anchor_counter) // 1000)).isoformat()
