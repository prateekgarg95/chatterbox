"""Scoped CPU-thread-count helper for `perth.PerthImplicitWatermarker`.

Real, measured root cause of a previously "unexplained" 5-600ms watermark
latency variance (chatterbox-turbo-hinglish-quantized project's own task
tracker, investigated 2026-08-17): `PerthImplicitWatermarker()` defaults to
`device="cpu"` and inherits the process-wide `torch.get_num_threads()`
(which defaults to the CPU's core count - 24 on the box this was measured
on). A single-request forward pass through a small conv-based model spread
across ALL cores incurs real thread-synchronization overhead, and gets
catastrophically worse under any real concurrent CPU load (other requests,
data-loading workers, etc.) since every thread has to wait on whichever one
gets pre-empted.

Benchmarked on a 24-core box, 5s of audio, n=15-30 calls per setting, WITH
real concurrent CPU load from another process (a training job's data
loader - i.e. the realistic "not actually idle" case this matters for):

    threads= 1: 37.6-51.0ms
    threads= 2: 21.8-25.5ms
    threads= 4: 13.3-23.7ms
    threads= 8:  9.1-16.9ms   <- fastest AND tightest range
    threads=24 (the unbounded default): 484.8-600.4ms

24 threads was 30-50x slower than 8, and its range matches the "up to
~600ms" ceiling almost exactly - not audio-duration-driven (a 1s clip hit
the highest max latency of any duration tested in the same investigation,
which rules out duration as the dominant factor).

`torch.set_num_threads()` is process-global, not thread-local - changing
it permanently would affect every other CPU op in the process, including
ones that might genuinely want more threads. Scoping it to just the
watermark call (this module's whole purpose) avoids that."""

from __future__ import annotations

from contextlib import contextmanager

import torch

# 8 was the fastest AND most stable setting measured (see module
# docstring) - not "as many cores as available", which is what caused the
# original pathology. Revisit if re-measured on different hardware; this
# is a real number from one box's benchmark, not a theoretically-derived
# constant (same "measure it, don't guess" convention as this project's
# other latency-tuned constants).
WATERMARK_NUM_THREADS = 8


@contextmanager
def scoped_watermark_threads(num_threads: int = WATERMARK_NUM_THREADS):
    """Temporarily sets `torch.set_num_threads(num_threads)` for the
    duration of the `with` block, restoring whatever the process-wide
    value was beforehand on exit (including on an exception - `finally`
    guarantees restoration isn't skipped by a raised error inside the
    watermark call). Wrap `watermarker.apply_watermark(...)` calls in this,
    not the whole inference pipeline - T3/S3Gen's own CPU-side glue code
    may have different, legitimate reasons to want more threads."""
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
