# CUDA graph capture/replay for CausalConditionalCFM's Euler ODE decode
# loop (solve_euler) - added for the HashStudioz Hinglish TTS project's
# latency sprint (see its project plan, "Workstream 1: fork chatterbox-tts,
# target S3Gen's CFM loop"), mirroring the pattern that project already
# established for T3's own decode step
# (engines/chatterbox_hinglish/engine/cuda_graph.py's `DecodeGraph`):
# static input buffers allocated once and overwritten in place via
# `.copy_()`, a warmup on a side stream before the real capture, and lazy
# capture keyed on the first REAL call for a given shape (not dummy data
# at import time).
#
# Precondition confirmed by a direct read of this fork's own s3gen/flow.py,
# flow_matching.py, decoder.py, hifigan.py (that project's plan documents
# the full discovery pass): `solve_euler` itself has NO host-sync-forcing
# calls (no `.item()`/`.cpu()`/`.any()` anywhere inside the loop or the
# `ConditionalDecoder` estimator it calls) - the one host sync in the
# whole S3Gen forward path (`flow.py`'s `(token >= vocab_size).any()`
# guard) runs one level up, in `CausalMaskedDiffWithXvec.inference`,
# BEFORE the encoder/decoder are ever called - entirely outside the region
# this module captures, so it does not need fixing for THIS graph to be
# valid. (It's still a real, if much smaller, per-call sync cost - tracked
# separately as a stray-sync audit item, not a blocker here.)
#
# batch_size is always 1 - S3Gen's own documented contract ("This function
# is designed for batch_size=1 only", `s3gen.py`'s `S3Token2Mel.forward`
# docstring). What DOES vary call-to-call is T (mel-timestep count,
# proportional to the caller's speech-token window length), so - unlike a
# single fixed-batch-size graph - this class captures ONE GRAPH PER BUCKET
# LENGTH, lazily, the first time a call actually needs that bucket.

from __future__ import annotations

import torch

_WARMUP_ITERS = 3


class CFMDecodeGraph:
    """Optional CUDA-graph accelerator for one loaded
    ``CausalConditionalCFM`` instance's ``solve_euler`` (non-meanflow)
    path. Not constructed by ``chatterbox-tts`` itself - a caller (e.g.
    the serving project's ``lifespan.py``) builds one after loading S3Gen
    and assigns it to ``cfm.cuda_graph``; ``CausalConditionalCFM.forward``
    uses it automatically when present and eligible, falling back to the
    ordinary eager ``solve_euler`` otherwise (see that method's own
    eligibility check - same shape/mode preconditions this class assumes:
    batch size 1, ``meanflow=False``, ``noised_mels is None``, and
    ``n_timesteps`` matching what this graph was built for).
    """

    def __init__(
        self, cfm, *, n_timesteps: int, bucket_lens: list[int], device: torch.device | str,
    ) -> None:
        if not bucket_lens:
            raise ValueError("CFMDecodeGraph requires at least one bucket length")
        self.cfm = cfm
        self.n_timesteps = n_timesteps
        self.bucket_lens = sorted(bucket_lens)
        self.device = device
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # Per-bucket static buffers - allocated lazily on first use of that
        # bucket, then reused (`.copy_()`/`.zero_()` in place) forever.
        self._buffers: dict[int, dict[str, torch.Tensor]] = {}

    def bucket_for(self, t_actual: int) -> int:
        for b in self.bucket_lens:
            if t_actual <= b:
                return b
        raise ValueError(
            f"window length {t_actual} exceeds the largest configured CFM graph bucket "
            f"{self.bucket_lens[-1]} - add a larger bucket."
        )

    def is_captured(self, bucket: int) -> bool:
        return bucket in self._graphs

    def _estimator_dtype(self, fallback: torch.dtype) -> torch.dtype:
        return getattr(self.cfm.estimator, "dtype", fallback)

    def _t_span(self, dtype: torch.dtype) -> torch.Tensor:
        t_span = torch.linspace(0, 1, self.n_timesteps + 1, device=self.device, dtype=dtype)
        if self.cfm.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return t_span

    def _forward_once(self, bucket: int) -> torch.Tensor:
        buf = self._buffers[bucket]
        return self.cfm.solve_euler(
            buf["x"], buf["t_span"], buf["mu"], buf["mask"], buf["spks"], buf["cond"],
            meanflow=False, graph_mode=True,
        )

    def _capture(self, bucket: int) -> torch.Tensor:
        """Captures using whatever REAL data the caller has already
        `.copy_()`'d into this bucket's static buffers (see ``run()`` -
        capture only ever happens on a real cache-miss call, never with
        dummy/zero data) - mirrors ``DecodeGraph.capture()``'s own
        discipline: the capture's final iteration IS the real forward
        pass, used directly, not a throwaway.
        """
        assert bucket not in self._graphs, f"CFMDecodeGraph bucket {bucket} captured twice"

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(_WARMUP_ITERS):
                self._forward_once(bucket)
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self._forward_once(bucket)
        self._graphs[bucket] = graph
        self._buffers[bucket]["out"] = out
        return out

    def run(
        self, t_actual: int, *, mu: torch.Tensor, mask: torch.Tensor, spks: torch.Tensor, cond: torch.Tensor,
    ) -> torch.Tensor:
        """Real caller-facing entrypoint: pads real (``T=t_actual``)
        inputs up to the chosen bucket length, replays (or captures, on
        first use of that bucket) the graph, and returns the UNPADDED
        slice of the result. Padded positions are masked (``mask=0``) so
        the decoder's attention/conv layers never treat them as real
        content - the same zero-pad-plus-mask contract ``flow.py``'s own
        variable-length batching already relies on elsewhere in this
        codebase, not a new convention.
        """
        bucket = self.bucket_for(t_actual)
        out_dtype = mu.dtype
        est_dtype = self._estimator_dtype(out_dtype)

        if bucket not in self._buffers:
            self._buffers[bucket] = dict(
                x=torch.zeros(1, 80, bucket, device=self.device, dtype=est_dtype),
                mu=torch.zeros(1, 80, bucket, device=self.device, dtype=est_dtype),
                mask=torch.zeros(1, 1, bucket, device=self.device, dtype=est_dtype),
                spks=torch.zeros(1, 80, device=self.device, dtype=est_dtype),
                cond=torch.zeros(1, 80, bucket, device=self.device, dtype=est_dtype),
                t_span=self._t_span(est_dtype),
            )
        buf = self._buffers[bucket]

        buf["x"].copy_(torch.randn(1, 80, bucket, device=self.device, dtype=est_dtype))
        buf["mu"].zero_()
        buf["mu"][:, :, :t_actual].copy_(mu.to(est_dtype))
        buf["mask"].zero_()
        buf["mask"][:, :, :t_actual].copy_(mask.to(est_dtype))
        buf["cond"].zero_()
        buf["cond"][:, :, :t_actual].copy_(cond.to(est_dtype))
        buf["spks"].copy_(spks.to(est_dtype))

        if not self.is_captured(bucket):
            self._capture(bucket)
        else:
            self._graphs[bucket].replay()
        # .clone() is load-bearing, not defensive style: buf["out"] is the
        # graph's OWN static output buffer, mutated in place by every
        # subsequent replay() for this bucket - handing back a view (which
        # `.to(dtype)` silently is, whenever the dtype already matches) lets
        # a caller's "result" mutate out from under them the next time this
        # bucket runs. Confirmed as a real, reproducible bug (not a
        # theoretical one) via a direct correctness test: two consecutive
        # calls with genuinely different inputs, same bucket, returned
        # bit-identical tensors (both views into the same storage) before
        # this fix.
        return buf["out"][:, :, :t_actual].clone().to(out_dtype)
