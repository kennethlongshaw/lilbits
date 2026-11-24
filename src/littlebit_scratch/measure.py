"""Measurement primitives shared by training, inference and the benchmarks.

Every comparison in this repo should report the same dimensions:
    quality, stored size (disk), memory (RAM weights, peak RSS growth, train state),
    CPU time and wall time, for both training and inference.
`Meter` captures wall + CPU + peak RSS for a block of code; `size_report` / `state_dict_bytes` cover size.
"""

from __future__ import annotations

import gc
import io
import os
import threading
import time
from dataclasses import dataclass, field

import torch
from torch import nn

from .bitnet import BitLinear
from .littlebit import LittleBitLinear


def _rss_mib() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def trim():
    """Free cached allocator memory so RSS-based peaks start from a clean baseline (glibc only)."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


@dataclass
class Meter:
    """with Meter() as m: ...  ->  m.wall_s, m.cpu_s (this process, all threads), m.cpu_util, m.peak_extra_mib.

    Peak RSS is sampled from a background thread (Linux /proc). Resetting the kernel's peak counter via
    /proc/self/clear_refs is avoided: in this environment it coincided with torch processes dying.
    """

    interval: float = 0.001
    wall_s: float = 0.0
    cpu_s: float = 0.0
    peak_extra_mib: float = 0.0
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cpu_util(self) -> float:
        """CPU seconds per wall second (1.0 = one core busy)."""
        return self.cpu_s / self.wall_s if self.wall_s else 0.0

    def __enter__(self):
        trim()
        self._base = self._peak = _rss_mib()

        def loop():
            while not self._stop.is_set():
                self._peak = max(self._peak, _rss_mib())
                time.sleep(self.interval)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        self._w0, self._c0 = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall_s = time.perf_counter() - self._w0
        self.cpu_s = time.process_time() - self._c0
        self._stop.set()
        self._t.join()
        self.peak_extra_mib = max(self._peak, _rss_mib()) - self._base

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}wall_s": self.wall_s, f"{prefix}cpu_s": self.cpu_s, f"{prefix}cpu_util": self.cpu_util,
                f"{prefix}peak_extra_mib": self.peak_extra_mib}


def tensor_bytes(tensors) -> int:
    seen, total = set(), 0
    for t in tensors:
        if id(t) not in seen:
            seen.add(id(t))
            total += t.numel() * t.element_size()
    return total


def state_dict_bytes(model: nn.Module) -> int:
    """Bytes of torch.save(model.state_dict()), i.e. the on-disk checkpoint size in this format."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def size_report(model: nn.Module) -> dict:
    """Training-time tensors (FP32) and deployed size: FP16 for FP parts, packed bits for LittleBit / BitNet."""
    q_params, q_bits, q_dense = set(), 0, 0
    fp_lin_params, fp_lin_dense = set(), 0
    for name, m in model.named_modules():
        if isinstance(m, (LittleBitLinear, BitLinear)):
            q_params |= {id(p) for p in m.parameters()}
            q_bits += m.storage_bits()
            q_dense += m.in_features * m.out_features
        elif isinstance(m, nn.Linear) and name.startswith("blocks."):
            fp_lin_params.add(id(m.weight))
            fp_lin_dense += m.weight.numel()
    block_ids = q_params | fp_lin_params
    other = [p for p in model.parameters() if id(p) not in block_ids]
    block_bits = q_bits + 16 * fp_lin_dense
    other_bits = 16 * sum(p.numel() for p in other)
    return {
        "trainable_params": sum(p.numel() for p in model.parameters()),
        "block_linear_weights": q_dense + fp_lin_dense,
        "train_param_bytes": tensor_bytes(model.parameters()),
        "deployed_block_bytes": block_bits / 8,
        "deployed_other_bytes": other_bits / 8,
        "deployed_total_bytes": (block_bits + other_bits) / 8,
        "block_bpw": block_bits / max(1, q_dense + fp_lin_dense),
    }
