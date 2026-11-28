"""One-file model bundle (.lbm) for the standalone Rust CLI (`rust/lilbits_fused/src/bin/lilbits.rs`): no numpy or
Python at inference.

Layout: a text header, then the raw arrays, each starting at a 64-byte aligned offset.

    LBM1
    <n arrays>
    <name> <dtype> <offset> <nbytes>        (one line per array; dtype is the numpy dtype string, e.g. int8, float32)
    ...
    <blank line> then zero padding up to the first offset

The arrays are exactly what `fused.export` (plain LeanLM) or `fused_hnet.export` (SpaceHNet) return.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ALIGN = 64


def write(path: str | Path, arrays: dict[str, np.ndarray]) -> int:
    """Write the bundle; returns its size in bytes."""
    items = [(k, np.ascontiguousarray(v)) for k, v in sorted(arrays.items())]
    for k, _ in items:
        if " " in k or "\n" in k:
            raise ValueError(f"array name {k!r} has whitespace")

    def header(offsets: list[int]) -> bytes:
        lines = ["LBM1", str(len(items))]
        lines += [f"{k} {v.dtype} {o} {v.nbytes}" for (k, v), o in zip(items, offsets, strict=True)]
        return ("\n".join(lines) + "\n\n").encode()

    offsets = [0] * len(items)
    for _ in range(3):  # offsets depend on the header length, which depends on the offsets' digits
        pos = -(-len(header(offsets)) // ALIGN) * ALIGN
        new = []
        for _, v in items:
            new.append(pos)
            pos = -(-(pos + v.nbytes) // ALIGN) * ALIGN
        if new == offsets:
            break
        offsets = new
    h = header(offsets)
    assert len(h) <= offsets[0] if items else True
    with open(path, "wb") as f:
        f.write(h)
        for (_, v), o in zip(items, offsets, strict=True):
            f.write(b"\0" * (o - f.tell()))
            f.write(v.tobytes())
        return f.tell()


def read(path: str | Path) -> dict[str, np.ndarray]:
    """Inverse of `write` (for tests)."""
    blob = Path(path).read_bytes()
    lines = blob[: blob.index(b"\n\n")].decode().split("\n")
    assert lines[0] == "LBM1", "not an .lbm bundle"
    out = {}
    for line in lines[2 : 2 + int(lines[1])]:
        k, dt, o, n = line.split()
        out[k] = np.frombuffer(blob, dtype=np.dtype(dt), count=int(n) // np.dtype(dt).itemsize, offset=int(o))
    return out
