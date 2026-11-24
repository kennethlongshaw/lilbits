"""The recurrent token mixers side by side: parameters, state size per sequence, and state carry.

Each mixer takes `initial_state` and returns its final recurrent state, which has a fixed size, unlike a
softmax K/V cache that grows with length. (Mixers with a short conv also need its last conv_size-1 inputs
to continue a sequence; see 03_state_rollback.py for carrying state through a recurrence.)

    uv run python examples/02_recurrent_mixers.py
"""

import torch

from lilbits.mixers.recurrent import MIXERS
from lilbits.models.tinylm import Config, make_linear_factory


def state_numel(s):
    return sum(t.numel() for t in s) if isinstance(s, tuple) else s.numel()


torch.manual_seed(0)
d, heads, T = 128, 4, 48
x = torch.randn(1, T, d)
print(f"{'mixer':8s} {'params':>8s} {'state floats':>13s}  linears")
for name, cls in MIXERS.items():
    for linear in ("fp", "littlebit"):
        m = cls(d, heads, make_linear_factory(Config(linear=linear, bpw=1.0))).eval()
        with torch.no_grad():
            _, state = m(x, return_state=True)  # final recurrent state: fixed size, whatever T is
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{name:8s} {n_params:8,d} {state_numel(state):13,d}  ({linear})")
print(f"\nsoftmax attention would instead cache 2 x {d} floats per token per layer: {2 * d * T:,} at T={T}, growing.")
