"""Speculative-decoding style state handling on a recurrent rule: snapshot, try k draft tokens, roll back.

The gated delta rule (and every recurrence in `mixers/recurrent.py`) maps (state, tokens) -> (outputs, state), so
verifying k drafts is one call from a snapshot, and rejecting drafts j.. means recomputing from the snapshot
over the accepted prefix only. This shows that the accepted-prefix state equals the state a model that never
saw the rejected drafts would have.

    uv run python examples/03_state_rollback.py
"""

import torch
import torch.nn.functional as F

from lilbits.mixers.recurrent import gated_delta_rule

torch.manual_seed(0)
B, H, D = 1, 2, 8


def toks(n):
    q, k = torch.randn(B, n, H, D), F.normalize(torch.randn(B, n, H, D), dim=-1)
    return q, k, torch.randn(B, n, H, D), -torch.rand(B, n, H) * 0.1, torch.rand(B, n, H)


prompt = toks(20)
_, S = gated_delta_rule(*prompt)  # prefill
snapshot = S.clone()

k_drafts = 5
drafts = toks(k_drafts)
_, S_all = gated_delta_rule(*drafts, initial_state=snapshot)  # verify pass writes all k

accepted = 3  # say the verifier accepts the first 3 drafts
prefix = tuple(t[:, :accepted] for t in drafts)
_, S_rolled = gated_delta_rule(*prefix, initial_state=snapshot)  # roll back = replay accepted prefix

full = tuple(torch.cat([p, a], 1) for p, a in zip(prompt, prefix, strict=False))
_, S_ref = gated_delta_rule(*full)
print("state after rollback matches a clean run:", torch.allclose(S_rolled, S_ref, atol=1e-6))
print("state with rejected drafts differs:      ", not torch.allclose(S_all, S_ref, atol=1e-6))
print(f"state size: {S.numel()} floats, independent of sequence length")
