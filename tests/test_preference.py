import random

import torch

from lilbits.shakespeare import copy_sort_example, load_qa, needle_example, run_tool, segment_tensors, tool_example
from lilbits.training.preference import dpo_loss, group_advantages, grpo_loss, sample, seq_logprob


def test_group_advantages_zero_mean_per_group():
    a = group_advantages(torch.tensor([1.0, 0, 0, 0, 1, 1, 0, 0]), 4)
    torch.testing.assert_close(a.view(2, 4).mean(1), torch.zeros(2), atol=1e-6, rtol=0)
    assert a[0] > 0 > a[1]


def test_grpo_pushes_up_rewarded_samples():
    logp = torch.zeros(4, requires_grad=True)
    loss, _ = grpo_loss(logp, torch.tensor([1.0, 0, 0, 0]), 4)
    loss.backward()
    assert logp.grad is not None and logp.grad[0] < 0 < logp.grad[1]


def test_dpo_prefers_chosen_and_logprob_masks():
    pc = torch.zeros(3, requires_grad=True)
    loss, info = dpo_loss(pc, torch.zeros(3), torch.zeros(3), torch.zeros(3))
    loss.backward()
    assert pc.grad is not None and (pc.grad < 0).all() and info["pref_acc"] == 0
    x = torch.tensor([[1, 2, 3, 4]])
    lp = seq_logprob(lambda t: torch.zeros(*t.shape, 8), x, torch.tensor([[0.0, 1, 1]]))
    torch.testing.assert_close(lp, torch.tensor([2 * -torch.log(torch.tensor(8.0)).item()]))


def test_sampling_stops_and_tasks_are_well_formed():
    def step(tok, state):
        return torch.full((*tok.shape, 256), -1e9).index_fill(-1, torch.tensor([10]), 0.0), state

    out = sample(step, torch.zeros(2, 3, dtype=torch.long), 5)
    assert (out == 10).all()
    rng = random.Random(0)
    p, a = needle_example(400, rng)
    assert len(p) == 400 and a.encode() in p
    p, a = copy_sort_example(rng, 3, "sort")
    assert a == ", ".join(sorted(a.split(", ")))
    t = next(t for t in (tool_example(q) for q in load_qa("test")) if t)
    call = t[1][0].decode().strip()
    assert run_tool(call) != "ERROR" and run_tool("bogus()") == "ERROR"
    _, m = segment_tensors([t], 200)
    assert m.sum() == len(t[1][0]) + len(t[4][0])
