"""Looped index LM: the full lean-decoder design, trained with its staged curriculum (smoke run).

    tokens -> embed (tied) -> PRELUDE (LittleBit GDN, writes the INDEX) -> LOOPED CORE x N (shared LittleBit
    signs, per-loop FP scales, GDN state per loop, loop memory, retrieval: block summaries -> sign signatures ->
    exact top-k, dense fallback) -> CODA (FP GDN, LM head, MTP draft heads) -> VERIFY next pass

See src/lilbits/models/looped_index.py for the design. This script trains it in the planned stages, next to a
dense FP reference (same architecture, FP everywhere, dense retrieval), on long-context recall and two-hop
questions:

    stage 1  looped GDN core, FP, fixed N, loop memory
    stage 2  + LittleBit core (Dual-SVID from the stage-1 weights), per-loop scales
    stage 3  + sparse index and deep summaries; the block scorer is trained to match dense attention
    stage 4  + MTP draft heads
    stage 5  + halting (decode only: no training, calibrates the early-exit threshold)

Then it measures decode: plain greedy (fixed N, and with halting), and MTP + parallel verify, checking that
speculative output is token-for-token the same as greedy. Every stage and decode mode reports wall time, CPU time
and peak RSS; the model reports params, checkpoint and deployed size, and the cache reports bytes per memory tier.

Default is a smoke run (a few steps per stage) that proves the code paths work. Recall and two-hop accuracy at
this size and step count are at chance; they are there so the same script can measure a real run.

    uv run python examples/09_looped_index_lm.py
    uv run python examples/09_looped_index_lm.py --steps 300 --length 512   # an actual (still small) run
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from lilbits.data import RecallSource
from lilbits.measure import Meter, size_report, state_dict_bytes, tensor_bytes
from lilbits.models.looped_index import LoopedIndexConfig, LoopedIndexLM, convert_core_to_littlebit, memory_tiers


@torch.no_grad()
def evaluate(model: LoopedIndexLM, x: torch.Tensor, kind: torch.Tensor) -> dict:
    model.eval()
    h = model.process(x, model.new_cache(x.shape[0]))
    logits = model.logits(h)
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item()
    hit = logits[:, -2].argmax(-1).eq(x[:, -1]).float()
    model.train()
    return {
        "loss": loss,
        "recall_acc": hit[kind == 0].mean().item() if (kind == 0).any() else float("nan"),
        "two_hop_acc": hit[kind == 1].mean().item() if (kind == 1).any() else float("nan"),
    }


def train_stage(model, src, steps, batch, lr, mtp) -> tuple[Meter, dict]:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.95))
    info: dict = {}
    model.train()
    with Meter() as m:
        for _ in range(steps):
            x, _ = src.sample(batch)
            loss, info = model.training_loss(x, mtp=mtp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return m, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=6, help="training steps per stage (default: smoke run)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--length", type=int, default=128, help="sequence length (blocks of 64 tokens)")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--gen", type=int, default=24, help="tokens to decode in the decode tests")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)

    src = RecallSource(length=args.length, seed=0)
    ev_x, ev_kind = RecallSource(length=args.length, seed=123).sample(32)
    cfg = LoopedIndexConfig(vocab_size=src.vocab, d_model=args.d, n_loops=args.loops)
    model = LoopedIndexLM(cfg)
    ref_cfg = copy.deepcopy(cfg)
    ref_cfg.prelude_linear, ref_cfg.flat = "fp", -1.0  # FP everywhere; flat < 0: retrieval is always dense
    ref = LoopedIndexLM(ref_cfg)
    print(
        f"recall / two-hop data: length {args.length}, {src.n_pairs} facts per sequence, vocab {src.vocab}; "
        f"d={args.d}, N={args.loops} loops, {args.steps} steps per stage\n"
    )

    stages = [
        ("1 looped GDN core, FP, fixed N, loop memory", dict(index=False, mtp=False, convert=False)),
        ("2 + LittleBit core, per-loop scales", dict(index=False, mtp=False, convert=True)),
        ("3 + sparse index, deep summaries, index distillation", dict(index=True, mtp=False, convert=False)),
        ("4 + MTP draft heads", dict(index=True, mtp=True, convert=False)),
    ]
    print(
        "| stage | model | train wall / CPU | peak RSS growth | last train loss | eval loss | recall acc | 2-hop acc |"
    )
    print("|---|---|---|---|---|---|---|---|")
    metrics: dict = {"stages": {}}
    for name, s in stages:
        if s["convert"]:
            convert_core_to_littlebit(model)
        for tag, m in (("lean", model), ("FP dense ref", ref)):
            m.use_index = m.distill = s["index"]
            tm, info = train_stage(m, src, args.steps, args.batch, 2e-3, s["mtp"])
            ev = evaluate(m, ev_x, ev_kind)
            print(
                f"| {name} | {tag} | {tm.wall_s:.1f} s / {tm.cpu_s:.1f} s | {tm.peak_extra_mib:.0f} MiB "
                f"| {info['lm']:.3f} | {ev['loss']:.3f} | {ev['recall_acc']:.2f} | {ev['two_hop_acc']:.2f} |"
            )
            metrics["stages"][f"{name} [{tag}]"] = {**tm.as_dict("train_"), **ev, **info}

    # ---- decode ------------------------------------------------------------------------------------------
    model.eval()
    prompt = ev_x[:1, : args.length - 1]
    print(f"\ndecode: prompt {prompt.shape[1]} tokens, {args.gen} new tokens, batch 1\n")
    print("| mode | wall | CPU | peak RSS growth | passes | tokens / pass | mean loops | same tokens as greedy |")
    print("|---|---|---|---|---|---|---|---|")
    with Meter() as m0:
        greedy, st0 = model.generate(prompt, args.gen)
    rows = [("greedy, fixed N", m0, st0, True)]
    for eps in (0.05, 0.2):  # stage 5: halting thresholds, decode only
        with Meter() as m1:
            out, st1 = model.generate(prompt, args.gen, halt_eps=eps)
        rows.append((f"greedy, halting (eps {eps})", m1, st1, torch.equal(out, greedy)))
    with Meter() as m2:
        spec, st2 = model.generate_speculative(prompt, args.gen)
    rows.append(("MTP drafts + parallel verify", m2, st2, torch.equal(spec, greedy)))
    metrics["decode"] = {}
    for name, m, st, same in rows:
        tpp = st.get("tokens_per_pass", 1.0)
        loops = st.get("mean_loops", cfg.n_loops)
        print(
            f"| {name} | {m.wall_s * 1e3:.0f} ms | {m.cpu_s * 1e3:.0f} ms | {m.peak_extra_mib:.1f} MiB "
            f"| {st['passes']} | {tpp:.2f} | {loops:.2f} | {'yes' if same else 'no'} |"
        )
        metrics["decode"][name] = {
            **m.as_dict("decode_"),
            "passes": st["passes"],
            "tokens_per_pass": tpp,
            "mean_loops": loops,
            "same_as_greedy": same,
            **({"acceptance": st["acceptance"], "k_trace": st["k_trace"]} if "acceptance" in st else {}),
        }
    print(f"\nspeculative: draft acceptance {st2['acceptance']:.2f}, k over passes {st2['k_trace']}")
    distinct = len(set(greedy[0, prompt.shape[1] :].tolist()))
    print(
        f"(greedy output has {distinct} distinct tokens: a smoke-trained model mostly repeats filler, which makes"
        " drafts easy to accept, so this acceptance rate says nothing about a trained model)"
    )

    tiers = memory_tiers(model, st2["cache"], prompt.shape[1] + args.gen)
    print("\nmemory tiers after decode:")
    print(f"  within token  loop memory        {tiers['within_token_loop_memory_bytes_per_position']:,} B / position")
    print(f"  across tokens GDN states         {tiers['gdn_state_bytes']:,} B (fixed size)")
    print(f"  across blocks summaries          {tiers['summary_bytes']:,} B")
    kv, kvt = tiers["kv_signature_bytes"], tiers["kv_signature_bytes_per_token"]
    print(f"  raw recall    K/V + signatures   {kv:,} B ({kvt:.0f} B / token)")

    sizes = {}
    for tag, m in (("lean", model), ("FP dense ref", ref)):
        rep = size_report(m)
        sizes[tag] = {
            "params": rep["trainable_params"],
            "checkpoint_bytes": state_dict_bytes(m),
            "stored_bytes": rep["deployed_total_bytes"],
            "block_bpw": rep["block_bpw"],
        }
    print("\n| model | params | checkpoint (training format) | deployed | LittleBit / FP16 block bpw |")
    print("|---|---|---|---|---|")
    for tag, z in sizes.items():
        print(
            f"| {tag} | {z['params']:,} | {z['checkpoint_bytes'] / 1024:,.0f} KiB | "
            f"{z['stored_bytes'] / 1024:,.0f} KiB | {z['block_bpw']:.2f} |"
        )
    print("\nsmoke-run accuracies are at chance (1 / 24); the run proves every stage and decode path executes.")
    metrics.update(
        train_steps=4 * args.steps,
        train_wall_s=sum(v["train_wall_s"] for k, v in metrics["stages"].items() if "[lean]" in k),
        train_cpu_s=sum(v["train_cpu_s"] for k, v in metrics["stages"].items() if "[lean]" in k),
        train_peak_extra_mib=max(v["train_peak_extra_mib"] for v in metrics["stages"].values()),
        memory_tiers=tiers,
        sizes=sizes,
        params=sizes["lean"]["params"],
        ram_weights_bytes=tensor_bytes([*model.parameters(), *model.buffers()]),
        checkpoint_bytes=sizes["lean"]["checkpoint_bytes"],
        stored_bytes=sizes["lean"]["stored_bytes"],
    )
    print("METRICS " + json.dumps(metrics))


if __name__ == "__main__":
    main()
