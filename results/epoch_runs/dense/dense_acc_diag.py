"""Teacher-forced accuracy on trained bytes: dense recall streams and easy lookup answers."""
import random, torch
from lilbits.models.op_loop import OpLoopLM
from lilbits.op_tasks import dense_stream, make
from lilbits.shakespeare import segment_tensors

sd = torch.load("/tmp/dense_init.pt", weights_only=True)
models = {"ops": OpLoopLM(96, 4), "attend": OpLoopLM(96, 4, ops=("attend",))}
rng = random.Random(777)
dense = [dense_stream(rng, 201) for _ in range(64)]
look = []
for _ in range(64):
    e = make("lookup", rng, level=0)
    look.append([(e.prompt, False), (str(e.answer).encode() + b"\n", True)])
for name, m in models.items():
    m.load_state_dict(sd[name]); m.eval(); m.tau, m.hard = 0.3, True
    for tag, rows in (("dense", dense), ("lookup0", look)):
        x, mk = segment_tensors(rows, 200)
        with torch.no_grad():
            logits = m(x[:, :-1])[0]
        pred = logits.argmax(-1); tgt = x[:, 1:]; msk = mk.bool()
        print(name, tag, "byte acc on trained bytes:", round(((pred == tgt) & msk).sum().item() / msk.sum().item(), 3))
