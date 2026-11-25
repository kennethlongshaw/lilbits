import random

from lilbits.op_tasks import TASKS, chain_match, make, score, score_path


def test_every_task_scores_its_reference_and_fits():
    rng = random.Random(0)
    for t in TASKS:
        for _ in range(20):
            e = make(t, rng)
            assert score(t, e.answer, e) == 1.0 and len(e.prompt) + len(e.answer) < 200


def test_path_scoring_and_chain_match():
    e = make("path", random.Random(3))
    _, a, b, _ = e.meta
    assert score_path(e.answer, e) == 1.0 and score_path(f"{a} {b}", e) == 0.0
    assert chain_match("count", ["attend", "filter", "count", "noop"]) == 1.0
    assert chain_match("count", ["count", "filter"]) == 0.0
    assert chain_match("argmax", ["max", "point"]) == 1.0


def test_easy_level_uses_one_letter_names_and_scores():
    import random

    from lilbits.op_tasks import TASKS, make, score

    rng = random.Random(0)
    for task in TASKS:
        e = make(task, rng, level=0)
        assert score(task, e.answer, e) == 1.0
        assert len(e.prompt) < 60


def test_dense_stream_fits_and_trains_only_predictable_bytes():
    import random

    from lilbits.op_tasks import dense_stream
    from lilbits.shakespeare import segment_tensors

    segs = dense_stream(random.Random(0), 201)
    assert 0 < sum(len(b) for b, _ in segs) <= 201
    _, mask = segment_tensors([segs], 200)
    assert mask.sum() >= 15
    text = b"".join(b for b, _ in segs)
    for b, trained in segs:  # every trained recall value appears earlier in its episode
        if trained and b.isdigit() and len(b) == 2:
            assert text.count(b) >= 2
