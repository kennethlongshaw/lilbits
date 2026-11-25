from lilbits.shakespeare import (
    ShakespeareBytes,
    exact_match,
    load_qa,
    make_qa,
    qa_tensors,
    split_speeches,
    text_of,
)


def test_splits_are_disjoint_and_cover_the_corpus():
    tr, te = split_speeches("train"), split_speeches("test")
    assert len(te) > 500 and len(tr) > 9 * len(te) - 10
    assert text_of("test") not in text_of("train")


def test_committed_qa_matches_generator_and_is_exact():
    assert load_qa("test") == make_qa("test", 400, seed=1)
    test_text = text_of("test")
    for p in load_qa("test")[:40]:
        assert p["a"] and (p["kind"] == "line_count" or p["a"] in test_text)


def test_qa_tensors_mask_covers_answer_only():
    pairs = load_qa("train")[:3]
    x, m = qa_tensors(pairs, 160)
    assert x.shape == (3, 161) and m.shape == (3, 160)
    for row, mask, p in zip(x, m, pairs, strict=True):
        ans = bytes(b for b, k in zip(row[1:].tolist(), mask.tolist(), strict=True) if k).decode()
        assert ans == p["a"] + "\n"


def test_exact_match_and_bytes():
    pairs = load_qa("test")[:8]
    gold = {p["q"]: p["a"] for p in pairs}
    assert exact_match(lambda q: gold[q], pairs)["exact_match"] == 1.0
    assert exact_match(lambda q: "", pairs)["exact_match"] == 0.0
    assert ShakespeareBytes("train").sample(2, 64).shape == (2, 65)
