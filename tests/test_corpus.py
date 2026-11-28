"""Offline checks of lilbits.corpus's decoders (no downloads)."""

import struct

import numpy as np

from lilbits import corpus


def _bpe_vocab():
    inv = {b: c for c, b in corpus._bytes_to_unicode().items()}
    words = ["Hello", " world", "\n", " café", "!"]
    return {"".join(inv[b] for b in w.encode()): i for i, w in enumerate(words)}


def test_gpt2_documents_round_trip_bytes_and_split_on_eot():
    enc = _bpe_vocab()
    eot = 9
    toks = np.array([0, 1, 4, eot, 3, 2, 0, eot, 1], dtype="<u2")
    docs = list(corpus.gpt2_documents(toks, enc, eot=eot, chunk=2))  # chunk smaller than a document
    assert docs == ["Hello world!", " café\nHello", " world"]


def _pb(field: int, payload: bytes) -> bytes:
    return bytes([(field << 3) | 2, len(payload)]) + payload


def _tfrecord(features: dict[str, bytes]) -> bytes:
    entries = b"".join(_pb(1, _pb(1, k.encode()) + _pb(2, _pb(1, _pb(1, v)))) for k, v in features.items())
    rec = _pb(1, entries)
    return struct.pack("<Q", len(rec)) + b"\0" * 4 + rec + b"\0" * 4


def test_tfrecord_examples_reads_bytes_features(tmp_path):
    p = tmp_path / "x.tfrecord"
    p.write_bytes(_tfrecord({"text": b"abc", "id": b"Q1"}) + _tfrecord({"text": b"def"}))
    assert list(corpus.tfrecord_examples(p)) == [{"text": b"abc", "id": b"Q1"}, {"text": b"def"}]


def test_wiki40b_clean():
    raw = (
        "\n_START_ARTICLE_\nKasturba Road\n_START_PARAGRAPH_\nA street._NEWLINE_Second line."
        "\n_START_SECTION_\nHistory\n_START_PARAGRAPH_\nOld."
    )
    assert corpus.wiki40b_clean(raw) == "Kasturba Road\n\nA street.\nSecond line.\n\nHistory\nOld."


def test_take_keeps_whole_documents_up_to_the_limit():
    assert corpus._take(["aaaa", " ", "bbbb", "cccc"], 13) == "aaaa\n\nbbbb"
    assert corpus._take(["x" * 100], 10) == "x" * 10  # one huge document is cut, not dropped
