"""A ~10 MB English byte corpus for models that tiny-shakespeare (1.1 MB) starves.

Sources, downloaded once and cached (pinned by SHA-256):
    tiny-shakespeare   data/shakespeare/tiny_shakespeare.txt (in the repo), 1.1 MB
    NLTK gutenberg     18 public-domain books (Austen, Melville, Milton, Chesterton, Carroll, Whitman, ...), the
                       KJV bible capped at 1 MB so it doesn't dominate, ~8.4 MB
    NLTK webtext       overheard conversations, Monty Python's Grail, Pirates of the Caribbean, wine reviews, ~1.1 MB
                       (the Firefox bug tracker and singles ads are left out: short noisy records)

Split: the last TEST_FRAC of every document is held out, so the test set covers every author and style and never
shares text with the train set.

    CorpusBytes(split, source=...) random byte windows (vocab 256) from the train or test text, like ShakespeareBytes
    documents()               {name: text}
    split_text(split, source) one split as a single string (documents separated by a blank line)

Larger corpora (``source=`` in CorpusBytes / split_text), ~100 MB train + ~5 MB test each, cut at document boundaries:
    fineweb_edu  FineWeb-Edu sample-10BT (educational web pages, ODC-By 1.0, Common Crawl terms). Fetched from a public
                 GitHub mirror that stores it as GPT-2 BPE uint16 shards; GPT-2 BPE is byte-level, so decoding gives the
                 original text exactly. Train = start of train_0000, test = start of the mirror's held-out val_000.
    wiki40b      Wiki40B en 1.3.0 (cleaned Wikipedia articles, CC BY-SA 3.0) from Google's public TFDS bucket.
                 Train = train shards 0-1, test = start of the official validation split.
The first use downloads ~190 MB (fineweb_edu) or ~210 MB (wiki40b), checks the SHA-256 pins and caches the decoded
text as .txt next to the downloads.

Cache dir: $LILBITS_CACHE/corpus (default ~/.cache/lilbits/corpus).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import urllib.request
import zipfile
from functools import cache
from pathlib import Path

import numpy as np
import torch

from lilbits.shakespeare import corpus as shakespeare_text

TEST_FRAC = 0.05
CAP = 1_000_000  # bytes kept per document (only the bible is longer)
_NLTK = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/"
ZIPS = {
    "gutenberg.zip": "2d3c3ab548c653944310f37f536443ec85d0a0ad855fcae217a0c9efdce2d611",
    "webtext.zip": "9e32dbae4879464b8f420a0dc721855bb26167b720d7695588d2ca2aeadf501a",
}
SKIP = {"README", "firefox.txt", "singles.txt"}


def cache_dir() -> Path:
    return Path(os.environ.get("LILBITS_CACHE", Path.home() / ".cache" / "lilbits")) / "corpus"


def _fetch(name: str) -> bytes:  # NLTK zips
    path = cache_dir() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_NLTK + name, timeout=120) as r:
            blob = r.read()
        if hashlib.sha256(blob).hexdigest() != ZIPS[name]:
            raise RuntimeError(f"{name}: SHA-256 mismatch (upstream changed?)")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(blob)
        tmp.replace(path)
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != ZIPS[name]:
        raise RuntimeError(f"{path}: SHA-256 mismatch; delete it to re-download")
    return blob


@cache
def documents() -> dict[str, str]:
    docs = {"tiny_shakespeare": shakespeare_text()}
    for z in ZIPS:
        with zipfile.ZipFile(io.BytesIO(_fetch(z))) as f:
            for info in sorted(f.infolist(), key=lambda i: i.filename):
                base = info.filename.rsplit("/", 1)[-1]
                if info.is_dir() or base in SKIP:
                    continue
                text = f.read(info).decode("latin-1").replace("\r\n", "\n")
                docs[f"{z[:-4]}/{base[:-4]}"] = text[:CAP]
    return docs


@cache
def _mixed_text(split: str) -> str:
    parts = []
    for text in documents().values():
        cut = int(len(text) * (1 - TEST_FRAC))
        parts.append(text[:cut] if split == "train" else text[cut:])
    return "\n\n".join(parts)


# --- larger corpora -------------------------------------------------------------------------------------------------

WEB_TRAIN_BYTES = 100_000_000
WEB_TEST_BYTES = 5_000_000
DOC_SEP = "\n\n"

_FW = "https://raw.githubusercontent.com/LBrunswic/a100lab-data/401956a8d32f08351c566bd15fc4e783da941a55/"
FINEWEB_FILES = {  # cache name: (url, sha256)
    "fineweb_edu_train_0000.bin": (_FW + "fineweb-edu/train_0000.bin",
                                   "3abad8292bc2b8060ed9c1875d29b1d736636b186953769cfb93bd61dd95ec7f"),
    "fineweb_edu_val_000.bin": (_FW + "fineweb-edu/val_000.bin",
                                "3e7b7a6397c10343e79ac2ea15c17f2b43e4244a5f004c998ffa97cb5f75e8c1"),
    "gpt2_encoder.json": (_FW + "tiktoken-cache/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",  # OpenAI's encoder.json
                          "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"),
}  # fmt: skip
_W40 = "https://storage.googleapis.com/tfds-data/datasets/wiki40b/en/1.3.0/"
WIKI40B_FILES = {
    "wiki40b_train_00000.tfrecord": (_W40 + "wiki40b-train.tfrecord-00000-of-00128",
                                     "db95b9311f886bad0c1d588027662956a1e0462b6009b214b4a1f8a3c09083c6"),
    "wiki40b_train_00001.tfrecord": (_W40 + "wiki40b-train.tfrecord-00001-of-00128",
                                     "7afffd929f55da9411d9eba1256b17e46ea502cc228a3c9fa69a53dbd5f12ee8"),
    "wiki40b_validation_00000.tfrecord": (_W40 + "wiki40b-validation.tfrecord-00000-of-00008",
                                          "802bab1790866003aca62aa044e3dc213f766d9fdd0a52573d65aab63bef6ba9"),
}  # fmt: skip


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _download(name: str, url: str, sha: str) -> Path:
    """Stream url to the cache (pinned by SHA-256) and return its path."""
    path = cache_dir() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        if _sha256(tmp) != sha:
            tmp.unlink()
            raise RuntimeError(f"{name}: SHA-256 mismatch (upstream changed?)")
        tmp.replace(path)
    elif _sha256(path) != sha:
        raise RuntimeError(f"{path}: SHA-256 mismatch; delete it to re-download")
    return path


def _take(docs, limit: int) -> str:
    """Join whole documents with DOC_SEP until about limit UTF-8 bytes (the last document may be cut)."""
    out, n = [], 0
    for d in docs:
        d = d.strip()
        if not d:
            continue
        b = len(d.encode()) + len(DOC_SEP)
        if n + b > limit:
            if n < limit // 2:  # one huge document: keep a cut of it rather than nothing
                out.append(d.encode()[: limit - n].decode(errors="ignore"))
            break
        out.append(d)
        n += b
    return DOC_SEP.join(out)


def _bytes_to_unicode() -> dict[str, int]:
    """GPT-2's printable stand-in character for every byte, inverted (char -> byte)."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


def gpt2_documents(tokens: np.ndarray, encoder: dict[str, int], eot: int = 50256, chunk: int = 1 << 20):
    """Yield the documents of a flat GPT-2 token stream (documents separated by the end-of-text token)."""
    inv = _bytes_to_unicode()
    table = [b""] * (max(*encoder.values(), eot) + 1)
    for tok, i in encoder.items():
        table[i] = bytes(inv[c] for c in tok)
    table[eot] = b"\x00"  # NUL never occurs in the text; marks document ends
    rest = b""
    for s in range(0, len(tokens), chunk):
        blob = rest + b"".join(table[t] for t in tokens[s : s + chunk].tolist())
        *docs, rest = blob.split(b"\x00")
        for d in docs:
            yield d.decode("utf-8", errors="replace")
    if rest:
        yield rest.decode("utf-8", errors="replace")


def tfrecord_examples(path: Path):
    """Yield each tf.train.Example in a TFRecord file as {feature name: first bytes value} (no TensorFlow needed)."""

    def varint(b, i):
        r = s = 0
        while True:
            c = b[i]
            i += 1
            r |= (c & 0x7F) << s
            s += 7
            if c < 0x80:
                return r, i

    def fields(b):  # (field number, payload) of the length-delimited fields of one protobuf message
        i = 0
        while i < len(b):
            key, i = varint(b, i)
            wire = key & 7
            if wire == 2:
                n, i = varint(b, i)
                yield key >> 3, b[i : i + n]
                i += n
            elif wire == 0:
                _, i = varint(b, i)
            elif wire == 5:
                i += 4
            elif wire == 1:
                i += 8
            else:
                raise ValueError(f"{path}: unsupported protobuf wire type {wire}")

    with open(path, "rb") as f:
        while len(head := f.read(12)) == 12:
            rec = f.read(struct.unpack("<Q", head[:8])[0])
            f.read(4)  # data CRC
            ex = {}
            for _, features in fields(rec):  # Example.features
                for _, entry in fields(features):  # Features.feature map entries
                    key, val = None, b""
                    for num, v in fields(entry):
                        if num == 1:
                            key = v.decode()
                        else:  # Feature -> BytesList -> first value
                            for _, blist in fields(v):
                                for _, x in fields(blist):
                                    val = x
                                    break
                    ex[key] = val
            yield ex


def wiki40b_clean(raw: str) -> str:
    """Wiki40B markup to plain text: the title, then each section heading and paragraph on its own line."""
    lines, mark = [], None
    for line in raw.split("\n"):
        if line in ("_START_ARTICLE_", "_START_SECTION_", "_START_PARAGRAPH_"):
            mark = line
        elif line:
            if mark == "_START_SECTION_" and lines and lines[-1]:
                lines.append("")
            lines.append(line.replace("_NEWLINE_", "\n"))
            if mark == "_START_ARTICLE_":
                lines.append("")
    return "\n".join(lines).strip()


def _cached_text(name: str, build) -> str:
    path = cache_dir() / name
    if not path.exists():
        text = build()
        tmp = path.with_suffix(".part")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return path.read_text(encoding="utf-8")


def _fineweb_text(split: str) -> str:
    def build():
        paths = {k: _download(k, *v) for k, v in FINEWEB_FILES.items()}
        enc = json.loads(paths["gpt2_encoder.json"].read_text(encoding="utf-8"))
        shard = paths["fineweb_edu_train_0000.bin" if split == "train" else "fineweb_edu_val_000.bin"]
        toks = np.memmap(shard, dtype="<u2", mode="r")
        return _take(gpt2_documents(toks, enc), WEB_TRAIN_BYTES if split == "train" else WEB_TEST_BYTES)

    return _cached_text(f"fineweb_edu_{split}.txt", build)


def _wiki40b_text(split: str) -> str:
    def build():
        names = [k for k in WIKI40B_FILES if ("train" in k) == (split == "train")]
        paths = [_download(k, *WIKI40B_FILES[k]) for k in names]
        docs = (wiki40b_clean(ex["text"].decode()) for p in paths for ex in tfrecord_examples(p))
        return _take(docs, WEB_TRAIN_BYTES if split == "train" else WEB_TEST_BYTES)

    return _cached_text(f"wiki40b_{split}.txt", build)


SOURCES = {"mixed": _mixed_text, "fineweb_edu": _fineweb_text, "wiki40b": _wiki40b_text}


@cache
def split_text(split: str, source: str = "mixed") -> str:
    """One split as a single string (documents separated by a blank line)."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be train or test, not {split!r}")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {sorted(SOURCES)}, not {source!r}")
    return SOURCES[source](split)


class CorpusBytes:
    """Random byte windows (vocab 256) from one split. Non-ASCII characters are stored as their UTF-8 bytes.

    source: "mixed" (~10 MB, the default), "fineweb_edu" or "wiki40b" (~100 MB train / ~5 MB test each)."""

    vocab = 256

    def __init__(self, split: str = "train", seed: int = 0, source: str = "mixed"):
        self.data = torch.frombuffer(bytearray(split_text(split, source).encode()), dtype=torch.uint8)  # 1 byte/byte
        self._g = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch: int, seq_len: int) -> torch.Tensor:
        ix = torch.randint(0, len(self.data) - seq_len - 1, (batch,), generator=self._g)
        return torch.stack([self.data[i : i + seq_len + 1] for i in ix]).long()
