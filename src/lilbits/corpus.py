"""A ~10 MB English byte corpus for models that tiny-shakespeare (1.1 MB) starves.

Sources, downloaded once and cached (pinned by SHA-256):
    tiny-shakespeare   data/shakespeare/tiny_shakespeare.txt (in the repo), 1.1 MB
    NLTK gutenberg     18 public-domain books (Austen, Melville, Milton, Chesterton, Carroll, Whitman, ...), the
                       KJV bible capped at 1 MB so it doesn't dominate, ~8.4 MB
    NLTK webtext       overheard conversations, Monty Python's Grail, Pirates of the Caribbean, wine reviews, ~1.1 MB
                       (the Firefox bug tracker and singles ads are left out: short noisy records)

Split: the last TEST_FRAC of every document is held out, so the test set covers every author and style and never
shares text with the train set.

    CorpusBytes(split)        random byte windows (vocab 256) from the train or test text, like ShakespeareBytes
    documents()               {name: text}
    split_text(split)         one split as a single string (documents separated by a blank line)

Cache dir: $LILBITS_CACHE/corpus (default ~/.cache/lilbits/corpus).
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from functools import cache
from pathlib import Path

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


def _fetch(name: str) -> bytes:
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
def split_text(split: str) -> str:
    if split not in ("train", "test"):
        raise ValueError(f"split must be train or test, not {split!r}")
    parts = []
    for text in documents().values():
        cut = int(len(text) * (1 - TEST_FRAC))
        parts.append(text[:cut] if split == "train" else text[cut:])
    return "\n\n".join(parts)


class CorpusBytes:
    """Random byte windows (vocab 256) from one split. Non-ASCII characters are stored as their UTF-8 bytes."""

    vocab = 256

    def __init__(self, split: str = "train", seed: int = 0):
        self.data = torch.frombuffer(bytearray(split_text(split).encode()), dtype=torch.uint8).long()
        self._g = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch: int, seq_len: int) -> torch.Tensor:
        ix = torch.randint(0, len(self.data) - seq_len - 1, (batch,), generator=self._g)
        return torch.stack([self.data[i : i + seq_len + 1] for i in ix])
