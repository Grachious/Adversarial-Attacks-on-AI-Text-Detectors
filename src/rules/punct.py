from __future__ import annotations
import random
import re
import string
from typing import Optional
from .text_utils import WORD_RE, tokenize_words_punct, detokenize


def remove_punctuation(text: str, seed: Optional[int] = None) -> str:
    basic_keep = {".", "!", "?"}
    drop = set(string.punctuation) - basic_keep
    out = [" " if ch in drop else ch for ch in text]
    return re.sub(r"\s+", " ", "".join(out)).strip()


def add_random_punct(
    text: str,
    p: float = 0.10,
    max_inserts: int = 3,
    seed: Optional[int] = None,
) -> str:
    rng = random.Random(seed)
    punct_pool = [",", ",", ",", ";", ":", "-"]

    toks = tokenize_words_punct(text)
    word_pos = [i for i, t in enumerate(toks) if WORD_RE.fullmatch(t)]
    if len(word_pos) < 2:
        return text

    gaps = []
    for a, b in zip(word_pos, word_pos[1:]):
        mid = toks[a + 1 : b]
        if any(x in {".", "!", "?", ";", ":", "—", "-"} for x in mid):
            continue
        gaps.append(b)

    if not gaps:
        return text

    for pos in rng.sample(gaps, k=min(len(gaps), max_inserts)):
        if rng.random() < p:
            toks.insert(pos, rng.choice(punct_pool))

    return detokenize(toks)
