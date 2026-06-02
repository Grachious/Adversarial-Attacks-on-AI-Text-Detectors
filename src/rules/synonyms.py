from __future__ import annotations
import random
from typing import Optional
from .text_utils import WORD_RE, tokenize_words_punct, detokenize


# ---------------------------------------------------------------------------
# Стоп-слова, которые нельзя заменять
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "so",
    "because", "as", "to", "of", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had",
    "this", "that", "these", "those", "it", "its",
    "he", "she", "they", "them", "we", "us", "you", "i",
    "my", "your", "his", "her",
}

_ROBERTA_CACHE: dict = {"name": None, "fill": None}


def _get_roberta_fill(model_name: str = "roberta-base"):
    from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

    if _ROBERTA_CACHE["fill"] is None or _ROBERTA_CACHE["name"] != model_name:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForMaskedLM.from_pretrained(model_name)
        _ROBERTA_CACHE["fill"] = pipeline("fill-mask", model=mdl, tokenizer=tok, top_k=50)
        _ROBERTA_CACHE["name"] = model_name
    return _ROBERTA_CACHE["fill"]


def _preserve_case(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src.istitle():
        return dst[:1].upper() + dst[1:]
    return dst


# ---------------------------------------------------------------------------
# Замена на контекстуальные синонимы
# ---------------------------------------------------------------------------

def roberta_synonyms(
    text: str,
    p: float = 0.15,
    max_replacements: int = 3,
    seed: Optional[int] = None,
    top_k: int = 25,
    model_name: str = "roberta-base",
    min_len: int = 4,
) -> str:
    rng = random.Random(seed)
    fill = _get_roberta_fill(model_name)

    toks = tokenize_words_punct(text)
    word_idxs = [i for i, t in enumerate(toks) if WORD_RE.fullmatch(t)]
    if not word_idxs:
        return text

    try:
        import nltk
        words = [toks[i] for i in word_idxs]
        tags = nltk.pos_tag(words)
    except Exception:
        return text

    def _ok_pos(tag: str) -> bool:
        return tag.startswith(("NN", "JJ", "VB", "RB")) and tag not in {"NNP", "NNPS"}

    candidates = []
    for j, (w, tag) in enumerate(tags):
        if not _ok_pos(tag):
            continue
        wl = w.lower()
        if len(w) < min_len or wl in STOPWORDS or any(ch.isdigit() for ch in w):
            continue
        candidates.append(word_idxs[j])

    if not candidates:
        return text

    rng.shuffle(candidates)
    replaced = 0

    for tok_i in candidates:
        if replaced >= max_replacements:
            break
        if rng.random() >= p:
            continue

        src = toks[tok_i]
        src_l = src.lower()

        masked = toks[:]
        masked[tok_i] = "<mask>"
        try:
            preds = fill(" ".join(masked))
        except Exception:
            continue

        best = None
        for d in preds[: max(1, min(top_k, len(preds)))]:
            cand = (d.get("token_str") or "").strip()
            if not WORD_RE.fullmatch(cand):
                continue
            if cand.lower() == src_l or cand.lower() in STOPWORDS:
                continue
            best = _preserve_case(src, cand)
            break

        if best:
            toks[tok_i] = best
            replaced += 1

    return detokenize(toks)
