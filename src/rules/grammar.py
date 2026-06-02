from __future__ import annotations
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from .text_utils import (
    WORD_RE,
    tokenize_words_punct,
    detokenize,
    _is_allcaps_short,
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

ARTICLES = {"a", "an", "the"}
DET_LIKE = ARTICLES | {
    "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "each", "every", "no",
}

IRREGULAR_VERBS: Dict[str, List[str]] = {
    "be": ["is", "are", "was", "were"], "is": ["be", "was"],
    "are": ["be", "were"], "was": ["be", "were"], "were": ["be", "was"],
    "have": ["has", "had"], "has": ["have", "had"], "had": ["have", "has"],
    "do": ["does", "did"], "does": ["do", "did"], "did": ["do", "does"],
    "go": ["goes", "went"], "went": ["go"],
    "make": ["makes", "made"], "made": ["make"],
}

CONTRACTIONS = [
    (r"\bdo not\b", "don't"),  (r"\bdoes not\b", "doesn't"),
    (r"\bdid not\b", "didn't"), (r"\bcan not\b",  "can't"),
    (r"\bwill not\b", "won't"), (r"\bI am\b",     "I'm"),
    (r"\byou are\b",  "you're"),(r"\bwe are\b",   "we're"),
    (r"\bthey are\b", "they're"),(r"\bit is\b",   "it's"),
]

MODALS: Dict[str, str] = {
    "can": "could", "could": "can",
    "may": "might", "might": "may",
    "will": "would", "would": "will",
    "shall": "should", "should": "shall",
}

PREPOSITIONS: Dict[str, List[str]] = {
    "in":    ["on", "at", "within"],
    "on":    ["in", "at", "upon"],
    "at":    ["in", "on", "by"],
    "for":   ["to", "of", "with"],
    "to":    ["for", "into", "toward"],
    "over":  ["above", "across", "at"],
    "under": ["below", "beneath", "at"],
    "with":  ["by", "along with", "using"],
    "about": ["on", "regarding", "around"],
}

# ---------------------------------------------------------------------------
# Вспомогательные морфологические функции
# ---------------------------------------------------------------------------

def _preserve_case(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src.istitle():
        return dst[:1].upper() + dst[1:]
    return dst


def _pluralize(w: str) -> str:
    wl = w.lower()
    if wl.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if wl.endswith("y") and len(w) > 2 and w[-2].lower() not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def _singularize(w: str) -> str:
    wl = w.lower()
    if wl.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if wl.endswith("es") and len(w) > 2 and (
        wl[-3:-1] in {"ch", "sh"} or wl[-3] in {"s", "x", "z"}
    ):
        return w[:-2]
    if wl.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


def _ly_remove(w: str) -> str:
    wl = w.lower()
    if len(w) <= 4 or not wl.endswith("ly") or w[:-2].lower().endswith("i"):
        return w
    return w[:-2]


def _ly_add(w: str) -> str:
    wl = w.lower()
    if wl.endswith("ly"):
        return w
    if wl.endswith("ic"):
        return w + "ally"
    if wl.endswith("y"):
        return w[:-1] + "ily"
    return w + "ly"


# ---------------------------------------------------------------------------
# Искажение грамматики
# ---------------------------------------------------------------------------

def grammar_distort(
    text: str,
    p: float = 0.12,
    p_contr: float = 0.25,
    p_article_insert: float = 0.10,
    p_prep: float = 0.06,
    max_ops_total: int = 8,
    max_ops_per_sent: int = 2,
    seed: Optional[int] = None,
) -> str:
    rng = random.Random(seed)
    if rng.random() < p_contr:
        for pat, repl in rng.sample(CONTRACTIONS, k=rng.randint(1, 2)):
            text = re.sub(pat, repl, text, flags=re.IGNORECASE)

    toks = tokenize_words_punct(text)
    widx = [i for i, t in enumerate(toks) if WORD_RE.fullmatch(t)]
    if not widx:
        return text

    try:
        import nltk
        tags = nltk.pos_tag([toks[i] for i in widx])
    except Exception:
        return text

    def _sent_ids(ts: List[str]) -> Dict[int, int]:
        sid = 0
        m: Dict[int, int] = {}
        for i, t in enumerate(ts):
            m[i] = sid
            if t in {".", "!", "?"}:
                sid += 1
        return m

    out = toks[:]
    tok2sent = _sent_ids(out)
    ops_total = 0
    ops_by_sent: Dict[int, int] = defaultdict(int)

    inserts: List[Tuple[int, str]] = []
    for j, (w, tag) in enumerate(tags):
        if ops_total >= max_ops_total:
            break
        if tag != "NN":
            continue
        ti = widx[j]
        sid = tok2sent.get(ti, 0)
        if ops_by_sent[sid] >= max_ops_per_sent:
            continue
        if (
            any(ch.isdigit() for ch in w)
            or _is_allcaps_short(w)
            or tag in {"NNP", "NNPS"}
        ):
            continue
        prev_w = tags[j - 1][0].lower() if j > 0 else None
        prev_t = tags[j - 1][1] if j > 0 else None
        if prev_w in DET_LIKE or (prev_t and prev_t.startswith("JJ")):
            continue
        if rng.random() < p_article_insert:
            art = "an" if w[:1].lower() in "aeiou" else "a"
            if rng.random() < 0.25:
                art = "the"
            inserts.append((ti, art))
            ops_total += 1
            ops_by_sent[sid] += 1

    for pos, art in sorted(inserts, reverse=True):
        out[pos:pos] = [art]

    widx2 = [i for i, t in enumerate(out) if WORD_RE.fullmatch(t)]
    try:
        import nltk
        tags2 = nltk.pos_tag([out[i] for i in widx2])
    except Exception:
        return detokenize(out)

    tok2sent2 = _sent_ids(out)

    for j, (w, tag) in enumerate(tags2):
        if ops_total >= max_ops_total:
            break
        ti = widx2[j]
        sid = tok2sent2.get(ti, 0)
        if ops_by_sent[sid] >= max_ops_per_sent:
            continue
        wl = w.lower()
        if any(ch.isdigit() for ch in w) or _is_allcaps_short(w) or tag in {"NNP", "NNPS"}:
            continue

        if tag in {"NN", "NNS"} and rng.random() < p:
            out[ti] = _preserve_case(w, _pluralize(w) if tag == "NN" else _singularize(w))
            ops_total += 1; ops_by_sent[sid] += 1; continue

        if tag.startswith("RB") and wl.endswith("ly") and rng.random() < p:
            out[ti] = _preserve_case(w, _ly_remove(w))
            ops_total += 1; ops_by_sent[sid] += 1; continue

        if tag.startswith("JJ") and rng.random() < p:
            if ti == 0 or out[ti - 1] in {".", "!", "?"} or wl.endswith("ly"):
                continue
            out[ti] = _preserve_case(w, _ly_add(w))
            ops_total += 1; ops_by_sent[sid] += 1; continue

        if tag == "MD" and wl in MODALS and rng.random() < p:
            out[ti] = _preserve_case(w, MODALS[wl])
            ops_total += 1; ops_by_sent[sid] += 1; continue

        if tag in {"IN", "TO"} and wl in PREPOSITIONS and rng.random() < p_prep:
            cand = rng.choice(PREPOSITIONS[wl])
            out[ti : ti + 1] = cand.split() if " " in cand else [_preserve_case(w, cand)]
            ops_total += 1; ops_by_sent[sid] += 1; continue

        if tag.startswith("VB") and rng.random() < p:
            if tag == "VBG":
                continue
            if wl in IRREGULAR_VERBS:
                out[ti] = _preserve_case(w, rng.choice(IRREGULAR_VERBS[wl]))
            elif tag in {"VBD", "VBN"} and wl.endswith("ed"):
                stem = w[:-2]
                if len(stem) >= 3:
                    out[ti] = stem
                else:
                    continue
            elif tag == "VBZ":
                out[ti] = w.rstrip("s")
            else:
                out[ti] = w + "ed"
            ops_total += 1; ops_by_sent[sid] += 1; continue

        if wl in ARTICLES and rng.random() < p:
            out[ti] = "" if rng.random() < 0.4 else rng.choice(["a", "the", "an"])
            ops_total += 1; ops_by_sent[sid] += 1; continue

    return detokenize([t for t in out if t])
