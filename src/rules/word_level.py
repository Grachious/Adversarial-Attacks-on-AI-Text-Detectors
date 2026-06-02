from __future__ import annotations
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple

from .text_utils import (
    WORD_RE,
    tokenize_words_punct,
    detokenize,
    normalize_spacing,
    _is_allcaps_short,
    _cleanup_after_phrase_deletion,
    _lowercase_after_starter,
)


# ---------------------------------------------------------------------------
# Умное удаление слов
# ---------------------------------------------------------------------------

def smart_delete_words(
    text: str,
    p: float = 0.10,
    max_deletes: int = 3,
    seed: Optional[int] = None,
    protected: Optional[Iterable[str]] = None,
) -> str:
    rng = random.Random(seed)
    protected_set = {w.lower() for w in (protected or [])}

    NEVER_DELETE = {"not", "no", "never", "n't", "without", "unless", "until"}

    # Клише, характерные для AI-текстов — удаляем целиком
    SAFE_PHRASE_PATTERNS = [
        r"in particular", r"in general", r"as a result", r"on the other hand",
        r"for instance", r"for example", r"in conclusion", r"to some extent",
        r"in most cases", r"it seems that", r"it is possible that",
    ]

    t = text
    phrase_hits = [
        ph for ph in SAFE_PHRASE_PATTERNS
        if re.search(rf"\b{ph}\b", t, flags=re.IGNORECASE)
    ]
    if phrase_hits and rng.random() < (p * 0.6):
        rng.shuffle(phrase_hits)
        n_ph = rng.randint(1, min(2, len(phrase_hits)))
        for ph in phrase_hits[:n_ph]:
            t = re.sub(rf"\b{ph}\b\s*,?\s*", "", t, flags=re.IGNORECASE)
        t = _cleanup_after_phrase_deletion(t)

    toks = tokenize_words_punct(t)
    word_idxs = [i for i, x in enumerate(toks) if WORD_RE.fullmatch(x)]
    if not word_idxs:
        return t

    words = [toks[i] for i in word_idxs]
    try:
        import nltk
        tags = nltk.pos_tag(words)
    except Exception:
        tags = None

    # Слова, удаление которых снижает "AI-ность"
    SAFE_WORDS = {
        "however", "therefore", "moreover", "nevertheless", "thus", "also",
        "instead", "overall", "indeed", "furthermore", "additionally",
        "consequently", "similarly", "likewise", "otherwise", "meanwhile",
        "specifically", "hence", "accordingly", "nonetheless", "generally",
        "typically", "often", "usually", "sometimes", "largely", "mostly",
        "arguably", "probably", "likely", "somewhat", "rather", "quite",
        "frequently", "occasionally", "normally", "primarily", "mainly",
        "relatively",
    }

    candidates: List[Tuple[int, int]] = []
    for j, w in enumerate(words):
        wl = w.lower()
        if wl in NEVER_DELETE or wl in protected_set:
            continue
        if any(ch.isdigit() for ch in wl):
            continue
        if _is_allcaps_short(w):
            continue

        if tags is not None:
            _, tag = tags[j]
            if wl in SAFE_WORDS:
                candidates.append((j, 4))
            elif tag.startswith("RB"):
                candidates.append((j, 3))
            elif tag.startswith("JJ") and len(w) > 3:
                candidates.append((j, 2))
            elif tag == "DT":
                candidates.append((j, 1))
        else:
            if len(w) >= 5:
                candidates.append((j, 1))

    if not candidates:
        return detokenize(toks)

    idxs = [c[0] for c in candidates]
    weights = [c[1] for c in candidates]

    out = toks[:]
    deleted = 0
    while deleted < max_deletes and idxs and (rng.random() < p):
        pick = rng.choices(idxs, weights=weights, k=1)[0]
        out[word_idxs[pick]] = ""
        deleted += 1
        rm = idxs.index(pick)
        idxs.pop(rm)
        weights.pop(rm)

    return _cleanup_after_phrase_deletion(detokenize([x for x in out if x]))


# ---------------------------------------------------------------------------
# Перестановка синтаксических фраз
# ---------------------------------------------------------------------------

_SPACY_NLP = None


def swap_phrases_spacy(
    text: str,
    n_ops: int = 1,
    seed: Optional[int] = None,
) -> str:
    rng = random.Random(seed)
    try:
        import spacy
        global _SPACY_NLP
        if _SPACY_NLP is None:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        nlp = _SPACY_NLP
    except Exception:
        return text

    def _cand_spans(doc):
        spans = []
        for t in doc:
            if t.is_punct or t.is_space:
                continue
            if t.dep_ not in {"prep", "advcl", "appos", "acl", "relcl", "npadvmod", "advmod"}:
                continue
            idxs = [x.i for x in t.subtree]
            s, e = min(idxs), max(idxs) + 1
            if 2 <= (e - s) <= 14:
                spans.append((s, e))
        return list(dict.fromkeys(spans))

    cur = text
    for _ in range(max(0, int(n_ops))):
        doc = nlp(cur)
        spans = _cand_spans(doc)
        if len(spans) < 2:
            break

        rng.shuffle(spans)
        picked = None
        for a in spans:
            for b in spans:
                if a == b:
                    continue
                s1, e1 = a
                s2, e2 = b
                if e1 <= s2 or e2 <= s1:
                    picked = (a, b)
                    break
            if picked:
                break
        if not picked:
            break

        (s1, e1), (s2, e2) = sorted(picked, key=lambda x: x[0])
        toks = [t.text_with_ws for t in doc]
        chunk1, chunk2 = toks[s1:e1], toks[s2:e2]
        cur = "".join(toks[:s1] + chunk2 + toks[e1:s2] + chunk1 + toks[e2:])

    return normalize_spacing(cur)


# ---------------------------------------------------------------------------
# Добавление стоп-слов и дискурсивных маркеров
# ---------------------------------------------------------------------------

def add_stopwords(
    text: str,
    p: float = 0.15,
    max_inserts: int = 3,
    seed: Optional[int] = None,
) -> str:
    rng = random.Random(seed)

    SENT_START = [
        "Actually,", "Well,", "So,", "In fact,", "Honestly,", "Indeed,",
        "Overall,", "Generally,", "Basically,", "Therefore,", "Thus,", "Meanwhile,",
    ]
    AFTER_COMMA = [
        "however,", "though,", "still,", "yet,", "also,", "too,",
        "as well,", "specifically,", "therefore,", "thus,", "as a result,",
    ]
    PRE_VERB = [
        "often", "usually", "sometimes", "frequently", "occasionally",
        "actually", "basically", "generally", "typically", "probably",
        "likely", "arguably", "perhaps", "mostly", "largely", "mainly", "primarily",
    ]
    PRE_ADJ = [
        "very", "quite", "rather", "extremely", "highly", "somewhat",
        "fairly", "relatively", "moderately", "pretty", "really",
    ]
    MARKERS = {x.rstrip(",").lower() for x in (SENT_START + AFTER_COMMA)}

    toks = tokenize_words_punct(text)
    widx = [i for i, t in enumerate(toks) if WORD_RE.fullmatch(t)]
    if len(widx) < 3:
        return text

    try:
        import nltk
        tags = nltk.pos_tag([toks[i] for i in widx])
    except Exception:
        tags = None

    candidates: List[Tuple[str, int, int]] = []
    sent_starts = [0] + [
        i + 1 for i, t in enumerate(toks[:-1]) if t in {".", "!", "?"}
    ]
    candidates += [("sent_start", pos, 3) for pos in sent_starts]
    candidates += [
        ("after_comma", i + 1, 2) for i, t in enumerate(toks[:-1]) if t in {",", ";"}
    ]
    if tags:
        for j, (_, tg) in enumerate(tags):
            pos = widx[j]
            if tg.startswith("VB"):
                candidates.append(("pre_verb", pos, 2))
            elif tg.startswith("JJ"):
                candidates.append(("pre_adj", pos, 1))

    candidates.sort(key=lambda x: x[2], reverse=True)

    used: set = set()
    chosen: List[Tuple[str, int]] = []
    for kind, pos, _ in candidates:
        if len(chosen) >= max_inserts:
            break
        if pos in used or (pos - 1) in used or (pos + 1) in used:
            continue
        if rng.random() >= p:
            continue
        if (pos > 0 and toks[pos - 1] == "-") or (pos < len(toks) and toks[pos] == "-"):
            continue
        if pos > 0 and toks[pos - 1].lower() == "to":
            continue
        if (
            pos > 0
            and WORD_RE.fullmatch(toks[pos - 1])
            and toks[pos - 1].lower() in MARKERS
        ):
            continue
        chosen.append((kind, pos))
        used.add(pos)

    out = toks[:]
    fix_after: List[int] = []
    for kind, pos in sorted(chosen, key=lambda x: x[1], reverse=True):
        if kind == "sent_start":
            ins = rng.choice(SENT_START).split()
            out[pos:pos] = ins
            fix_after.append(pos + len(ins))
        elif kind == "after_comma":
            out[pos:pos] = rng.choice(AFTER_COMMA).split()
        elif kind == "pre_verb":
            out[pos:pos] = [rng.choice(PRE_VERB)]
        else:
            out[pos:pos] = [rng.choice(PRE_ADJ)]

    for pos in fix_after:
        _lowercase_after_starter(out, pos)

    return detokenize(out)
