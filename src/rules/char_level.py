from __future__ import annotations
import random
import re
import string
from typing import Optional
from .text_utils import WORD_RE


# ---------------------------------------------------------------------------
# Гомоглифы (Кириллица ↔ Латиница)
# ---------------------------------------------------------------------------

HOMOGLYPHS: dict[str, str] = {
    "a": "а", "а": "a",
    "e": "е", "е": "e",
    "o": "о", "о": "o",
    "c": "с", "с": "c",
    "p": "р", "р": "p",
    "x": "х", "х": "x",
    "y": "у", "у": "y",
    "B": "В", "В": "B",
    "E": "Е", "Е": "E",
    "K": "К", "К": "K",
    "M": "М", "М": "M",
    "H": "Н", "Н": "H",
    "O": "О", "О": "O",
    "P": "Р", "Р": "P",
    "C": "С", "С": "C",
    "T": "Т", "Т": "T",
    "X": "Х", "Х": "X",
    "A": "А", "А": "A",
}


def homoglypgs_random(text: str, p: float = 0.2, seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    return "".join(
        HOMOGLYPHS[ch] if (ch in HOMOGLYPHS and rng.random() < p) else ch
        for ch in text
    )


# ---------------------------------------------------------------------------
# Перестановка соседних букв
# ---------------------------------------------------------------------------

def swap_chars_random(
    text: str,
    p: float = 0.02,
    seed: Optional[int] = None,
    min_len: int = 4,
) -> str:
    rng = random.Random(seed)
    spans = [(m.start(), m.end()) for m in re.finditer(r"\w+", text)]
    chars = list(text)

    for start, end in spans:
        word = text[start:end]
        if len(word) < min_len or any(ch.isupper() for ch in word):
            continue
        candidates = [
            i for i in range(start, end - 1)
            if chars[i].isalnum() and chars[i + 1].isalnum()
        ]
        if not candidates:
            continue
        if rng.random() < p:
            i = rng.choice(candidates)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]

    return "".join(chars)


# ---------------------------------------------------------------------------
# Вставка случайных букв
# ---------------------------------------------------------------------------

def insert_random_char(
    text: str,
    p: float = 0.02,
    seed: Optional[int] = None,
    alphabet: Optional[str] = None,
) -> str:
    rng = random.Random(seed)
    if alphabet is None:
        alphabet = string.ascii_lowercase
    alphabet = "".join(ch for ch in alphabet if ch.isalpha() and ch.islower()) or "abcdefghijklmnopqrstuvwxyz"

    out = []
    for ch in text:
        out.append(ch)
        if ch.isalnum() and rng.random() < p:
            out.append(rng.choice(alphabet))
    return "".join(out)


# ---------------------------------------------------------------------------
# Удвоение случайной буквы
# ---------------------------------------------------------------------------

def double_random_letter(
    text: str,
    p: float = 0.10,
    seed: Optional[int] = None,
) -> str:
    rng = random.Random(seed)
    words = text.split()
    out = []
    for w in words:
        core = re.sub(r"\W+$", "", w)
        suffix = w[len(core):]
        if len(core) >= 3 and rng.random() < p:
            idxs = [i for i, ch in enumerate(core) if ch.isalpha()]
            if idxs:
                i = rng.choice(idxs)
                core = core[:i] + core[i] + core[i:]
        out.append(core + suffix)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Удаление случайной буквы
# ---------------------------------------------------------------------------

def delete_random_letter(
    text: str,
    p: float = 0.10,
    seed: Optional[int] = None,
) -> str:
    rng = random.Random(seed)
    words = text.split()
    out = []
    for w in words:
        core = re.sub(r"\W+$", "", w)
        suffix = w[len(core):]
        if len(core) >= 4 and rng.random() < p:
            idxs = [i for i, ch in enumerate(core) if ch.isalpha() and i > 0]
            if idxs:
                i = rng.choice(idxs)
                core = core[:i] + core[i + 1:]
        out.append(core + suffix)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Замена на ближайшую клавишу клавиатуры
# ---------------------------------------------------------------------------

QWERTY_NEIGHBORS: dict[str, str] = {
    "q": "was",   "w": "qase",  "e": "wsdr",  "r": "edft",  "t": "rfgy",
    "y": "tghu",  "u": "yhji",  "i": "ujko",  "o": "iklp",  "p": "ol",
    "a": "qwsz",  "s": "awedxz","d": "serfcx","f": "drtgvc","g": "ftyhbv",
    "h": "gyujnb","j": "huikmn","k": "jiolm", "l": "kop",
    "z": "asx",   "x": "zsdc",  "c": "xdfv",  "v": "cfgb",  "b": "vghn",
    "n": "bhjm",  "m": "njk",
}


def keyboard_miss(
    text: str,
    p: float = 0.03,
    seed: Optional[int] = None,
) -> str:
    rng = random.Random(seed)
    parts = re.findall(r"\w+|[^\w]+", text, flags=re.UNICODE)
    out = []
    for part in parts:
        if not re.fullmatch(r"\w+", part, flags=re.UNICODE):
            out.append(part)
            continue
        idxs = [
            i for i, ch in enumerate(part)
            if ch.isalpha() and ch.lower() in QWERTY_NEIGHBORS
        ]
        if not idxs or rng.random() >= p:
            out.append(part)
            continue
        i = rng.choice(idxs)
        ch = part[i]
        repl = rng.choice(QWERTY_NEIGHBORS[ch.lower()])
        repl = repl.upper() if ch.isupper() else repl
        out.append(part[:i] + repl + part[i + 1:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Фонетические и графические замены
# ---------------------------------------------------------------------------

COMMON_WRONG: dict[str, list[str]] = {
    "i": ["y"], "y": ["i"],
    "s": ["z"], "z": ["s"],
    "c": ["k", "s"],
    "k": ["c"],
    "g": ["j"], "j": ["g"],
    "v": ["w"], "w": ["v"],
}

COMMON_MULTI: dict[str, list[str]] = {
    "f":  ["ph"],
    "ph": ["f"],
    "x":  ["gz", "ks"],
    "ks": ["x"],
}


def replace_equivalents(
    text: str,
    p: float = 0.03,
    allow_multi: bool = True,
    seed: Optional[int] = None,
) -> str:
    rng = random.Random(seed)
    t = text

    if allow_multi:
        i = 0
        out = []
        keys = sorted(COMMON_MULTI.keys(), key=len, reverse=True)
        while i < len(t):
            matched = False
            for k in keys:
                seg = t[i : i + len(k)]
                if seg.lower() == k and rng.random() < p:
                    repl = rng.choice(COMMON_MULTI[k])
                    if seg and seg[0].isupper():
                        repl = repl[0].upper() + repl[1:]
                    out.append(repl)
                    i += len(k)
                    matched = True
                    break
            if not matched:
                out.append(t[i])
                i += 1
        t = "".join(out)

    out2 = []
    for ch in t:
        if ch.isalpha():
            low = ch.lower()
            if low in COMMON_WRONG and rng.random() < p:
                repl = rng.choice(COMMON_WRONG[low])
                out2.append(repl.upper() if ch.isupper() else repl)
            else:
                out2.append(ch)
        else:
            out2.append(ch)
    return "".join(out2)
