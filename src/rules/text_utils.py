from __future__ import annotations
import re
import string
from typing import List

# ---------------------------------------------------------------------------
# Скомпилированные регулярные выражения
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.UNICODE)
_PUNCT = r",\.!\?;:"

# ---------------------------------------------------------------------------
# Нормализация пробелов
# ---------------------------------------------------------------------------

def normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"(\w)\s*-\s*(\w)", r"\1-\2", text)
    return text

# ---------------------------------------------------------------------------
# Токенизация и детокенизация
# ---------------------------------------------------------------------------

def tokenize_words_punct(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: List[str]) -> str:
    s = " ".join(t for t in tokens if t)
    s = re.sub(r"'\s+(\w+)\s+'", r"'\1'", s)
    s = re.sub(
        r"\b([A-Za-z]+)\s+'\s*(ll|re|ve|d|m|s|t)\b",
        r"\1'\2",
        s,
        flags=re.IGNORECASE,
    )
    return normalize_spacing(s)

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _is_allcaps_short(word: str) -> bool:
    return word.isupper() and len(word) <= 4


def _cleanup_after_phrase_deletion(text: str) -> str:
    text = re.sub(r"\.\s*,", ".", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(rf"\s+([{_PUNCT}])", r"\1", text)
    text = re.sub(rf"([{_PUNCT}])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    def _cap(m):
        return m.group(1) + m.group(2).upper()

    text = re.sub(r"([.!?]\s+)([a-z])", _cap, text)
    return text


def _lowercase_after_starter(toks: List[str], pos: int) -> None:
    for j in range(pos, len(toks)):
        if toks[j].isalpha():
            if toks[j] != "I" and not toks[j].isupper():
                toks[j] = toks[j][0].lower() + toks[j][1:]
            break
