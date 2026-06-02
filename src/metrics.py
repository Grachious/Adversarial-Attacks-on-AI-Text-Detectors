from __future__ import annotations
import math
import re
from typing import Any, Dict, Optional, Tuple
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from wordfreq import zipf_frequency

# ---------------------------------------------------------------------------
# Перплексия (GPT-2)
# ---------------------------------------------------------------------------

PPL_MODEL_NAME = "gpt2"

_ppl_tokenizer = None
_ppl_model = None
_ppl_device = None


def _load_ppl_model():
    global _ppl_tokenizer, _ppl_model, _ppl_device
    if _ppl_model is not None:
        return
    _ppl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ppl_tokenizer = GPT2TokenizerFast.from_pretrained(PPL_MODEL_NAME)
    _ppl_model = GPT2LMHeadModel.from_pretrained(PPL_MODEL_NAME).to(_ppl_device)
    _ppl_model.eval()


def perplexity(text: str, max_length: int = 256) -> float:
    _load_ppl_model()
    enc = _ppl_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc.input_ids.to(_ppl_device)
    with torch.no_grad():
        loss = _ppl_model(input_ids, labels=input_ids).loss
    return torch.exp(loss).item()


# ---------------------------------------------------------------------------
# Критерий человекочитаемости
# ---------------------------------------------------------------------------

def human_readability_score(text: str, return_details: bool = True) -> Tuple[float, Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return (0.0, {"reason": "empty"}) if return_details else (0.0, {})

    word_re = r"[A-Za-z]+(?:'[A-Za-z]+)?"
    words = re.findall(word_re, t)
    sents = [s.strip() for s in re.findall(r"[^.!?]+[.!?]?", t) if s.strip()] or [t]
    n_chars = len(t)
    n_words = len(words)

    letters = sum(c.isalpha() for c in t)
    spaces  = sum(c.isspace() for c in t)
    punct   = len(re.findall(r"[.,;:!?]", t))
    weird   = len(re.findall(r"[^\x09\x0A\x0D\x20-\x7E]", t))

    letters_ratio   = letters / n_chars if n_chars else 0.0
    space_ratio     = spaces  / n_chars if n_chars else 0.0
    punct_per_word  = punct   / n_words if n_words else 0.0
    weird_ratio     = weird   / n_chars if n_chars else 0.0

    sent_lens = [len(re.findall(word_re, s)) for s in sents if re.search(word_re, s)]
    avg_sent_len = (sum(sent_lens) / len(sent_lens)) if sent_lens else float(n_words)

    def _band(x: float, lo: float, hi: float, soft: float = 0.6) -> float:
        if lo <= x <= hi:
            return 1.0
        d = (
            (lo - x) / (soft * (hi - lo + 1e-9))
            if x < lo
            else (x - hi) / (soft * (hi - lo + 1e-9))
        )
        return max(0.0, min(1.0, math.exp(-d)))

    structure = (
        0.45 * _band(letters_ratio, 0.60, 0.92, 0.5)
        + 0.35 * _band(space_ratio,   0.10, 0.28, 0.7)
        + 0.20 * _band(weird_ratio,   0.00, 0.002, 1.0)
    )
    punct_s  = _band(punct_per_word, 0.00, 0.20, 0.9)
    sentlen  = _band(avg_sent_len,   6.0,  35.0, 0.8)
    length_s = _band(n_words,        8,    400,  1.0)

    if zipf_frequency and words:
        zipfs = [zipf_frequency(w.lower(), "en") for w in words]
        real_ratio = sum(z >= 2.5 for z in zipfs) / len(zipfs)
        mean_zipf  = sum(zipfs) / len(zipfs)
        lexical = (
            0.65 * _band(real_ratio, 0.70, 0.98, 0.7)
            + 0.35 * _band(mean_zipf, 2.8,  6.0,  0.8)
        )
    else:
        real_ratio = mean_zipf = None
        lexical = 0.60

    rep_rate = len(re.findall(r"([A-Za-z])\1{2,}", t)) / max(1, n_words)
    odd_rate = (
        sum(
            1 for w in words
            if len(w) >= 6 and re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", w.lower())
        ) / len(words)
    ) if words else 0.0
    typo = (
        0.5 * _band(rep_rate, 0.0, 0.04, 1.2)
        + 0.5 * _band(odd_rate, 0.0, 0.06, 1.2)
    )

    score = max(
        0.0,
        min(
            1.0,
            0.26 * structure
            + 0.12 * punct_s
            + 0.20 * sentlen
            + 0.22 * lexical
            + 0.12 * typo
            + 0.08 * length_s,
        ),
    )

    if not return_details:
        return score, {}

    return score, {
        "score": score,
        "n_words": n_words,
        "letters_ratio": letters_ratio,
        "space_ratio": space_ratio,
        "punct_per_word": punct_per_word,
        "avg_sent_len": avg_sent_len,
        "weird_ratio": weird_ratio,
        "lex_real_ratio": real_ratio,
        "lex_mean_zipf": mean_zipf,
        "components": {
            "structure": structure,
            "punct":     punct_s,
            "sent_len":  sentlen,
            "lexical":   lexical,
            "typo_proxy": typo,
            "length":    length_s,
        },
        "wordfreq_available": bool(zipf_frequency),
    }
