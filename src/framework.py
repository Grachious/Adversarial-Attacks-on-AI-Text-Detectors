from __future__ import annotations
import inspect
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
from .rules.text_utils import normalize_spacing


# ---------------------------------------------------------------------------
# RuleSpec
# ---------------------------------------------------------------------------

@dataclass
class RuleSpec:
    name: str
    fn: Callable[..., str]
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    bucket: str = "misc" 
    def __post_init__(self):
        if self.default_kwargs is None:
            self.default_kwargs = {}
        if not isinstance(self.default_kwargs, dict):
            raise TypeError(
                f"Rule '{self.name}': default_kwargs must be dict, "
                f"got {type(self.default_kwargs).__name__}: {self.default_kwargs!r}"
            )
        if not isinstance(self.bucket, str) or not self.bucket:
            self.bucket = "misc"

    def sample_kwargs(self, strength: float) -> Dict[str, Any]:
        kw = dict(self.default_kwargs)
        mult = 0.6 + 0.8 * strength

        if "p" in kw:
            kw["p"] = min(1.0, max(0.0, float(kw["p"]) * mult))
        if "p_contr" in kw:
            kw["p_contr"] = min(1.0, max(0.0, float(kw["p_contr"]) * (0.6 + 0.6 * strength)))
        if "max_inserts" in kw:
            kw["max_inserts"] = max(1, int(round(int(kw["max_inserts"]) * mult)))
        if "max_deletes" in kw:
            kw["max_deletes"] = max(1, int(round(int(kw["max_deletes"]) * mult)))
        if "n_ops" in kw:
            kw["n_ops"] = max(1, int(round(int(kw["n_ops"]) * (0.6 + 0.8 * strength))))
        if "max_replacements" in kw:
            kw["max_replacements"] = max(1, int(round(int(kw["max_replacements"]) * mult)))
        return kw


# ---------------------------------------------------------------------------
# TextBreakFramework
# ---------------------------------------------------------------------------

class TextBreakFramework:
    """
    Фреймворк для состязательных атаки на детектор.

    Выбирает случайные комбинации правил, применяет их к тексту,
    оценивает результат и возвращает лучший вариант.

    Parameters
    ----------
    rulebook : dict[str, RuleSpec]
        Словарь правил {name: RuleSpec}.
    radar_score_fn : callable
        Функция-детектор: text → (prob_ai, prob_human, logits).
    perplexity_fn : callable
        Функция перплексии: text → float.
    readability_fn : callable
        Функция читаемости: text → (score, details).
    normalize_fn : callable, optional
        Пост-обработка после применения правил (по умолчанию: normalize_spacing).
    min_readability : float
        Минимально допустимая читаемость кандидата.
    max_ppl_ratio : float
        Максимально допустимый рост перплексии относительно оригинала.
    max_ppl_abs : float, optional
        Максимально допустимая абсолютная перплексия кандидата.
    """

    def __init__(
        self,
        rulebook: Dict[str, RuleSpec],
        radar_score_fn: Callable[[str], Tuple[float, float, Any]],
        perplexity_fn: Callable[[str], float],
        readability_fn: Callable[[str], Tuple[float, Dict[str, Any]]],
        normalize_fn: Optional[Callable[[str], str]] = None,
        min_readability: float = 0.90,
        max_ppl_ratio: float = 6.0,
        max_ppl_abs: Optional[float] = None,
    ):
        self.rulebook = rulebook
        self.radar_score = radar_score_fn
        self.perplexity = perplexity_fn
        self.readability = readability_fn
        self.normalize = normalize_fn or normalize_spacing

        self.min_readability = float(min_readability)
        self.max_ppl_ratio = float(max_ppl_ratio)
        self.max_ppl_abs = max_ppl_abs

        # Проверяем, что все правила принимают seed или rng
        for name, rs in self.rulebook.items():
            sig = inspect.signature(rs.fn)
            if "seed" not in sig.parameters and "rng" not in sig.parameters:
                raise ValueError(
                    f"Rule '{name}' must accept 'seed' or 'rng' parameter. "
                    f"Got signature: {sig}"
                )

    def _call_rule(
        self,
        fn: Callable,
        text: str,
        rng: random.Random,
        kwargs: Dict[str, Any],
    ) -> str:
        sig = inspect.signature(fn)
        if "seed" in sig.parameters:
            return fn(text, seed=rng.randint(0, 10**9), **kwargs)
        if "rng" in sig.parameters:
            return fn(text, rng=rng, **kwargs)
        return fn(text, **kwargs)

    def measure(self, text: str) -> Dict[str, float]:
        ai, human, _ = self.radar_score(text)
        ppl = self.perplexity(text)
        read, _ = self.readability(text)
        return {
            "radar_ai":    float(ai),
            "radar_human": float(human),
            "ppl":         float(ppl),
            "read":        float(read),
        }

    def passes_constraints(
        self,
        base: Dict[str, float],
        cand: Dict[str, float],
    ) -> bool:
        if cand["read"] < self.min_readability:
            return False
        if self.max_ppl_abs is not None and cand["ppl"] > float(self.max_ppl_abs):
            return False
        if base["ppl"] > 0 and cand["ppl"] > base["ppl"] * self.max_ppl_ratio:
            return False
        return True

    # ------------------------------------------------------------------
    # Выбор комбинации правил
    # ------------------------------------------------------------------

    def pick_combo(self, rng: random.Random, k_max: int = 3) -> List[str]:
        buckets: Dict[str, List[str]] = defaultdict(list)
        for name, r in self.rulebook.items():
            b = r.bucket if isinstance(r.bucket, str) and r.bucket else "misc"
            buckets[b].append(name)

        chosen: List[str] = []

        if buckets.get("char"):
            chosen.append(rng.choice(buckets["char"]))

        pool = buckets.get("word", []) + buckets.get("grammar", [])
        if pool and rng.random() < 0.9 and len(chosen) < k_max:
            chosen.append(rng.choice(pool))

        if buckets.get("punct") and rng.random() < 0.5 and len(chosen) < k_max:
            chosen.append(rng.choice(buckets["punct"]))

        if not chosen:
            chosen.append(rng.choice(list(self.rulebook.keys())))

        return list(dict.fromkeys(chosen))  # дедупликация с сохранением порядка

    # ------------------------------------------------------------------
    # Применение правил
    # ------------------------------------------------------------------

    def apply_rule(
        self,
        text: str,
        rule_name: str,
        seed: int,
        strength: float = 0.5,
        override_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        rs = self.rulebook[rule_name]
        kw = rs.sample_kwargs(strength)
        if override_kwargs:
            kw.update(override_kwargs)

        rng = random.Random(seed)
        out = self._call_rule(rs.fn, text, rng=rng, kwargs=kw)
        out = self.normalize(out)

        meta = {
            "rule":   rs.name,
            "bucket": rs.bucket,
            "seed":   int(seed),
            "kwargs": kw,
        }
        return out, meta

    def apply_combo(
        self,
        text: str,
        combo: List[str],
        seed: int,
        strength: float = 0.5,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        rng = random.Random(seed)
        t = text
        log: List[Dict[str, Any]] = []
        for rule_name in combo:
            step_seed = rng.randint(0, 10**9)
            t, meta = self.apply_rule(t, rule_name, seed=step_seed, strength=strength)
            log.append(meta)
        return t, log

    # ------------------------------------------------------------------
    # Основной метод: поиск лучшего кандидата
    # ------------------------------------------------------------------

    def generate_best_candidate(
        self,
        text: str,
        tries: int = 16,
        strength: float = 0.5,
        seed: int = 0,
        k_max: int = 3,
        target_drop: float = 0.0,
    ) -> Dict[str, Any]:
        rng = random.Random(seed)
        base_m = self.measure(text)
        base_ai = base_m["radar_ai"]

        best = {
            "text": text, "log": [], "base": base_m,
            "cand": base_m, "combo": [],
        }
        best_ai = base_ai

        for _ in range(tries):
            combo_seed = rng.randint(0, 10**9)
            combo_rng  = random.Random(combo_seed)
            combo = self.pick_combo(combo_rng, k_max=k_max)

            cand_text, log = self.apply_combo(text, combo, seed=combo_seed, strength=strength)
            cand_m = self.measure(cand_text)

            if not self.passes_constraints(base_m, cand_m):
                continue

            if cand_m["radar_ai"] < best_ai:
                best_ai = cand_m["radar_ai"]
                best = {
                    "text": cand_text, "log": log,
                    "base": base_m, "cand": cand_m, "combo": combo,
                }

            if cand_m["radar_ai"] <= base_ai - float(target_drop):
                return best

        return best

    # ------------------------------------------------------------------
    # Анализ отдельных правил
    # ------------------------------------------------------------------

    def best_per_rule(
        self,
        text: str,
        n_runs: int = 5,
        strength: float = 0.5,
        seed: int = 0,
    ) -> pd.DataFrame:
        rows = []
        base = self.measure(text)

        for rname in self.rulebook:
            best_row = None
            best_ai = float("inf")

            for i in range(n_runs):
                run_seed = seed + abs(hash(rname)) % 10_000_000 + i
                cand_text, log = self.apply_rule(text, rname, seed=run_seed, strength=strength)
                cand_m = self.measure(cand_text)

                if not self.passes_constraints(base, cand_m):
                    continue

                if cand_m["radar_ai"] < best_ai:
                    best_ai = cand_m["radar_ai"]
                    best_row = {
                        "rule":          rname,
                        "best_text":     cand_text,
                        "radar_ai":      cand_m["radar_ai"],
                        "ppl":           cand_m["ppl"],
                        "read":          cand_m["read"],
                        "base_radar_ai": base["radar_ai"],
                        "base_ppl":      base["ppl"],
                        "base_read":     base["read"],
                    }

            if best_row:
                rows.append(best_row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("radar_ai", ascending=True).reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# Построение стандартной книги правил
# ---------------------------------------------------------------------------

def build_rulebook(rules: Dict[str, Callable[..., str]]) -> Dict[str, RuleSpec]:
    return {
        "remove_punctuation": RuleSpec(
            name="remove_punctuation", fn=rules["remove_punctuation"],
            default_kwargs={}, bucket="punct",
        ),
        "add_random_punct": RuleSpec(
            name="add_random_punct", fn=rules["add_random_punct"],
            default_kwargs={"p": 0.10, "max_inserts": 3}, bucket="punct",
        ),
        "smart_delete_words": RuleSpec(
            name="smart_delete_words", fn=rules["smart_delete_words"],
            default_kwargs={"p": 0.10, "max_deletes": 3}, bucket="word",
        ),
        "swap_phrases_spacy": RuleSpec(
            name="swap_phrases_spacy", fn=rules["swap_phrases_spacy"],
            default_kwargs={"n_ops": 1}, bucket="word",
        ),
        "add_stopwords": RuleSpec(
            name="add_stopwords", fn=rules["add_stopwords"],
            default_kwargs={"p": 0.15, "max_inserts": 3}, bucket="word",
        ),
        "homoglypgs_random": RuleSpec(
            name="homoglypgs_random", fn=rules["homoglypgs_random"],
            default_kwargs={"p": 0.02}, bucket="char",
        ),
        "swap_chars_random": RuleSpec(
            name="swap_chars_random", fn=rules["swap_chars_random"],
            default_kwargs={"p": 0.01, "min_len": 4}, bucket="char",
        ),
        "insert_random_char": RuleSpec(
            name="insert_random_char", fn=rules["insert_random_char"],
            default_kwargs={"p": 0.004}, bucket="char",
        ),
        "double_random_letter": RuleSpec(
            name="double_random_letter", fn=rules["double_random_letter"],
            default_kwargs={"p": 0.04}, bucket="char",
        ),
        "delete_random_letter": RuleSpec(
            name="delete_random_letter", fn=rules["delete_random_letter"],
            default_kwargs={"p": 0.10}, bucket="char",
        ),
        "keyboard_miss": RuleSpec(
            name="keyboard_miss", fn=rules["keyboard_miss"],
            default_kwargs={"p": 0.03}, bucket="char",
        ),
        "replace_equivalents": RuleSpec(
            name="replace_equivalents", fn=rules["replace_equivalents"],
            default_kwargs={"p": 0.01, "allow_multi": True}, bucket="char",
        ),
        "roberta_synonyms": RuleSpec(
            name="roberta_synonyms", fn=rules["roberta_synonyms"],
            default_kwargs={"p": 0.15, "max_replacements": 3}, bucket="word",
        ),
        "grammar_distort": RuleSpec(
            name="grammar_distort", fn=rules["grammar_distort"],
            default_kwargs={"p": 0.12, "p_contr": 0.25}, bucket="grammar",
        ),
    }

