# Adversarial Attacks on AI-Text Detectors

Дипломная работа. Исследование adversarial-атак на детектор AI-сгенерированных текстов **RADAR**.

---

## Структура проекта

```
adversarial_ai_detection/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── broken_multilevel_dataset/       # Многоуровневый датасет с атаками (weak/medium/strong)
│   └── gemma_rewrite_dataset_effective/ # Датасет для дообучения Gemma-rewriter
│
├── notebooks/
│   └── experiments.ipynb                # Все эксперименты
│
└── src/
    ├── detector.py       # Детектор RADAR: radar_score()
    ├── metrics.py        # perplexity(), human_readability_score()
    ├── framework.py      # RuleSpec, TextBreakFramework, build_rulebook()
    └── rules/
        ├── __init__.py   # RULES — словарь всех правил
        ├── text_utils.py # normalize_spacing, tokenize, detokenize
        ├── punct.py      # Пунктуационный уровень
        ├── word_level.py # Словесный уровень
        ├── char_level.py # Символьный уровень
        ├── synonyms.py   # Синонимы (RoBERTa fill-mask)
        └── grammar.py    # Грамматический уровень
```

---

## Быстрый старт

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

```python
from src.detector import radar_score
from src.metrics import perplexity, human_readability_score
from src.rules import RULES
from src.framework import TextBreakFramework, build_rulebook

# Инициализация фреймворка
fw = TextBreakFramework(
    rulebook=build_rulebook(RULES),
    radar_score_fn=radar_score,
    perplexity_fn=perplexity,
    readability_fn=human_readability_score,
    min_readability=0.90,
    max_ppl_ratio=6.0,
)

# Атака на один текст
text = "Artificial intelligence has rapidly evolved..."
result = fw.generate_best_candidate(text, tries=16, strength=0.5, seed=42)

print(f"Исходный radar_ai:   {result['base']['radar_ai']:.3f}")
print(f"Изменённый radar_ai: {result['cand']['radar_ai']:.3f}")
print(f"Применённые правила: {result['combo']}")
```

---

## Описание правил

### Пунктуационный уровень (`punct`)
| Правило | Описание |
|---|---|
| `remove_punctuation` | Удаляет запятые, точки с запятой, двоеточия и т.д. |
| `add_random_punct` | Вставляет случайные знаки препинания |

### Словесный уровень (`word`)
| Правило | Описание |
|---|---|
| `smart_delete_words` | Удаляет наречия, прилагательные, AI-клише |
| `swap_phrases_spacy` | Переставляет синтаксические фразы (требует spaCy) |
| `add_stopwords` | Вставляет дискурсивные маркеры (Actually, Well, So…) |
| `roberta_synonyms` | Заменяет слова на контекстуальные синонимы (RoBERTa) |

### Символьный уровень (`char`)
| Правило | Описание |
|---|---|
| `homoglypgs_random` | Латиница ↔ Кириллица (визуально похожие буквы) |
| `swap_chars_random` | Перестановка соседних букв в слове |
| `insert_random_char` | Вставка случайных букв |
| `double_random_letter` | Удвоение случайной буквы |
| `delete_random_letter` | Удаление случайной буквы (не первой) |
| `keyboard_miss` | Замена на ближайшую клавишу QWERTY |
| `replace_equivalents` | Фонетические замены (c↔k, f↔ph, x↔ks…) |

### Грамматический уровень (`grammar`)
| Правило | Описание |
|---|---|
| `grammar_distort` | Сокращения, смена числа, артикли, модальные глаголы |

---

## Датасеты

| Датасет | Путь | Описание |
|---|---|---|
| MAGE | `yaful/MAGE` (HuggingFace) | Смешанный датасет human/machine текстов |
| WikiText-103 | `wikitext-103-v1` (HuggingFace) | Человеческие тексты для проверки правил |
| broken_multilevel_dataset | `data/broken_multilevel_dataset/` | Аугментированный датасет (weak/medium/strong атаки) |
| gemma_rewrite_dataset | `data/gemma_rewrite_dataset_effective/` | Датасет для дообучения Gemma-rewriter |

---

## Детектор RADAR

- **Модель:** [`TrustSafeAI/RADAR-Vicuna-7B`](https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B)
- **Класс 0:** AI-generated
- **Класс 1:** Human-written

---

## Основные результаты

### Сравнение стратегий атаки (n=30, машинные тексты)

| Стратегия | RADAR drop ↑ | Semantic sim ↑ | PPL ratio ↓ |
|---|:---:|:---:|:---:|
| **rules-only** | **+0.073** | **0.955** | **2.06** |
| gemma-only | −0.269 | 0.819 | 2.29 |
| rules → gemma | −0.273 | 0.808 | 2.32 |
| gemma → rules | −0.213 | 0.772 | 3.91 |

### Дообучение RADAR

| Модель | Clean F1 | Attacked F1 | F1 drop |
|---|:---:|:---:|:---:|
| Base RADAR | 0.690 | 0.696 | — |
| Clean-only FT | 0.866 | 0.862 | −0.004 |
| **Mixed FT** | **0.887** | **0.891** | **+0.004** |

### Held-out Attack Generalization

| Модель | Seen F1 drop | Unseen F1 drop |
|---|:---:|:---:|
| clean-only | 0.061 | 0.022 |
| **seen-attacks mixed** | **−0.006** | **−0.002** |

> Adversarial training на подмножестве правил обеспечивает защиту даже от новых, ранее не встречавшихся атак.

### IMGTB бенчмарк (лучшие результаты)

| Method | Clean AUC | Attacked AUC | AUC Drop |
|---|:---:|:---:|:---:|
| LogRankMetric | 0.728 | 0.676 | 0.052 |
| roberta-base-openai | 0.781 | 0.571 | **0.210** |
| **ModernBERT-base FT** | **0.985** | **0.977** | **0.009** |
