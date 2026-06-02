# Повышение устойчивости детекторов машинно-сгенерированного текста к состязательным атакам

Дипломная работа посвящена исследованию устойчивости детекторов машинно-сгенерированного текста к состязательным атакам. Целью работы является выяснение того, как различные типы состязательных текстовых искажений влияют на качество детекции сгенерированного текста, а также разработка методов повышения устойчивости детекторов посредством состязательного обучения. 

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
│   └── gemma_rewrite_dataset_effective/ # Датасет для дообучения Gemma
│
├── notebooks/
│   └── experiments.ipynb                # Все эксперименты
│
└── src/
    ├── detector.py       # Детектор RADAR
    ├── metrics.py        # perplexity(), human_readability_score()
    ├── framework.py      # RuleSpec, TextBreakFramework, build_rulebook()
    └── rules/
        ├── __init__.py   # RULES — словарь всех правил
        ├── text_utils.py # Дополнительные функции для обработки текстов
        ├── punct.py      # Пунктуационный уровень
        ├── word_level.py # Словесный уровень
        ├── char_level.py # Символьный уровень
        ├── synonyms.py   # Синонимы (RoBERTa fill-mask)
        └── grammar.py    # Грамматический уровень
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
| broken_multilevel_dataset | `data/broken_multilevel_dataset/` | Аугментированный датасет (weak/medium/strong атаки) |
| gemma_rewrite_dataset | `data/gemma_rewrite_dataset_effective/` | Датасет для дообучения Gemma-rewriter |

---
