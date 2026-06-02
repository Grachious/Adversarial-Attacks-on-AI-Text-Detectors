from .text_utils import (
    WORD_RE, TOKEN_RE,
    normalize_spacing,
    tokenize_words_punct,
    detokenize,
)
from .punct import remove_punctuation, add_random_punct
from .word_level import smart_delete_words, swap_phrases_spacy, add_stopwords
from .char_level import (
    homoglypgs_random,
    swap_chars_random,
    insert_random_char,
    double_random_letter,
    delete_random_letter,
    keyboard_miss,
    replace_equivalents,
)
from .synonyms import roberta_synonyms
from .grammar import grammar_distort

RULES = {
    "remove_punctuation": remove_punctuation,
    "add_random_punct": add_random_punct,
    "smart_delete_words": smart_delete_words,
    "swap_phrases_spacy": swap_phrases_spacy,
    "add_stopwords": add_stopwords,
    "homoglypgs_random": homoglypgs_random,
    "swap_chars_random": swap_chars_random,
    "insert_random_char": insert_random_char,
    "double_random_letter": double_random_letter,
    "delete_random_letter": delete_random_letter,
    "keyboard_miss": keyboard_miss,
    "replace_equivalents": replace_equivalents,
    "roberta_synonyms": roberta_synonyms,
    "grammar_distort": grammar_distort,
}

__all__ = ["RULES"]
